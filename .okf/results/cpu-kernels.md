---
type: Benchmark Result
title: CPU kernels — the fused AVX2 top-1 on x86, Accelerate tuning on the Mac, small-op threading
description: Profile of the CPU training loop per op on the 2x Xeon box and the M1 Pro, the new `cpu-fused` kernel (packed GEMM with the argmax fused per tile, AVX2+FMA, no dependency), the physical-core pool, the Accelerate changes, final sweeps against faiss NumPy and torch CPU, and exactness (bit-identical to the old CPU and CUDA runs).
resource: crates/emeans/src/assign/fused.rs
tags: [benchmark, emeans, cpu, avx2, accelerate, rayon, exactness]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T20:35:00Z'
sources:
  - id: ops
    resource: bench/compare/results/ubuntu-cpu2-ops-sweep.log
    title: ops_bench on the box (one 16,384-row batch, n 200k, d 768), thread scaling 1/8/16/32/64, tile sweep, NUMA interleave, 2026-11-12 18:43-19:02
  - id: redo
    resource: bench/compare/results/ubuntu-cpu2-redo.log
    title: Redo of the box runs that overlapped another job (18:45-18:50 ops runs, 19:56-19:57 run.py cell), idle box, 2026-11-12 20:42-20:45
  - id: box-runs
    resource: bench/compare/results/ubuntu-cpu2/
    title: Box sweep, run.py --host ubuntu-cpu2 (py-numpy, py-torch-cpu 19:29-19:55; rs-cpu-fused, rs-cpu-blocked rerun 19:56-20:04 after the pool fix), logs ubuntu-cpu2-final.log, ubuntu-cpu2-rerun.log
  - id: fair
    resource: bench/compare/results/ubuntu-fair/
    title: The fair run before this work (rs-cpu-blocked, py-torch-cpu, rs-cuda; same box, same harness), 2026-11-12 17:56-18:38
  - id: mac-runs
    resource: bench/compare/results/mac-cpu2/
    title: Mac sweep, run.py --host mac-cpu2 --impls rs-cpu,py-torch-cpu,py-numpy, 3 seeds, 2026-11-12 20:25-20:29 (load 11-18, swap 14.8/16 GB)
  - id: mac-ab
    resource: crates/emeans/examples/cpu_probe.rs
    title: Mac probe (gather, sgemm, argmax, tiled and pipelined variants) and an interleaved old-vs-new emeans-compare A/B at 200k, 2026-11-12
  - id: parity
    resource: bench/data/results/
    title: cpu2-cpu/ and cpu2-cpu-fused/ (examples/parity.rs, 7 faiss parity cases, seeds 1-5) vs core-cpu-accelerate/
---

# Overview

The CPU backend of the [core](/design/core.md) had no fast x86 kernel: on the box (2x Xeon E5-2683 v4,
32 cores / 64 threads, AVX2, no AVX-512) `rs-cpu-blocked` ran scalar code (16.5 s at 200k x 1024)
while faiss torch CPU (oneDNN/MKL) took 3.7 s.[^fair] This work adds `cpu-fused`, the x86 default,
and tunes the Mac's Accelerate path and the small per-batch ops. Every result is bit-identical to
the CPU runs before it, and `cpu-fused` reproduces the CUDA FP32 backend exactly.

# Profile: where a training step goes

`examples/ops_bench.rs` times each op for one 16,384-row batch (n 200k, d 768); the ops add up to the
`loop` time (one training run), so they are the profile. `perf` is not available to users on the box
(`perf_event_paranoid` 4).[^ops]

| Op, k 1024 | Box, `cpu-blocked`, 64 thr (before) | Box, `cpu-fused`, 32 thr (after) | M1 Pro, Accelerate (before) | M1 Pro (after) |
|---|---|---|---|---|
| `top1` (rows by id) | 265-267 ms | 22-26 ms | 17-18 ms (gather 1.4, sgemm 11.1, scalar argmax 3.2) | 13.8-14.6 ms |
| `scatter_add` | 5.4 ms | 2.7-3.2 ms | 1.4-1.8 ms | 1.4-2.4 ms (noise) |
| `normalize_rows` | 1.6 ms | 0.4-0.5 ms | 0.4 ms | 0.4 ms |
| `ema_update` | 0.66 ms | 0.5-0.8 ms | 0.12 ms | 0.12 ms |
| training loop, 65 steps | 16.5 s | 1.62-1.79 s | 1.24-1.51 s | 1.03-1.33 s |

The "before" box column's small ops come from the first `cpu-fused` run (old small-op code, 64 threads);
they are the same code for every kernel. At k = 64 on the box, `cpu-fused` top-1 is 3.8-4.2 ms and scatter 2.4-2.9 ms: the scatter is then
40% of a step.

# Thread scaling on the box

`cpu-fused`, k 1024, one batch (`RAYON_NUM_THREADS`):[^ops]

| Threads | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| `top1` | 407-418 ms | 60-61 ms | 33-34 ms | 23-26 ms | 24-25 ms |
| GFLOP/s per core | 62 | 53 | 49 | 33-36 | — |
| Busy-core clock | 3.0 GHz | 2.55 GHz | 2.22 GHz | 1.71 GHz | — |
| FLOP per cycle per core | 21 | 21 | 22 | 20 | — |
| loop (65 steps) | — | 4.02 s | 2.39 s | 1.68-1.79 s | 1.88-2.13 s |

The kernel runs at ~21 of the 32 FLOP per cycle a Broadwell core can issue (65%) at every thread
count; the per-core drop is the AVX clock falling from 3.0 to 1.7 GHz as cores join. Hyperthreads
add nothing to the FMA-bound kernel and slow every small op (more threads to wake): 32 threads beat
64 in the loop (1.68-1.72 vs 1.88-1.91 s at k 1024, 0.56 vs 0.74 s at k 64, interleaved). The scalar
`cpu-blocked` kernel is the opposite (top-1 265-267 ms on 64 threads, 472-473 ms on 32), so only `cpu-fused` runs its
top-1 on the physical-core pool.

The tile sweep (mc 48-192, nc 64-256, kc 128-768) moved top-1 between 21.8 and 30.9 ms with
run-to-run noise of about 3 ms: no setting beat the default 96 x 128 x 256 reliably. `numactl
--interleave=all` did not help (24.0-26.2 ms).[^ops]

Runs redone: two other jobs overlapped my box timing (a CUDA build 18:46-18:49, one 19:56-19:57).
Every ops run from 18:45-18:50 (`cpu-fused` 8 and 16 threads, `cpu-blocked` and `cpu-candle-gemm` at
k 1024 on 64 and 32 threads) and the 200k x 256 Rust cell of the final sweep were rerun on an idle box
(no other process above 50% CPU before any run); the tables use the redone values.[^redo]

# Final sweeps

**Box** (otherwise idle; ps showed no other job; the 1-minute load average of 7-60 is the benchmark's
own 32-64 threads), train s, median of 3 seeds (IQR in `table.md`):[^box-runs]

| n | k | py-numpy | py-torch-cpu | rs-cpu-blocked (old) | rs-cpu-blocked | **rs-cpu-fused** | fused vs torch |
|---|---|---|---|---|---|---|---|
| 200,000 | 256 | 7.56 | 1.98 | 4.57 | 4.51 | **0.77** | 2.6x |
| 200,000 | 1024 | 12.83 | 3.73 | 16.83 | 16.52 | **1.83** | 2.0x |
| 1,000,000 | 1024 | 63.29 | 18.31 | 76.32 | 76.06 | **6.92** | 2.6x |

"Old" is the fair run before this work.[^fair] Goals met: at or below torch CPU (3.74 s and 16.8 s).

**Mac** (M1 Pro), train s, median of 3 seeds:[^mac-runs]

Not a quiet machine: load average 11.2-18.3 (median 12.7) before the runs, and 14.8 of 16 GB swap in
use by other sessions, so the absolute times are 1.5-5x the calm Mac's and the spread is wide. Use the
same-sweep columns and the A/B below, not the absolute values. "Old" is the [core engine](/results/core-engine.md)
sweep (load 7.7-14.5, 5 seeds).

| n | k | py-numpy | py-torch-cpu | rs-cpu (this sweep) | rs-cpu old | rs-cpu vs best faiss, same sweep |
|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.14 | 0.20 | 0.10 | 0.07 | 1.4x |
| 16,000 | 256 | 0.16 | 0.22 | 0.08 | 0.08 | 2.0x |
| 16,000 | 1024 | 0.32 | 0.31 | 0.15 | 0.13 | 2.1x |
| 100,000 | 64 | 0.51 | 0.73 | 0.23 | 0.19 | 2.2x |
| 100,000 | 256 | 0.75 | 0.85 | 0.37 | 0.26 | 2.0x |
| 100,000 | 1024 | 1.83 | 1.49 | 0.93 | 0.62 | 1.6x |
| 200,000 | 64 | 1.86 | 1.70 | 1.95 (IQR 0.59) | 0.35 | 0.9x |
| 200,000 | 256 | 2.20 | 2.25 | 1.36 | 0.48 | 1.6x |
| 200,000 | 1024 | 3.42 | 2.38 | 2.10 | 1.28 | 1.1x |

The Mac goal (faster than 1.28 s at 200k x 1024) is shown by an interleaved A/B of the old and new
`emeans-compare` binaries under the same load (16-19), alternating per seed:[^mac-ab]

| 200k, k | old median | new median | new vs old | final objectives |
|---|---|---|---|---|
| 64 (9 seeds) | 0.626 s | 0.630 s | 0.99x | identical in 9 of 9 |
| 256 (9 seeds) | 0.827 s | 0.794 s | 1.04x | identical in 9 of 9 |
| 1024 (5 seeds) | 1.508 s | 1.199 s | 1.26x | identical in 5 of 5 |

On the calm Mac that is 1.28 s / 1.26 = ~1.0 s, in line with `ops_bench` (loop 1.03-1.12 s new vs
1.24-1.51 s old, interleaved). A quiet-Mac rerun of the 9-cell sweep is still to do.

# Exactness

| Check | Result |
|---|---|
| `cpu-fused` AVX2 vs portable `mul_add` kernel, several tile sizes, IP and L2 | same bits (unit tests in `assign/fused.rs`, run on the box) |
| Contract (`tests/contract.rs`), incl. `check_production` (16,384 x 768, k 1024 and 1000, duplicate centroids and rows) on `cpu-fused` and Accelerate | pass on the Mac and the box |
| `streaming_api_reproduces_train_bit_for_bit` (new one-pass scatter vs the streaming API's `add_rows`) | pass |
| faiss parity, 7 cases, seeds 1-5, on Accelerate and on `cpu-fused`[^parity] | 33 of 33 runs bit-identical to the old Accelerate table (final and every epoch objective); worst difference of means 0.43%, unchanged |
| Box: `rs-cpu-fused` vs the fair run's `rs-cuda` (FP32), same seeds | every epoch objective identical in 9 of 9 runs (200k k 256/1024, 1M k 1024)[^box-runs] |
| Box `rs-cpu-fused` vs the Mac's `rs-cpu` (Accelerate) | identical in 6 of 6 runs (200k, k 256 and 1024) |
| `rs-cpu-blocked` new vs old | same objectives per seed (its own float order; 4e-5 to 4e-3 from the others, near-tie flips) |
| Mac Accelerate old vs new (`emeans-compare`, 200k, k 64/256/1024) | final objectives identical in all 33 interleaved pairs[^mac-ab] |
| Mac `rs-cpu` sweep vs the old mac-core sweep, 9 cells x seeds 1-3 | every epoch objective identical in 27 of 27 runs[^mac-runs] |

Why `cpu-fused` matches the GPU bit for bit: each dot product is one sequential FMA chain over the
dimensions, which is what the CUDA FP32 kernel computes; Accelerate happens to give the same top-1
on these sets.

# New dependency and running cost

None. `cpu-fused` is ~200 lines of Rust (`std::arch` AVX2+FMA intrinsics, runtime detection, a
portable twin with the same bits); no BLAS, no system package, nothing installed on the box. The
physical-core pool is one rayon pool built once per process. Options not taken: OpenBLAS/MKL (a system
or vendored library to keep in step on two hosts) and the `gemm` crate as a direct dependency (it is
already compiled for candle, but `cpu-candle-gemm` ran 275-326 ms per 16k x 1024 batch on 32-64
threads against 22-26 ms for `cpu-fused`, and it writes the `rows x k` matrix).

[^ops]: `bench/compare/results/ubuntu-cpu2-ops-sweep.log`: `ops_bench --backend cpu-fused|cpu-blocked|cpu-candle-gemm --k 1024|64 --reps 5-9` with `RAYON_NUM_THREADS`, `EMEANS_CPU_TILES`, `numactl`; clocks read from `/proc/cpuinfo` during the run.
[^fair]: `bench/compare/results/ubuntu-fair/table.md` (on the box), the "old" column.
[^box-runs]: `bench/compare/results/ubuntu-cpu2/table.md`, `runs/`; logs `ubuntu-cpu2-final.log` (py rows, first fused/blocked runs) and `ubuntu-cpu2-rerun.log` (Rust rows after the pool fix, box clippy and tests).
[^mac-runs]: `bench/compare/results/mac-cpu2/table.md`, `runs/`, log `mac-cpu2.log` (load average per run in `run.json`).
[^mac-ab]: Interleaved runs of the old and new `emeans-compare` (same CLI), 200k, seeds 1-5 (k 1024) and 1-9 (k 64, 256), Mac load 16-19.
[^parity]: `bench/data/results/cpu2-cpu/`, `cpu2-cpu-fused/` (gitignored), `uv run bench/py/compare.py <dir>`.
[^redo]: `bench/compare/results/ubuntu-cpu2-redo.log` (load average and busy-process count before each run).
