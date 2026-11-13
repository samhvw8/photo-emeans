---
type: Design
title: Bend e-means
description: How bend/emeans.bend implements faiss e-means in Bend 2, how it runs in parallel, which backend wins on this Mac, its parity with the faiss reference, and the Bend limits it hit.
tags: [bend, e-means, parity, benchmark]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T08:00:00Z'
sources:
  - id: faiss-emeans
    resource: /reference/faiss_e_means/
    title: Vendored faiss contrib/e_means (base.py, e_means.py, e_means_spherical.py)
  - id: bend-guide
    resource: https://github.com/bendlang/bend/blob/v2.0.40/guide/GUIDE.md
    title: Bend 2 guide (v2.0.40)
  - id: bend-shaders
    resource: https://github.com/bendlang/bend/blob/v2.0.40/guide/SHADERS.md
    title: Shaders in Bend (the GPU cost model)
  - id: bend-wontfix
    resource: https://github.com/bendlang/bend/blob/v2.0.40/WONTFIX.txt
    title: Bend WONTFIX list
  - id: bend-comp
    resource: https://github.com/bendlang/bend/blob/v2.0.40/bend2/comp.ts
    title: Bend 2 compiler (Metal build and GPU heap span)
  - id: results-cpu
    resource: bend/results/full_cpu.log
    title: Full CPU comparison run, 2026-11-12
  - id: results-gpu
    resource: bend/results/d768_gpu.log
    title: 768-d GPU comparison runs (with d768_gpu_k256.log), 2026-11-12
---

# Overview

`bend/emeans.bend` is the second e-means: a Bend 2 port of the faiss
`NumpyEmeans` reference,[^faiss-emeans] next to the Rust/candle one in
`crates/emeans` (see [runtime choice](/decisions/runtime-choice.md)). It reads the
same `embed/embeddings.f32` file as the [pipeline](/design/pipeline.md) and writes
the same `labels.i32` and `centroids.f32` files, plus a `report.json`.

```bash
mise run bend:build     # bend emeans.bend -o build/emeans (also writes build/emeans.gpu)
bend/build/emeans <vectors.f32> <n> <d> <k> <nepochs> <seed> <spherical 0|1> <outdir> [gpu 0|1]
mise run bend:test      # small synthetic sets vs faiss; fails past a 2% gap or a recipe mismatch
uv run bend/compare.py --full --seeds 5 --json bend/results/full_cpu.json
```

Install: `mise use -g github:bendlang/bend@2.0.40` works as is. The release tarball
keeps `bin/bend` beside `bend2/`, and mise keeps that layout, so the binary finds
its Base library. The project pins it in `mise.toml` with `BEND_NO_TELEMETRY=1`.

# What it implements

Everything below follows `base.py` and `e_means.py`, not only the README.

| Part | Behaviour |
|---|---|
| Recipe | Batch law `auto` (quality when d <= 16, else throughput), equations 1-3, nearest power of two with the Python tie rule, throughput cap `round_pow2(n/2)`, quality cap `n // 2`, alpha0 clamped to [1e-4, 1] |
| Epochs | `_epoch_rows`: whole passes, a half-even rounded fractional prefix before the closing pass; fractional and sub-one `nepochs` work |
| Order | A fresh Fisher-Yates order per epoch, keyed by `(seed * 1000003 + epoch) mod 2^32` as `_epoch_seed` |
| Update | Sum-and-count EMA as `_apply_statistics`: usage decays every step, touched clusters get `mass = decayed + alpha n_j`, move by `alpha n_j / mass` (exactly to the mean when that is 1), and `alpha == 1` replaces the usage |
| Schedule | Cosine from alpha0 to alpha0 / 10 over the total steps |
| Revival | Every 5 updates, `usage < 0.1 * mean` (0 when d <= 2), donors drawn by Gumbel top-k on `log usage`, targets in index order, split by `1 -+ 1e-4` with alternating signs, half the usage each, identical pairs nudged one float up |
| Spherical | EMA positions stay raw; assignment uses their unit directions (`normalize_into`: prescale by max abs); objective `sum(2 - 2 cos)`; closing means normalized |
| Euclidean | Assignment by `x.c - |c|^2 / 2`; objective `|x|^2 - 2 x.c + |c|^2` |
| Closing | Means of the last pass's labels; centroids with no rows keep their position; rows the last pass skipped get label -1 |

Differences from the reference, by design:

- **Random numbers.** Bend has no NumPy `RandomState`. A counter-based 32-bit hash
  (lowbias32) drives the initial rows, the epoch orders and the Gumbel draws, so
  runs are reproducible per seed but not bit-equal to the Python runs. Parity is
  statistical: mean objective over seeds.
- **Precision.** Bend has F32 only (no F64, issue #1120).[^bend-wontfix] alpha0
  is computed in F32 from `F32.pow` (C `pow` in double, rounded to float), so it
  lands 0 to 2 ULP from `float32(python_alpha0)`. Batch size, steps per pass,
  epoch rows and total steps are equal in every case. The epoch objective is a
  Kahan sum of per-batch objectives; each batch objective is a pairwise sum up the
  fork tree; dot products use four running sums.

# Program structure

One file, about 1,700 lines, in these sections:

| Section | Defs |
|---|---|
| Loading | `load.go` reads 1 MiB chunks with `File.read_at`, turns each four bytes into a U32 and bit-casts it (`match u: case U32{w}: F32{w}`) into one flat `Array<F32>` of n x d cells (4 bytes each), then checks the file size and the float count |
| Recipe | `batch.size`, `alpha0`, `pow2.round`, `epoch.rows`, `alpha.at` |
| Assignment kernel | `atree` (fork tree over batch rows) -> `row.go` (flat loop over rows) -> `ctree` (optional fork over centroid blocks) -> `cen` (flat loop over centroids) -> `dot4` (flat loop, four sums) |
| Buckets (host) | `bucket`: a counting sort of the batch by label (counts, offsets, rows in batch order) |
| Update kernel | `utree` (fork tree over clusters) -> `u.one`: sum the cluster's rows, EMA, pass statistics, readout |
| Revival (host) | `revive`: threshold, scan, Gumbel argmax, split, then a full readout |
| Driver (IO) | `epochs` -> `steps` -> `step`; one IO step per update, so a `!` can run at the root of its step |
| Closing, output | `fz.go`, `write.u`, `write.f`, `report` |

Bend style the code follows: a value is destructured only when it is a parameter,
so every read of an array (`Array.get` returns the array and the cell) is passed
to a helper or to the next iteration as an argument. Loops are tail calls over a
`Nat` counter; those compile to native loops (`spin_*` in the C) with
`Array.get` as one `blk_read`.[^bend-guide]

# Parallelism

Bend's only parallel primitive is the parallel let, `a b = f(..) g(..)`, and a
`!` call sends a whole fork tree to the GPU.[^bend-guide] An `Array` has one
owner, so the lanes share arrays through `@unsafe` defs with `+` array
parameters (one block, many handles), as Bend's own histogram bench and
`array_atomic_fadd` test do. Each lane writes only its own cells:

- Assignment: lane = a range of batch rows; writes `labels[row]`; returns the
  objective, summed pairwise on the joins. On the CPU, `min(batch, 256)` leaves.
  On the GPU, one row per leaf, then each row forks over centroid blocks so rows
  x blocks fill about 2^14 lanes; the join keeps the higher score and the lower
  index on ties.
- Update: lane = a range of clusters (up to 256 leaves); writes that cluster's
  slots of centroids, unit directions, usage, batch sums, pass sums and counts.
- Host work stays sequential: loading, the order shuffle, bucketing, revival,
  closing.

The CPU and GPU runs follow the same per-lane order and pairwise joins: 29 of
30 paired runs printed the same objectives, and one differed in the last printed
digit (112901.67 vs 112901.68), probably float contraction in the Metal
compiler. Final objectives were identical in all 30.

# Backend

`bend -o` writes a C file, compiles it with clang for the CPU, and writes
`emeans.gpu`, a Metal binary archive that Bend builds at run time through the
Metal framework (`newLibraryWithSource`).[^bend-comp] The offline Metal compiler
(`xcrun metal`) is not needed; it is not installed on this Mac, and `!` works.

The CPU (C backend, 10 threads) is faster than the M1 Pro GPU for this kernel,
so the results table uses the CPU. On the GPU each lane runs a long serial dot
loop, which Bend's guide warns is slow per lane ("a P-core is ~75x a lane on
serial work").[^bend-shaders]

| Case (5 seeds each) | Bend CPU train s | Bend GPU train s |
|---|---|---|
| n20k_d768, k=16 | 0.14 ± 0.03 | 1.12 ± 0.25 |
| n20k_d768, k=64 | 0.44 ± 0.06 | 0.98 ± 0.30 |
| n20k_d768, k=256 | 1.38 ± 0.17 | 4.30 ± 1.54 |
| n100k_d768, k=16 | 0.72 ± 0.10 | 1.31 ± 0.22 |
| n100k_d768, k=64 | 2.56 ± 0.56 | 3.20 ± 0.58 |
| n100k_d768, k=256 | 8.07 ± 1.79 | 13.45 ± 1.65 |

GPU numbers are from `bend/results/d768_gpu.log` and `d768_gpu_k256.log`.[^results-gpu]
The first GPU attempt at n100k_d768, k=256 failed with
`kIOGPUCommandBufferCallbackErrorInnocentVictim`: another process's Metal job
(the gemma-embed benchmark, started the same minute) faulted the GPU and macOS
discarded Bend's command buffer. The rerun on an idle GPU passed. Bend does not
retry a discarded command buffer; the process exits with status 1.

# Results

Synthetic unit-sphere Gaussian mixtures (100 true clusters, Dirichlet sizes,
cosine to center ~0.75), from `bend/gen_data.py`. Spherical, `nepochs` 5, seeds
1-5, `batch_law` auto, Bend on the CPU backend, faiss-cpu 1.16.0 with 10 OpenMP
threads, 2026-11-12 on the busy 16 GB M1 Pro (load average 7-30 during runs, so
times are noisy).[^results-cpu]

"Last-pass obj" is `em.obj[-1]` (measured while the centroids move). "Final obj"
is every row against its nearest final centroid, computed by `compare.py` in
float64 for both implementations. Gap = Bend mean / faiss mean - 1. Bend wall
includes process start and reading the file; faiss train excludes loading.

| Dataset | k | Batch | alpha0 | Bend last-pass obj | faiss last-pass obj | Gap | Bend final obj | faiss final obj | Gap | Bend train s | Bend wall s | faiss train s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n20k_d16 | 16 | 128 | 0.177351 | 14821.7 ± 56.2 | 14844.6 ± 59.6 | -0.15% | 14777.6 ± 55.7 | 14799.7 ± 60.4 | -0.15% | 0.11 ± 0.03 | 0.27 ± 0.15 | 0.18 ± 0.01 |
| n20k_d16 | 64 | 128 | 0.0935107 | 8991.9 ± 29.9 | 9009.8 ± 20.8 | -0.20% | 8932.6 ± 29.6 | 8949.1 ± 21.1 | -0.18% | 0.13 ± 0.01 | 0.21 ± 0.01 | 0.19 ± 0.00 |
| n20k_d16 | 256 | 256 | 0.0920254 | 7274.3 ± 7.2 | 7267.2 ± 5.1 | +0.10% | 7169.2 ± 6.7 | 7162.4 ± 5.0 | +0.09% | 0.19 ± 0.01 | 0.27 ± 0.01 | 0.12 ± 0.00 |
| n20k_d768 | 16 | 1024 | 1 | 22558.3 ± 203.2 | 22617.3 ± 169.7 | -0.26% | 22490.0 ± 202.3 | 22548.7 ± 166.4 | -0.26% | 0.14 ± 0.03 | 0.74 ± 0.04 | 0.53 ± 0.01 |
| n20k_d768 | 64 | 1024 | 0.608014 | 11153.6 ± 95.3 | 11277.5 ± 157.8 | -1.10% | 11074.2 ± 98.6 | 11202.3 ± 151.3 | -1.14% | 0.44 ± 0.06 | 1.11 ± 0.14 | 0.55 ± 0.01 |
| n20k_d768 | 256 | 2048 | 0.598356 | 9204.8 ± 9.5 | 9203.8 ± 9.4 | +0.01% | 9080.8 ± 9.8 | 9079.4 ± 9.3 | +0.02% | 1.38 ± 0.17 | 1.98 ± 0.18 | 0.58 ± 0.01 |
| n100k_d16 | 16 | 512 | 0.299551 | 74385.2 ± 395.6 | 74276.1 ± 147.9 | +0.15% | 74303.5 ± 394.9 | 74194.0 ± 147.4 | +0.15% | 0.57 ± 0.18 | 0.75 ± 0.21 | 0.79 ± 0.23 |
| n100k_d16 | 64 | 512 | 0.157943 | 45673.0 ± 270.1 | 45607.1 ± 142.9 | +0.14% | 45564.5 ± 270.9 | 45499.8 ± 142.9 | +0.14% | 0.38 ± 0.05 | 0.54 ± 0.08 | 0.38 ± 0.09 |
| n100k_d16 | 256 | 1024 | 0.155434 | 37285.4 ± 26.0 | 37303.2 ± 28.0 | -0.05% | 37076.4 ± 24.5 | 37094.4 ± 27.9 | -0.05% | 0.66 ± 0.05 | 0.81 ± 0.04 | 0.28 ± 0.01 |
| n100k_d768 | 16 | 4096 | 1 | 113403.3 ± 682.1 | 113154.6 ± 1595.9 | +0.22% | 113322.6 ± 679.7 | 113072.8 ± 1595.6 | +0.22% | 0.72 ± 0.10 | 4.34 ± 0.66 | 2.92 ± 0.31 |
| n100k_d768 | 64 | 4096 | 1 | 56812.9 ± 472.9 | 56645.6 ± 782.4 | +0.30% | 56684.3 ± 472.9 | 56447.5 ± 761.8 | +0.42% | 2.56 ± 0.56 | 6.86 ± 2.48 | 3.75 ± 1.41 |
| n100k_d768 | 256 | 8192 | 1 | 46251.7 ± 25.4 | 46229.7 ± 29.1 | +0.05% | 46011.5 ± 32.3 | 45983.9 ± 44.8 | +0.06% | 8.07 ± 1.79 | 11.23 ± 2.19 | 3.11 ± 0.32 |

Findings:

- Every gap is within -1.14% .. +0.42%, inside one seed standard deviation of
  either side; the 2% gate passes in all 60 paired runs.
- Recipes match: batch size, steps per pass, epoch rows and total steps are
  equal in all 60 runs; alpha0 is 0, 1 or 2 F32 ULP from the reference.
- Revivals agree in count (for example 505 vs 523 and 822 vs 827 splits), and
  the share of last-pass labels equal to the nearest final centroid matches
  (0.97-1.00 for both).
- Also checked on fewer seeds: Euclidean mode, `nepochs` 2.5 and 0.5 (only the
  fractional prefix is labelled).
- Speed: faiss assigns with BLAS (AMX on the M1 Pro); Bend's scalar dot loop
  reaches roughly 25 GFLOP/s on 10 cores. faiss is 2-4x faster at d=768, k >= 64;
  Bend is as fast or faster at small k, where faiss's per-batch overhead
  dominates. Bend's wall time at n=100k, d=768 includes 1.5-3 s to read 307 MB
  through `File.read_at` byte lists.

# Limits hit

| Limit | Effect here | Reference |
|---|---|---|
| No F64 | alpha0 0-2 ULP off; F32 sums need Kahan and pairwise care | #1120 (SOON) |
| No U64 | `File.read_at` offsets are U32: input under 4 GiB; the program refuses n x d >= 2^30 | #1027 (SOON) |
| Arrays have one owner | Lanes share arrays only in `@unsafe` defs, so `bend --check-only` prints SOME PROOFS FAIL for 41 defs; `bend -o` still exits 0 | GUIDE "Arrays", changelog 2.0.22 (#885) |
| `!` runs on the GPU only at the root of an IO step | One IO step per update | #1065 |
| No computed `match` or destructuring of a computed pair | Every array read goes through a helper def; most of the 270 defs exist for this | GUIDE "Recursion and Termination" |
| Binder order | Parameters must be matched in declaration order, before any let | GUIDE |
| Array size | A power of two, at most 2^31 cells, 4 bytes per F32 cell; 100k x 768 takes a 2^27-cell (512 MiB) array | #779 |
| GPU heap | The Metal span is capped at 2 GiB in `comp.ts` (`gpu_span`) | bend2/comp.ts |
| Files are byte lists | `File.read_at` returns `List<U32>` of bytes, so loading costs about 5 ns per byte | Base |
| No CSE | Values used twice are bound once by hand | #812 |

No runtime bug was hit; every failure during development was a checker error
with a clear message.

[^faiss-emeans]: Vendored faiss contrib/e_means: `base.py` recipe and lifecycle, `e_means.py` NumpyEmeans.
[^bend-guide]: Bend 2 guide: parallel calls, arrays, flat loops, IO.
[^bend-shaders]: "Shaders in Bend": GPU lanes, serial work per lane, 4^7 leaves per bang.
[^bend-wontfix]: WONTFIX.txt lists F64 (#1120) and U64 (#1027) under SOON.
[^bend-comp]: `gpu_make` and `gpu_desc` compile `BEND_SRC` with `newLibraryWithSource` and serialize a binary archive.
[^results-cpu]: `bend/results/full_cpu.log`, run with `uv run bend/compare.py --full --seeds 5`.
[^results-gpu]: `bend/results/d768_gpu.log` (`--datasets n20k_d768 n100k_d768 --seeds 5 --gpu`) and `d768_gpu_k256.log` (the rerun of n100k_d768, k=256).
