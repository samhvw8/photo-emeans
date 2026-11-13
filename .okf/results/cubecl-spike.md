---
type: Benchmark Result
title: CubeCL spike — one fused nearest-centroid kernel for Metal and CUDA
description: CubeCL 0.11.4 fused argmax(x . c_j) kernels (scalar FMA, cmma f32/tf32/bf16) vs our hand-written Metal kernel on the M1 Pro, and vs faiss Triton and candle CUDA on the RTX 3060, with exactness against an f64 reference, build cost and a recommendation.
resource: bench/spikes/cubecl-argmax
tags: [benchmark, cubecl, metal, cuda, triton, candle, assignment, decision-input]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T16:05:00Z'
sources:
  - id: spike
    resource: bench/spikes/cubecl-argmax/
    title: Spike crate (src/kernels.rs, src/main.rs, box/run.sh, box/triton_assign.py, box/tune.sh) and raw outputs in results/ (2026-11-12)
  - id: cubecl
    resource: https://github.com/tracel-ai/cubecl/tree/v0.11.4
    title: CubeCL v0.11.4 source (README, cubecl-core frontend/cmma.rs, cubecl-cpp metal/dialect.rs and cuda/mma/manual.rs, cubecl-metal runtime.rs), cloned to ~/workspace/oss/cubecl
  - id: cubecl-crates
    resource: https://crates.io/crates/cubecl/versions
    title: crates.io version list (0.11.4 latest stable, 2026-11-09; 0.12.0-pre.2)
  - id: cubecl-issues
    resource: https://github.com/tracel-ai/cubecl/issues
    title: Open issues on 2026-11-12 (186), the Metal/CUDA/cmma ones cited below
  - id: triton
    resource: bench/compare/.upstream/faiss_e_means/triton_assignment.py
    title: faiss Triton assignment (_assign_kernel, _launch_assignment, assign_rows) and e_means_torch.py BF16 cast
  - id: ours
    resource: crates/emeans/src/metal.rs
    title: Hand-written fused Metal kernel (FUSED_SOURCE) and examples/bench_assign.rs
---

# Question

Should emeans write its GPU kernels once in CubeCL instead of once per backend (Metal source compiled at
runtime, CUDA C through NVRTC)? The test case is the hottest primitive: for a row batch X (b x d, f32,
unit rows) and centroids C (k x d), output per row argmax_j x.c_j and its score, without a b x k matrix.

# Setup

- CubeCL **0.11.4** (latest stable, released 2026-11-09; 0.12.0-pre.2 exists).[^cubecl-crates] Runtimes
  used: native Metal (`metal-native`), wgpu with MSL (`metal`), CUDA (`cuda`). Not run: Vulkan/SPIR-V,
  ROCm/HIP, CPU.
- Two CubeCL kernels in `src/kernels.rs`:[^spike]
  - `assign_fma`: scalar FMA, 16 x 16 units, 64 x 64 register tile (4 x 4 per unit), k-steps of 16 in
    shared memory, running best in registers, shared-memory reduction at the end. Exact f32.
  - `assign_mma<A>`: `cmma` fragments (f32 8x8x8 on Metal; tf32 16x16x8 or bf16 16x16x16 on CUDA),
    4 planes, 64 x 64 tile. The accumulator is opaque, so each centroid tile is stored to shared memory
    and scanned by two units per row.
- Data: `bench/data/s768_200k.f32` (synthetic, 500 centers, seed 5; same sha256 on both hosts), d = 768.
  Centroids: rows `(j * 7919 + 13) % n`. b = 4,096 and 65,536 are row prefixes; "200k" is the full file
  in 65,536-row chunks bound on the device.
- Timing protocol, all implementations: one pass = upload centroids, launch per chunk, read labels and
  scores back; median of 7 after one warm pass. `bench_assign` (ours, candle) uploads row ids instead of
  centroids; otherwise the same.
- Exactness: f64 CPU reference; a label mismatch is a tie when the two exact scores differ by at most
  1e-5 (the `bench_assign` rule). "Non-tie" counts the rest.

# Mac M1 Pro (busy machine), ms

Two calm interleaved rounds (r1/r2); every run under the Mac GPU lock.

| b | k | ours, fused Metal | CubeCL cmma f32, native Metal | CubeCL cmma f32, wgpu-MSL | CubeCL scalar FMA, native Metal (single busy run) |
|---|---|---|---|---|---|
| 4,096 | 64 | 1.2 / 0.9 | 1.36 / 1.09 | 1.58 / 1.68 | 2.11 |
| 4,096 | 256 | 3.3 / 2.7 | 1.67 / 1.41 | 1.96 / 1.88 | 7.47 |
| 4,096 | 1024 | 6.1 / 5.6 | 5.07 / 4.73 | 5.72 / 5.67 | 8.30 |
| 65,536 | 64 | 5.7 / 5.0 | 5.24 / 4.52 | 5.89 / 4.81 | 29.90 |
| 65,536 | 256 | 15.0 / 14.5 | 15.55 / 15.19 | 15.77 / 15.70 | 45.69 |
| 65,536 | 1024 | 57.7 / 56.5 | 61.87 / 62.21 | 61.35 / 60.79 | 116.08 |
| 200k | 64 | 14.6 / 14.2 | 14.80 / 13.90 | 15.21 / 15.50 | 74.34 |
| 200k | 256 | 45.3 / 44.9 | 47.42 / 46.33 | 48.25 / 48.26 | 114.75 |
| 200k | 1024 | 177.2 / 175.7 (1,790 GFLOP/s) | 190.05 / 185.92 (1,692) | 188.72 / 187.57 | 355.05 |

- CubeCL `cmma` f32 is within 0-6% of our kernel on the large shapes (200k x 1024: 186 vs 176 ms),
  and faster at b = 4,096, k = 256. Both use `simdgroup_float8x8`; CubeCL pays a threadgroup-memory
  round trip for the scores because it cannot read fragment elements per lane.
- The scalar FMA kernel is 2x slower on Metal: Apple GPUs need the simdgroup matrix path.
- A first, noisy run (Mac busy) showed up to 3.9x swings in the same cells (`results/mac-*.md`).

# Ubuntu RTX 3060 (idle), ms

| b | k | CubeCL FMA f32 | CubeCL cmma tf32 | CubeCL cmma bf16 | faiss Triton f32 | faiss Triton BF16 | candle unfused f32 (ours, `cuda`) |
|---|---|---|---|---|---|---|---|
| 4,096 | 64 | 0.58 | 0.44 | 0.34 | 0.68 | 0.45 | 0.4 |
| 4,096 | 256 | 1.56 | 1.36 | 1.06 | 1.17 | 0.54 | 0.7 |
| 4,096 | 1024 | 5.85 | 5.20 | 3.81 | 3.66 | 1.00 | 2.2 |
| 65,536 | 64 | 2.77 | 2.09 | 1.46 | 5.07 | 1.89 | 5.0 |
| 65,536 | 256 | 10.43 | 7.98 | 5.53 | 9.87 | 2.03 | 8.1 |
| 65,536 | 1024 | 46.19 | 36.86 | 27.25 | 38.02 | 4.18 | 28.8 |
| 200k | 64 | 8.16 | 6.05 | 4.32 | 15.38 | 5.64 | 15.4 |
| 200k | 256 | 29.66 | 22.66 | 15.75 | 29.73 | 5.91 | 25.8 |
| 200k | 1024 | 113.90 | 88.47 (56.1 tuned) | 60.36 (38.7 tuned) | 115.80 | 12.04 (26 TFLOP/s) | 89.2 |

- **f32, exact:** the CubeCL FMA kernel equals faiss Triton FP32 (`input_precision="ieee"`, no tensor
  cores): 114 vs 116 ms at 200k x 1024, and 1.9x faster at k = 64.[^triton]
- **BF16:** faiss's default CUDA path casts rows and centroids to BF16 (cast included above). Triton is
  3.2x faster than the best CubeCL bf16 tile (12 vs 39 ms) and 5x faster than the first one. A 5-tile
  sweep (`box/tune.sh`) gave 1.6x; the rest of the gap is Triton's multi-stage async copies and
  register-level epilogue, not tile size.
- Triton with FP32 rows and BF16 centroids (`assign_rows` direct) fails on sm_86 for every shape:
  `OutOfResources: shared memory, Required: 131072, Hardware limit: 101376`.
- candle unfused (cuBLAS f32 + argmax) beats every fused f32 kernel at k = 1024 (89 ms) and loses at
  k = 64 (15.4 vs 8.2 ms).

# Exactness (200k x 1024; smaller shapes are subsets and show the same pattern)

| Implementation | Label mismatches vs f64 | Non-tie | Max score gap of a flipped label | Max score error |
|---|---|---|---|---|
| CubeCL FMA f32 (Metal, wgpu, CUDA), CubeCL cmma f32 (Metal), Triton FP32 | 6 | 0 | 6.3e-7 | 1.5e-6 |
| CubeCL cmma tf32 (CUDA) | 121 | 54 | 3.7e-5 | 1.0e-4 |
| CubeCL cmma bf16 and Triton BF16 (identical counts) | 890 | 809 | 4.1e-4 | 7.2e-4 |

Ours (Metal fused) and candle CUDA were not re-checked here; `bench_assign` only compares backends to
each other.

# Code size and build cost

| | Hand-written (today) | CubeCL |
|---|---|---|
| Kernel lines (non-blank, no comments) | MSL fused kernel: 198 (also does L2 and indexed rows) | FMA 139, cmma 177 (inner product, dense rows) |
| New dependencies | none (candle already compiles MSL at runtime) | 393 crates (`cubecl` + `metal-native`) |
| Clean release build | — | Mac 4 m 12 s (busy); box 1 m 47 s |
| Rebuild after a kernel edit | — | 2.2 s |
| Binary | — | 39.9 MB (Metal), 50.3 MB (Metal + wgpu), 114 MB (CUDA) |
| First launch of a kernel variant (JIT) | MSL compile at first use | 130-1,070 ms on Mac, 145-355 ms on CUDA |
| Runtime needs on CUDA | NVRTC from the toolkit | the same (`libnvrtc` 12.4); also built with rustup 1.99.1 (box default 1.88 not tried) |

# What CubeCL could not express (0.11.4)

- **Per-lane fragment access on Metal.** `cmma::Matrix` is opaque; the register-level `MmaDefinition`
  path exists for CUDA only (f16, bf16, tf32 at 16x8x{8,16}). Our kernel and Triton keep the running best
  in registers; the CubeCL kernel must store each 64 x 64 score tile to shared memory.[^cubecl]
- **f32 tensor-core MMA on CUDA.** Not a CubeCL gap (the hardware has only tf32), but it means the
  "same source" runs as cmma f32 on Metal and has to switch to FMA (exact) or tf32/bf16 (labels change)
  on CUDA. CubeCL reports it clearly at JIT: "the device doesn't support a 8x8 Accumulator cooperative
  matrix fragment of Float(F32)".
- **Backend rules leak through.** WMMA needs a leading dimension that is a multiple of 8 for 16-bit
  operands; padding 4 gave `cuEventSynchronize failed with status 716` and a poisoned device with no
  kernel name. Metal allows 32 KB of threadgroup memory, so tiles are per backend anyway.
- **Known open issues that touch this use:** Metal bf16 x bf16 -> f32 simdgroup matrices not registered
  (#1951); Metal cmma column-major store transposed (#1726); generated MSL differs between processes,
  defeating Metal's shader cache (#1878); CUDA launch over the local-memory limit is an anonymous
  error (#2108); wrong matmul results on wgpu (#2023).[^cubecl-issues]
- **No buffer sharing with candle found.** Only the wgpu runtime accepts an existing external device; a
  grep of the 0.11.4 runtime, CUDA and Metal crates found no API to wrap a candle Metal or CUDA buffer
  (not exhaustively verified). Using CubeCL means a second GPU runtime and memory pool next to candle,
  or moving the whole e-means core onto CubeCL buffers.
- **API churn.** 0.x releases with breaking minors: 0.10 (2026-05), 0.11 (2026-10), 0.12-pre (2026-11).
  0.11 changed kernel arguments to slices and the IR to pliron; the matmul crate moved to `cubek`.

# Recommendation

**A: keep hand-written kernels per backend for the fused assignment.** Not a split.

- On Metal, CubeCL gets within 0-6% of our kernel with no tuning. It does not beat it, and our kernel
  is already written.
- On CUDA, exact f32 is FMA-bound: a hand-written CUDA C FMA kernel should land near the same 114 ms
  as CubeCL and Triton FP32. The fast path is BF16, and there CubeCL's portable cmma kernel is 3.2x
  behind Triton. Closing that gap needs CUDA-only features (async copy pipelines, manual MMA with
  register access). Written in CubeCL, that is still a CUDA-specific kernel, so "write once" no longer
  holds.
- Cost after shipping: CubeCL adds 393 crates, 40-114 MB binaries, a second GPU runtime beside candle
  (no buffer sharing found) and a 0.x API that breaks about every five months. The hand-written path
  adds two ~200-line kernel sources and nothing that runs.
- A split (CubeCL for scatter-add, norms, EMA, top-k) takes on all of those costs for the ops that are
  cheapest to hand-write.

**Reopen B when:** a third GPU backend is needed (Vulkan or ROCm), or the e-means core leaves candle
entirely. Then one CubeCL `Device` implementation could replace several hand-written ones, and the
FMA/cmma kernels here are a working start.

**Uncertainty.** The CubeCL kernels got one tile sweep and no async copies, manual MMA or autotune; a
tuned CubeCL CUDA kernel could be much closer to Triton. The Mac numbers come from a busy machine (two
calm rounds shown). Vulkan, ROCm and the CPU runtime were not run. The CUDA C kernel that core-engine
writes has not been measured against these numbers yet.

# Rerun

```bash
cd bench/spikes/cubecl-argmax && export CARGO_TARGET_DIR=$PWD/target/spike
cargo build --release --features metal-native,wgpu-msl
<flock> /tmp/emeans-mac-gpu.lock target/spike/release/cubecl-argmax --runtime metal \
  --data ../../data/s768_200k.f32 --kernels fma,mma-f32 --ks 64,256,1024 --bs 4096,65536,all --reps 7
# Box: tmux new-session -d -s emeans-cubecl "bash bench/spikes/cubecl-argmax/box/run.sh ..."
```

macOS has no `flock(1)`; the runs used a Python `fcntl.flock` wrapper with the same lock semantics.

[^spike]: `bench/spikes/cubecl-argmax/src/kernels.rs`; raw tables in `results/` (`ab-*`, `mac-*`, `box-*`, `tune.log`).
[^cubecl-crates]: crates.io lists 0.11.4 as max stable (2026-11-09) and 0.12.0-pre.2 as max version.
[^triton]: `_launch_assignment` uses `_DIRECT_FP32_BLOCK_D = 32` and `input_precision="ieee"` for FP32; `_resolve_input_dtype` casts FP32 CUDA input to BF16 by default.
[^cubecl]: `supported_cmma_combinations_metal` registers 8x8x8 only (f16, bf16 acc bf16, f32); CUDA `supported_mma_combinations` (manual MMA) lists f16/bf16/tf32/i8 for sm >= 80.
[^cubecl-issues]: Issue numbers from `gh issue list -R tracel-ai/cubecl` on 2026-11-12.
