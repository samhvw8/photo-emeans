---
type: Benchmark Result
title: e-means and embedding comparison — Mac M1 Pro and Ubuntu RTX 3060
description: faiss contrib/e_means (NumPy, torch CPU/MPS/CUDA with native and Triton kernels) vs the Rust/candle port vs the Bend 2 port on synthetic d=768 data, plus EmbeddingGemma 2 image throughput (candle vs sentence-transformers), measured on the busy Mac and the idle Ubuntu GPU box.
resource: bench/compare
tags: [benchmark, emeans, faiss, bend, candle, metal, cuda, triton, embeddinggemma]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T12:45:00Z'
sources:
  - id: step5
    resource: bench/compare/results/encoder-step5/
    title: Encoder step 5 interleaved idle-box session (2026-11-12 17:32-17:43), 3 rounds x 5 cells x batch 1, 8
  - id: results
    resource: bench/compare/RESULTS.md
    title: Full results (per-run long tables with IQRs, objective/ARI/memory matrices, conditions, rerun steps)
  - id: runs
    resource: bench/compare/results/
    title: run.json per run, summary.json and table.md per host, embed/ runs, ubuntu-checks/ (2026-11-12)
  - id: faiss-emeans
    resource: https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means
    title: faiss contrib/e_means at the pinned commit (fetched by bench/compare/fetch_upstream.sh)
---

# Overview

A fair comparison of three e-means implementations — faiss's Python `contrib/e_means`, the Rust/candle
port in `crates/emeans` ([e-means algorithm](/design/emeans-algorithm.md)) and the Bend 2 port
([Bend e-means](/design/bend-emeans.md)) — on the Mac and on the Ubuntu box (an RTX 3060 Linux machine), plus an
EmbeddingGemma 2 throughput check ([EmbeddingGemma 2 family](/design/embeddinggemma2.md)).[^results]

Setup: spherical Gaussian mixtures, d = 768 (n = 16k, 100k, 200k; 1M on the box), k = 64, 256, 1024,
`nepochs` 5, batch law auto, seeds 1-5 for every impl. One shared evaluator computes the final
objective (mean 2 - 2 max cos, float64) and ARI vs `py-numpy` from each run's centroids. Train time
excludes loading and one-time device setup (untimed in-process warm-up). One warm-up run per impl and
case, then impls interleaved round-robin per seed, fresh process per run.[^runs]

**Superseded for the Rust GPU rows:** since 2026-11-12 the Rust backends train with the core layer's
device loop. See [core engine results](/results/core-engine.md): Metal 1.21 s and CUDA 0.29 s (FP32)
/ 0.15 s (BF16) at 200k x 1024. The tables below are the earlier host-loop sweep.

# Train time (s, median of 5 seeds)

Mac M1 Pro, 16 GB, busy (load average 12-97, median 21; the 200k x 1024 row ran at load 60-97):

| n | k | bend-cpu | bend-metal | py-numpy | py-torch-cpu | rs-cpu | rs-metal-fused |
|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.37 | 0.63 | 0.11 | 0.22 | 0.09 | 0.12 |
| 16,000 | 256 | 1.23 | 1.32 | 0.18 | 0.24 | 0.13 | 0.25 |
| 16,000 | 1024 | 4.02 | 3.92 | 0.35 | 0.31 | 0.19 | 0.33 |
| 100,000 | 64 | 1.61 | 2.57 | 0.62 | 0.65 | 0.31 | 0.36 |
| 100,000 | 256 | 6.07 | 11.83 | 0.86 | 0.74 | 0.35 | 0.58 |
| 100,000 | 1024 | 23.66 | 47.43 | 1.87 | 1.15 | 0.74 | 1.46 |
| 200,000 | 64 | 3.31 | 4.88 | 1.32 | 1.06 | 0.64 | 0.54 |
| 200,000 | 256 | 12.56 | 22.99 | 1.66 | 1.23 | 0.73 | 0.99 |
| 200,000 | 1024 | 65.08 | 89.31 | 5.02 | 3.48 | 2.34 | 2.89 |

`py-torch-mps` fails on every case: faiss keeps statistics in float64, which MPS does not support.

Ubuntu box, 2x Xeon E5-2683 v4 (64 threads), RTX 3060 12 GB, otherwise idle (`-f32` = native FP32
operands; without it faiss converts FP32 CUDA input to BF16):

| n | k | bend-cpu | bend-cuda | py-numpy | py-torch-cpu | torch-cuda-native | torch-cuda-native-f32 | torch-cuda-triton | torch-cuda-triton-f32 | rs-cpu-blocked | rs-cpu-candle-gemm | rs-cuda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.30 | 0.69 | 0.47 | 0.24 | 0.03 | 0.03 | 0.03 | 0.03 | 0.50 | 0.75 | 0.19 |
| 16,000 | 256 | 0.95 | 1.10 | 0.81 | 0.34 | 0.05 | 0.05 | 0.03 | 0.04 | 1.58 | 1.03 | 0.32 |
| 16,000 | 1024 | 2.98 | 3.25 | 1.33 | 0.44 | 0.05 | 0.05 | 0.03 | 0.05 | 1.90 | 1.39 | 0.35 |
| 100,000 | 64 | 1.39 | 2.13 | 2.10 | 0.58 | 0.07 | 0.07 | 0.06 | 0.07 | 1.00 | 1.87 | 0.43 |
| 100,000 | 256 | 4.54 | 5.26 | 7.99 | 0.82 | 0.08 | 0.09 | 0.06 | 0.08 | 2.47 | 2.98 | 0.41 |
| 100,000 | 1024 | 17.56 | 18.77 | 22.76 | 1.82 | 0.13 | 0.19 | 0.10 | 0.17 | 9.06 | 11.75 | 0.78 |
| 200,000 | 64 | 2.59 | 3.72 | 4.72 | 1.06 | 0.11 | 0.11 | 0.10 | 0.11 | 1.63 | 3.39 | 0.64 |
| 200,000 | 256 | 8.97 | 10.25 | 7.11 | 1.75 | 0.12 | 0.14 | 0.10 | 0.13 | 4.45 | 5.90 | 0.69 |
| 200,000 | 1024 | 34.40 | 36.41 | 12.03 | 3.74 | 0.20 | 0.34 | 0.15 | 0.31 | 16.53 | 22.84 | 1.20 |
| 1,000,000 | 1024 | 167.58 | 175.27 | 57.55 | 16.84 | 0.79 | 1.51 | 0.60 | 1.38 | 76.21 | 111.54 | 5.04 |

Objectives, ARI and peak memory per cell are in `bench/compare/RESULTS.md`.[^results]

# Embedding throughput (EmbeddingGemma 2, 64 images end to end, images/s)

| Host | Impl | dtype | batch 1 | batch 8 | cos vs f32 reference (min) |
|---|---|---|---|---|---|
| Mac | candle Metal | f32 | 1.43 | 1.45 | 1.000000 |
| Mac | sentence-transformers MPS | f32 | 0.98 | out of memory (11.9 GB footprint, swapping) | 1.000000 |
| Mac | sentence-transformers CPU | f32 | 0.16 | 0.21 | 1.000000 |
| Ubuntu | candle CUDA | f32 | 3.24 | 3.28 | 1.000000 |
| Ubuntu | sentence-transformers CUDA | f32 | 2.85 | 3.17 | 1.000000 |
| Ubuntu | candle CUDA | bf16 | 3.80 | 3.87 | 0.999615 |
| Ubuntu | sentence-transformers CUDA | bf16 | 5.84 | 9.27 | 0.999791 |
| Ubuntu, encoder step 5 | candle CUDA + FlashAttention-2 (vision tower) | bf16 | 9.67 | 10.31 | 0.999816 |

The last row is the [encoder recipe](/recipes/adding-an-encoder-family.md)'s step 5, measured later the
same day on the idle box in an interleaved session (median of 3 rounds) that also re-ran the other
CUDA cells: candle f32 3.23 / 3.27, sentence-transformers f32 3.00 / 3.16, sentence-transformers bf16
5.41 / 9.25, candle bf16 without flash 3.79 / 3.86.[^step5]

# Findings

- **Quality is equal.** All impls reach the same objective within seed noise (impl means within 0.4%
  at k = 64 and 1024, up to 3% at k = 256 where the seed sd is 0.4-2.5%). Bend and torch behave like
  other seeds of the same algorithm: their ARI vs `py-numpy` equals the seed-to-seed ARI of `py-numpy`.
  The Rust port repeats faiss's random draws (ARI 0.99-1.00 at k = 64). At 1M, the BF16 CUDA runs ended
  0.8% lower than FP32 (one case, ~2.5 sd).
- **Mac:** `rs-cpu` (Accelerate/AMX) is fastest in 8 of 9 cases, 1.2-2.5x faiss NumPy. The fused Metal
  kernel loses to it in training (per-batch overhead), although it is the fastest assignment kernel.
  Bend is 4-32x slower than `rs-cpu`, slower still on Metal.
- **Ubuntu GPU:** faiss Triton BF16 is fastest (0.15 s at 200k x 1024; FP32 0.31 s). The unfused
  `rs-cuda` backend of this sweep took 1.20 s (3.6-3.9x Triton FP32). The follow-up (fused CUDA kernel,
  device-side loop) reaches 0.29 s FP32 and 0.15 s BF16: [core engine results](/results/core-engine.md).
- **Ubuntu CPU:** torch CPU is fastest (3.7 s at 200k x 1024). faiss NumPy is slower with 64 OpenMP
  threads than with 16 (7.96 s vs 4.01 s at 100k x 256). Rust on x86 lacks a BLAS kernel: the blocked
  kernel is scalar code off aarch64 (16.5 s), candle GEMM 22.8 s. Since then the x86 default is
  `cpu-fused` (fused AVX2 kernel): 1.83 s, 6.92 s at 1M ([CPU kernels](/results/cpu-kernels.md)). Bend's CUDA path is no faster than
  its CPU path (SM mean 12%; `--gpu off` was faster).
- **Embedding:** candle beats sentence-transformers in f32 on both hosts. In bf16 on CUDA, PyTorch won
  while candle's vision attention built an explicit score matrix; with FlashAttention-2 (encoder step
  5) candle leads at both batch sizes: 9.67 vs 5.41 images/s at batch 1 (1.8x), 10.31 vs 9.25 at
  batch 8 (1.11x), with less GPU memory (0.98 vs 1.85 GiB at batch 8).

[^results]: `bench/compare/RESULTS.md`: tables, conditions, versions, rerun commands.
[^step5]: `bench/compare/results/encoder-step5/`: `ubuntu-idle-r{1,2,3}-{flash,st,noflash}/embed/*/run.json` (load average before each run) and the session log.
[^runs]: `bench/compare/results/<host>/runs/*/*/s*/run.json` and `summary.json`.
