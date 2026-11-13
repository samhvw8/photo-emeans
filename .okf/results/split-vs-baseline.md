---
type: Benchmark Result
title: Library split vs baseline — speed and quality
description: After moving the engine to github.com/samhvw8/emeans (crates core/metal/cuda/cluster), every backend gives byte-identical centroids to the pre-split baseline; CUDA and Metal speed are unchanged; an apparent 8–12% x86 CPU slowdown came from candle's CUDA libraries linked into the benchmark binary, not from the split, and is fixed.
tags: [benchmark, split, regression, cuda, metal, cpu, quality]
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T01:40:00Z'
---

# Setup

200k x 768, k = 1024, seeds 1–3, the same `emeans-compare` harness. Baseline: the results of the
kernel work before the split (`ubuntu-cuda2`, `ubuntu-cpu2`, `mac-metal2-200k1024`). After: the split
library at commit `ba59a96` (`ubuntu-split`, `mac-split`). Quality is compared per seed: objective and
a byte comparison of `centroids.f32`.

# Quality — identical everywhere

| Backend | Seed 1 | Seed 2 | Seed 3 | Centroids |
|---|---|---|---|---|
| CUDA FP32 (RTX 3060) | 0.284480 | 0.284843 | 0.285353 | byte-identical to baseline, 3/3 |
| CUDA BF16 | 0.284782 | 0.285531 | 0.284955 | byte-identical, 3/3 |
| x86 CPU `cpu-fused` | 0.284480 | 0.284843 | 0.285353 | byte-identical, 3/3 |
| Metal (M1 Pro) | 0.284480 | 0.284843 | 0.285353 | byte-identical, 3/3 |
| Mac CPU (Accelerate) | 0.284480 | 0.284843 | 0.285353 | byte-identical, 3/3 |

The same seed gives the same objective on CPU, Metal and CUDA FP32, before and after the split.

# Speed

| Backend | Baseline | After split | Verdict |
|---|---|---|---|
| CUDA FP32 | 0.158–0.163 s | 0.154–0.163 s | same |
| CUDA BF16 | 0.097–0.098 s | 0.097–0.102 s | same |
| Metal | 0.73–0.74 s (load 9) | 0.66–0.67 s (load 20) | same or faster |
| Mac CPU, interleaved old/new (100k x 1024, 4 rounds x 3 seeds) | 596 ms median | 552 ms median | same (an earlier 1.31–1.51 s vs 1.07–1.20 s gap was load 20 vs 9) |
| x86 CPU `cpu-fused`, interleaved, 6 rounds | 1.776 s median | 2.087 s before the fix, **1.809 s after** | same after the fix |

# The x86 slowdown was not the split

The post-split benchmark binary was built with `--features cuda`, which also turned on candle's CUDA
for the candle baselines. candle-core 0.11.1 links its CUDA libraries at program start
(`dynamic-linking`: libcudart, libcuda, libnvrtc, libcurand, libcublas, libcublasLt). With those six
merely loaded, the rayon CPU kernel ran 10–15% slower on the 2-socket Xeon; `LD_PRELOAD` of the six
into a CPU-only build reproduced it (1.81 → 2.03 s), a single thread showed no gap. The library's own
`emeans-cuda` loads CUDA lazily and is not affected. Fix: candle's CUDA sits behind a separate
`cuda-baselines` feature of the bench crate, so `--features cuda` maps no CUDA library at start
(checked with `ldd`). Details: `github.com/samhvw8/emeans/.okf/results/cpu-kernels.md`.
