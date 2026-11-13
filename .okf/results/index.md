# results

* [Mac vs Ubuntu comparison](compare-mac-ubuntu.md) - faiss Python vs Rust/candle vs Bend 2 e-means on CPU, Metal and CUDA, plus EmbeddingGemma 2 throughput.
* [CubeCL spike](cubecl-spike.md) - CubeCL 0.11.4 fused nearest-centroid kernels vs our Metal kernel, faiss Triton and candle CUDA; exactness, build cost, recommendation (keep hand-written kernels).
* [Core engine](core-engine.md) - device-loop training: Metal vs CPU on the M1 Pro, CUDA FP32/BF16 vs faiss Triton on the RTX 3060, timer definitions, parity and cross-backend exactness.
* [Original vs new core](original-vs-core.md) - faiss NumPy/torch/CUDA/Triton vs the rebuilt Rust core, fair interleaved run on the RTX 3060 plus a light Mac check.
* [Metal kernels](metal-kernels.md) - second Metal pass: GPU-timestamp profile, top-1 tile sweep, 9-cell sweep vs rs-cpu and faiss NumPy, exactness and the real-library check.
* [Video test run](video-test-run.md) - a test set of about 500 fetched and embedded videos: download size and time per video, Mac embed speed, how videos cluster with photos, full-library extrapolation.
* [CPU kernels](cpu-kernels.md) - CPU training loop profiled per op on the 2x Xeon box and the M1 Pro; the fused AVX2 `cpu-fused` kernel (x86 default), the physical-core pool, Accelerate tuning; final sweeps vs faiss NumPy and torch CPU; bit-identical to the old CPU and CUDA runs.
* [CUDA kernels](cuda-kernels.md) - second CUDA pass on the RTX 3060: nsys profile, exact FP32 top-1 through an FP16 tensor-core screen (bit-equal to the FP32 kernel), pipelined BF16 kernel, segmented scatter, host overlap; final sweep vs faiss Triton (200k / 1M x 1024: FP32 0.16 / 0.62 s, BF16 0.10 / 0.39 s).
* [Improvements over faiss](improvements-over-faiss.md) - one page: what emeans keeps from the original e-means and what it adds (backends, device loop, auto-k, dedup, incremental training with stable ids, outlier flag, embeddings), with numbers and limits.
* [Split vs baseline](split-vs-baseline.md) - after the library split: byte-identical results on every backend, same CUDA/Metal/CPU speed; the x86 slowdown came from candle's CUDA linked into the bench binary, fixed.
