# e-means: faiss Python vs Rust/candle vs Bend 2, on a Mac and an Ubuntu GPU box

Run on 2026-11-12. Harness: this folder ([how to rerun](#rerun)). Data: synthetic only.

## What was compared

| Impl | What it is | Device |
|---|---|---|
| `py-numpy` | faiss `NumpyEmeans` (assignment by `faiss.knn`) | CPU |
| `py-torch-cpu` | faiss `Emeans` on CPU tensors, `assignment_backend="native"` | CPU |
| `py-torch-mps` | faiss `Emeans` on MPS tensors | Apple GPU: **fails**, see below |
| `py-torch-cuda-native` / `-triton` | faiss `Emeans` on CUDA tensors, PyTorch or Triton kernels; FP32 input is converted to BF16 operands (the faiss default) | NVIDIA GPU |
| `py-torch-cuda-native-f32` / `-triton-f32` | the same with `QuantizedEmeans(quantizer=None)`: native FP32 operands | NVIDIA GPU |
| `rs-cpu` | `crates/emeans`, CPU kernel Accelerate `cblas_sgemm` (macOS default) | CPU |
| `rs-cpu-blocked` / `rs-cpu-candle-gemm` | `crates/emeans` CPU kernels on Linux: hand-blocked (portable scalar code on x86) / candle's matmul | CPU |
| `rs-metal-fused` | `crates/emeans` fused Metal kernel (dots + argmax in one pass) | Apple GPU |
| `rs-cuda` | `crates/emeans` new `cuda` backend: candle CUDA matmul, then argmax (not fused) | NVIDIA GPU |
| `bend-cpu` / `bend-metal` / `bend-cuda` | `bend/emeans.bend` on Bend 2.0.40, C backend, `gpu` arg 0 or 1 | CPU / GPU |

The Python package is faiss `contrib/e_means` at the pinned commit `2b648f06`
(`fetch_upstream.sh` downloads it; its NumPy files are byte-identical to `reference/faiss_e_means`).

**Workload.** Spherical Gaussian mixtures from `bench/py/gen_data.py`, d = 768, raw LE f32:
`s768_16k` (n 16,000, 200 centers, seed 17), `s768_100k` (100,000, 400, seed 3), `s768_200k`
(200,000, 500, seed 5), and on the box `s768_1m` (1,000,000, 1000, seed 23). The files are
byte-identical on both hosts (sha256 checked). k ∈ {64, 256, 1024}, `spherical=True`,
`nepochs=5`, `batch_law="auto"`, seeds 1-5 for every impl.

**Metrics.**

- **Objective**: one shared function (`evaluate.py`) loads each run's final centroids and computes
  mean(2 - 2 · max_j cos(x, c_j)) over all rows in float64. No impl's own report is used.
- **Train s**: median over the 5 seeds (IQR in the long tables). Excludes data loading (reported
  apart as load s) and one-time per-process setup: every Python and Rust run first does an untimed
  in-process warm-up (Python: one epoch on the first max(4k, 4096) rows with the same class, k, backend
  and representation; Rust: one assignment of 4096 rows), which pays CUDA context, cuBLAS, Triton JIT,
  Metal pipeline and thread-pool start-up. Bend cannot be warmed in-process; its `train_ms` comes from
  its own timer around init + train + closing pass.
- **Peak memory**: `/usr/bin/time` max RSS for the process; on macOS also the peak physical
  footprint (includes Metal buffers); on Linux the process tree's peak GPU memory from `nvidia-smi`
  sampled every 0.2 s.
- **ARI**: adjusted Rand index of each run's nearest-final-centroid labels vs `py-numpy`'s labels for
  the same seed. To read it, use the **noise floor**: the ARI between two `py-numpy` runs with
  consecutive seeds. Two equally good clusterings agree only that much.

**Fairness.** Fresh process per run. Per (data, k): one warm-up run per impl (seed 1000, discarded),
then for each seed every impl once, the impl order rotated by one each seed. The 1-minute load average
is recorded before every run.

## Mac — M1 Pro (8P + 2E cores), 16 GB, macOS 27.0.1

Conditions: a shared, busy machine. Load average before runs: min 11.8, median 21.1, max 96.7 (other
agents' jobs). The 200k / k = 1024 cells ran at load 60-97 and are noisy (see the IQRs in the long
table). All GPU runs (rs-metal-fused, bend-metal, py-torch-mps, the Metal and MPS embedding runs) were
done first, one GPU job at a time; the CPU-only runs came after. `rs-metal-fused` was rerun on its own
after the sweep (32 runs) because its binary gained the in-process warm-up mid-sweep; it was not
interleaved with bend-metal for that rerun.

Threads: faiss OpenMP 10, torch 8, rayon 10, Bend pool 10. Versions: faiss-cpu 1.16.0, numpy 2.5.4,
torch 2.11.0, candle 0.11.1, Bend 2.0.40, Rust 1.99.1.

`py-torch-mps` fails at the start of training on every case:

```text
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
  e_means_torch.py:445 _start_epoch: self._pass_counts = self._usage.new_zeros(self.k, dtype=torch.float64)
```

faiss keeps pass counts, objectives and statistics in float64 on the device (10 sites in
`e_means_torch.py`). Making it run would change its numerics, not add a device shim, so it was not
patched. (One shim was needed for every torch run on macOS: faiss-cpu and torch each bundle `libomp`,
and loading both aborts with OMP Error #15; `run_py.py` puts a placeholder `faiss` module in place for
torch runs, which never call faiss.)

<!-- BEGIN mac:matrices -->
Train s, median of 5 seeds:

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

Objective, mean of 5 seeds:

| n | k | bend-cpu | bend-metal | py-numpy | py-torch-cpu | rs-cpu | rs-metal-fused |
|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.62725 | 0.62725 | 0.62737 | 0.62968 | 0.62741 | 0.62741 |
| 16,000 | 256 | 0.29193 | 0.29193 | 0.29430 | 0.29083 | 0.29534 | 0.29534 |
| 16,000 | 1024 | 0.26222 | 0.26222 | 0.26226 | 0.26220 | 0.26223 | 0.26223 |
| 100,000 | 64 | 0.95402 | 0.95402 | 0.95231 | 0.95455 | 0.95231 | 0.95231 |
| 100,000 | 256 | 0.36930 | 0.36930 | 0.37609 | 0.38045 | 0.37838 | 0.37838 |
| 100,000 | 1024 | 0.28244 | 0.28244 | 0.28242 | 0.28277 | 0.28253 | 0.28253 |
| 200,000 | 64 | 1.04646 | 1.04646 | 1.04689 | 1.04600 | 1.04689 | 1.04689 |
| 200,000 | 256 | 0.45797 | 0.45797 | 0.46262 | 0.45979 | 0.46433 | 0.46433 |
| 200,000 | 1024 | 0.28510 | 0.28510 | 0.28498 | 0.28519 | 0.28495 | 0.28495 |

ARI vs py-numpy (same seed), mean of 5 seeds:

| n | k | bend-cpu | bend-metal | py-numpy | py-torch-cpu | rs-cpu | rs-metal-fused |
|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.699 | 0.699 | 1.000 | 0.712 | 0.999 | 0.999 |
| 16,000 | 256 | 0.553 | 0.553 | 1.000 | 0.553 | 0.609 | 0.609 |
| 16,000 | 1024 | 0.201 | 0.201 | 1.000 | 0.199 | 0.356 | 0.356 |
| 100,000 | 64 | 0.397 | 0.397 | 1.000 | 0.395 | 1.000 | 1.000 |
| 100,000 | 256 | 0.893 | 0.893 | 1.000 | 0.881 | 0.948 | 0.948 |
| 100,000 | 1024 | 0.360 | 0.360 | 1.000 | 0.361 | 0.376 | 0.376 |
| 200,000 | 64 | 0.338 | 0.338 | 1.000 | 0.330 | 1.000 | 1.000 |
| 200,000 | 256 | 0.838 | 0.838 | 1.000 | 0.824 | 0.972 | 0.972 |
| 200,000 | 1024 | 0.422 | 0.422 | 1.000 | 0.425 | 0.433 | 0.433 |

ARI noise floor (py-numpy seed s vs seed s+1, mean of 4 pairs):

| n | k | ARI |
|---|---|---|
| 16,000 | 64 | 0.711 |
| 16,000 | 256 | 0.554 |
| 16,000 | 1024 | 0.196 |
| 100,000 | 64 | 0.398 |
| 100,000 | 256 | 0.880 |
| 100,000 | 1024 | 0.365 |
| 200,000 | 64 | 0.335 |
| 200,000 | 256 | 0.822 |
| 200,000 | 1024 | 0.424 |

Peak memory GiB, RSS / GPU (Linux, nvidia-smi) or footprint (macOS):

| n | k | bend-cpu | bend-metal | py-numpy | py-torch-cpu | rs-cpu | rs-metal-fused |
|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.10 / 0.10 | 0.11 / 0.11 | 0.12 / 0.10 | 0.26 / 0.20 | 0.11 / 0.10 | 0.17 / 0.16 |
| 16,000 | 256 | 0.11 / 0.10 | 0.11 / 0.11 | 0.13 / 0.10 | 0.27 / 0.21 | 0.11 / 0.11 | 0.17 / 0.16 |
| 16,000 | 1024 | 0.12 / 0.11 | 0.12 / 0.12 | 0.18 / 0.16 | 0.30 / 0.24 | 0.14 / 0.13 | 0.19 / 0.18 |
| 100,000 | 64 | 0.54 / 0.54 | 0.55 / 0.54 | 0.39 / 0.36 | 0.53 / 0.47 | 0.60 / 0.59 | 0.89 / 0.88 |
| 100,000 | 256 | 0.54 / 0.54 | 0.55 / 0.55 | 0.42 / 0.40 | 0.55 / 0.50 | 0.62 / 0.62 | 0.89 / 0.89 |
| 100,000 | 1024 | 0.56 / 0.55 | 0.56 / 0.56 | 0.47 / 0.45 | 0.62 / 0.57 | 0.66 / 0.65 | 0.90 / 0.90 |
| 200,000 | 64 | 1.04 / 1.04 | 1.05 / 1.04 | 0.71 / 0.69 | 0.84 / 0.79 | 1.18 / 1.18 | 1.75 / 1.74 |
| 200,000 | 256 | 1.04 / 1.04 | 1.05 / 1.05 | 0.76 / 0.74 | 0.88 / 0.82 | 1.19 / 1.19 | 1.75 / 1.75 |
| 200,000 | 1024 | 1.04 / 1.05 | 1.06 / 1.06 | 0.79 / 0.79 | 0.93 / 0.88 | 0.99 / 1.23 | 1.77 / 1.76 |
<!-- END mac:matrices -->

## Ubuntu box — 2x Xeon E5-2683 v4 (32C/64T), 94 GB, RTX 3060 12 GB (SM86)

Conditions: the box was otherwise idle (no other tmux session, no other compute process on the GPU;
only the desktop's gnome-shell and Xwayland, 86 MiB). Load average before runs: min 7.6, median 19.1,
max 49.5, all from this sweep's own multi-threaded runs. Ubuntu 26.04, kernel 7.0.0-43, driver
580.178.04. GPU runs and CPU runs were interleaved (one process at a time).

Threads: faiss OpenMP 64, torch 32, rayon 64, Bend pool 64. Versions: faiss-cpu 1.16.0, numpy 2.5.4,
torch 2.11.0+cu130 (CUDA 13.0 runtime), Triton 3.6.0, candle 0.11.1 with CUDA kernels built by nvcc
12.4 (sm_86), Bend 2.0.40 (CUDA via NVRTC 12.4, clang 19.1.7 for the C backend), Rust 1.99.1.

<!-- BEGIN ubuntu:matrices -->
Train s, median of 5 seeds:

| n | k | bend-cpu | bend-cuda | py-numpy | py-torch-cpu | py-torch-cuda-native | py-torch-cuda-native-f32 | py-torch-cuda-triton | py-torch-cuda-triton-f32 | rs-cpu-blocked | rs-cpu-candle-gemm | rs-cuda |
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

Objective, mean of 5 seeds:

| n | k | bend-cpu | bend-cuda | py-numpy | py-torch-cpu | py-torch-cuda-native | py-torch-cuda-native-f32 | py-torch-cuda-triton | py-torch-cuda-triton-f32 | rs-cpu-blocked | rs-cpu-candle-gemm | rs-cuda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.62725 | 0.62725 | 0.62776 | 0.63025 | 0.63560 | 0.63517 | 0.63560 | 0.63419 | 0.62740 | 0.62779 | 0.62780 |
| 16,000 | 256 | 0.29193 | 0.29193 | 0.29459 | 0.28963 | 0.29241 | 0.29247 | 0.29217 | 0.29307 | 0.29445 | 0.29475 | 0.29623 |
| 16,000 | 1024 | 0.26222 | 0.26222 | 0.26229 | 0.26224 | 0.26240 | 0.26234 | 0.26240 | 0.26228 | 0.26220 | 0.26234 | 0.26228 |
| 100,000 | 64 | 0.95402 | 0.95402 | 0.95231 | 0.95455 | 0.95572 | 0.95429 | 0.95572 | 0.95429 | 0.95231 | 0.95231 | 0.95231 |
| 100,000 | 256 | 0.36930 | 0.36930 | 0.37814 | 0.38120 | 0.37477 | 0.37382 | 0.37477 | 0.37424 | 0.37927 | 0.37794 | 0.37847 |
| 100,000 | 1024 | 0.28244 | 0.28244 | 0.28256 | 0.28261 | 0.28260 | 0.28244 | 0.28260 | 0.28240 | 0.28253 | 0.28248 | 0.28248 |
| 200,000 | 64 | 1.04646 | 1.04646 | 1.04689 | 1.04600 | 1.04846 | 1.04875 | 1.04846 | 1.04875 | 1.04689 | 1.04689 | 1.04689 |
| 200,000 | 256 | 0.45797 | 0.45797 | 0.46148 | 0.45952 | 0.45715 | 0.45742 | 0.45715 | 0.45621 | 0.46374 | 0.46179 | 0.46232 |
| 200,000 | 1024 | 0.28510 | 0.28510 | 0.28515 | 0.28513 | 0.28504 | 0.28514 | 0.28504 | 0.28532 | 0.28520 | 0.28505 | 0.28501 |
| 1,000,000 | 1024 | 0.30408 | 0.30408 | 0.30295 | 0.30351 | 0.30060 | 0.30337 | 0.30060 | 0.30405 | 0.30324 | 0.30294 | 0.30344 |

ARI vs py-numpy (same seed), mean of 5 seeds:

| n | k | bend-cpu | bend-cuda | py-numpy | py-torch-cpu | py-torch-cuda-native | py-torch-cuda-native-f32 | py-torch-cuda-triton | py-torch-cuda-triton-f32 | rs-cpu-blocked | rs-cpu-candle-gemm | rs-cuda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.696 | 0.696 | 1.000 | 0.709 | 0.703 | 0.700 | 0.703 | 0.700 | 0.994 | 0.996 | 0.993 |
| 16,000 | 256 | 0.568 | 0.568 | 1.000 | 0.554 | 0.548 | 0.564 | 0.547 | 0.564 | 0.608 | 0.605 | 0.612 |
| 16,000 | 1024 | 0.200 | 0.200 | 1.000 | 0.198 | 0.196 | 0.200 | 0.196 | 0.198 | 0.276 | 0.309 | 0.315 |
| 100,000 | 64 | 0.397 | 0.397 | 1.000 | 0.395 | 0.393 | 0.394 | 0.393 | 0.394 | 1.000 | 1.000 | 1.000 |
| 100,000 | 256 | 0.885 | 0.885 | 1.000 | 0.873 | 0.881 | 0.880 | 0.881 | 0.885 | 0.937 | 0.952 | 0.945 |
| 100,000 | 1024 | 0.360 | 0.360 | 1.000 | 0.363 | 0.362 | 0.358 | 0.362 | 0.364 | 0.373 | 0.378 | 0.375 |
| 200,000 | 64 | 0.338 | 0.338 | 1.000 | 0.330 | 0.337 | 0.336 | 0.337 | 0.336 | 1.000 | 1.000 | 1.000 |
| 200,000 | 256 | 0.837 | 0.837 | 1.000 | 0.830 | 0.847 | 0.841 | 0.847 | 0.842 | 0.977 | 0.976 | 0.974 |
| 200,000 | 1024 | 0.427 | 0.427 | 1.000 | 0.421 | 0.435 | 0.428 | 0.435 | 0.429 | 0.442 | 0.438 | 0.439 |
| 1,000,000 | 1024 | 0.675 | 0.675 | 1.000 | 0.671 | 0.680 | 0.673 | 0.680 | 0.678 | 0.702 | 0.704 | 0.702 |

ARI noise floor (py-numpy seed s vs seed s+1, mean of 4 pairs):

| n | k | ARI |
|---|---|---|
| 16,000 | 64 | 0.704 |
| 16,000 | 256 | 0.562 |
| 16,000 | 1024 | 0.196 |
| 100,000 | 64 | 0.398 |
| 100,000 | 256 | 0.867 |
| 100,000 | 1024 | 0.360 |
| 200,000 | 64 | 0.335 |
| 200,000 | 256 | 0.831 |
| 200,000 | 1024 | 0.425 |
| 1,000,000 | 1024 | 0.681 |

Peak memory GiB, RSS / GPU (Linux, nvidia-smi) or footprint (macOS):

| n | k | bend-cpu | bend-cuda | py-numpy | py-torch-cpu | py-torch-cuda-native | py-torch-cuda-native-f32 | py-torch-cuda-triton | py-torch-cuda-triton-f32 | rs-cpu-blocked | rs-cpu-candle-gemm | rs-cuda |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16,000 | 64 | 0.16 / - | 0.16 / 0.10 | 0.13 / - | 0.75 / - | 1.47 / 0.27 | 1.46 / 0.28 | 1.32 / 0.22 | 1.32 / 0.24 | 0.20 / - | 0.20 / - | 0.36 / 0.19 |
| 16,000 | 256 | 0.16 / - | 0.16 / 0.10 | 0.13 / - | 0.76 / - | 1.47 / 0.29 | 1.46 / 0.27 | 1.32 / 0.22 | 1.32 / 0.22 | 0.20 / - | 0.20 / - | 0.35 / 0.19 |
| 16,000 | 1024 | 0.17 / - | 0.17 / 0.10 | 0.15 / - | 0.75 / - | 1.47 / 0.33 | 1.44 / 0.32 | 1.32 / 0.26 | 1.32 / 0.28 | 0.20 / - | 0.22 / - | 0.36 / 0.19 |
| 100,000 | 64 | 0.60 / - | 0.59 / 0.10 | 0.39 / - | 0.91 / - | 1.47 / 0.57 | 1.46 / 0.53 | 1.32 / 0.49 | 1.32 / 0.50 | 0.68 / - | 0.68 / - | 0.68 / 0.44 |
| 100,000 | 256 | 0.60 / - | 0.60 / 0.10 | 0.41 / - | 0.95 / - | 1.47 / 0.58 | 1.46 / 0.60 | 1.32 / 0.56 | 1.32 / 0.54 | 0.68 / - | 0.68 / - | 0.68 / 0.47 |
| 100,000 | 1024 | 0.61 / - | 0.61 / 0.10 | 0.43 / - | 0.98 / - | 1.47 / 0.66 | 1.44 / 0.64 | 1.32 / 0.53 | 1.32 / 0.58 | 0.68 / - | 0.68 / - | 0.68 / 0.50 |
| 200,000 | 64 | 1.10 / - | 1.09 / 0.10 | 0.70 / - | 1.22 / - | 1.47 / 0.89 | 1.46 / 0.88 | 1.32 / 0.84 | 1.32 / 0.79 | 1.25 / - | 1.25 / - | 1.25 / 0.75 |
| 200,000 | 256 | 1.10 / - | 1.10 / 0.10 | 0.76 / - | 1.25 / - | 1.47 / 0.99 | 1.46 / 0.97 | 1.32 / 0.94 | 1.32 / 0.89 | 1.25 / - | 1.25 / - | 1.25 / 0.78 |
| 200,000 | 1024 | 1.11 / - | 1.11 / 0.10 | 0.79 / - | 1.30 / - | 1.47 / 1.10 | 1.44 / 1.03 | 1.32 / 0.97 | 1.32 / 0.93 | 1.25 / - | 1.25 / - | 1.25 / 0.81 |
| 1,000,000 | 1024 | 4.12 / - | 4.12 / 0.10 | 3.37 / - | 3.60 / - | 3.54 / 4.49 | 3.54 / 4.28 | 3.54 / 3.88 | 3.54 / 3.69 | 5.83 / - | 5.83 / - | 5.83 / 3.44 |
<!-- END ubuntu:matrices -->

## Embedding throughput — EmbeddingGemma 2 (`google/embeddinggemma-2` @ 914f7f89)

The 15 public test images in `bench/gemma/images` (scikit-image sample data and crops) cycled to 64,
end to end: file decode, preprocessing, model. One warm-up batch first, fresh process per case.
"cos vs ST f32" compares each case's embeddings of the 15 images with the first sentence-transformers
f32 case of the same host; "cos vs CPU ref" with `bench/gemma/ref/images.f32` (sentence-transformers,
CPU, f32).

Mac (Metal / MPS / CPU):

<!-- BEGIN mac:embed -->
| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | metal | f32 | 1 | 1.43 | 1.4 | 1.18 | 2.56 | 1.000000 | 1.000000 |
| candle | metal | f32 | 8 | 1.45 | 2.3 | 1.20 | 3.49 | 1.000000 | 1.000000 |
| sentence-transformers | cpu | f32 | 1 | 0.16 | 6.8 | 0.95 | 2.74 | 1.000000 | 1.000000 |
| sentence-transformers | cpu | f32 | 8 | 0.21 | 6.1 | 1.45 | 4.23 | 1.000000 | 1.000000 |
| sentence-transformers | mps | f32 | 1 | 0.98 | 7.3 | 1.51 | 4.71 | 1.000000 | 1.000000 |
| st:mps:f32 | | | 8 | failed: killed by hand after 16 min: peak memory footprint 11.9 GB on the 16 GB Mac, process swapping (RSS 4 MB, state U); not a | | | | | |
<!-- END mac:embed -->

The sentence-transformers MPS batch-8 case reached an 11.9 GB peak footprint on the 16 GB Mac, then
swapped (RSS 4 MB, state U); it was stopped by hand after 16 minutes. It is a memory result, not a speed
result. The CPU batch-8 case finished at 4.2 GB.

Ubuntu (CUDA):

<!-- BEGIN ubuntu:embed -->
| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |
|---|---|---|---|---|---|---|---|---|---|
| candle | cuda | bf16 | 1 | 3.80 | 1.4 | 1.02 | 1.60 | 0.999627 | 0.999627 |
| candle | cuda | bf16 | 8 | 3.87 | 1.5 | 1.02 | 3.35 | 0.999615 | 0.999614 |
| candle | cuda | f32 | 1 | 3.24 | 2.5 | 1.03 | 2.03 | 1.000000 | 1.000000 |
| candle | cuda | f32 | 8 | 3.28 | 1.4 | 1.02 | 3.06 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | bf16 | 1 | 5.84 | 5.9 | 2.10 | 1.08 | 0.999791 | 0.999791 |
| sentence-transformers | cuda | bf16 | 8 | 9.27 | 6.3 | 2.24 | 1.85 | 0.999847 | 0.999847 |
| sentence-transformers | cuda | f32 | 1 | 2.85 | 10.6 | 3.32 | 1.99 | 1.000000 | 1.000000 |
| sentence-transformers | cuda | f32 | 8 | 3.17 | 6.3 | 3.29 | 3.26 | 1.000000 | 1.000000 |
<!-- END ubuntu:embed -->

**Encoder step 5 (later the same day, idle box, interleaved, median of 3 rounds).** With
FlashAttention-2 on the vision tower (`crates/encoder`, feature `flash-attn`), candle bf16 runs 9.67
images/s at batch 1 and 10.31 at batch 8, against 5.41 / 9.25 for sentence-transformers bf16 and
3.79 / 3.86 for candle bf16 without it, in the same session. f32 is unchanged (candle 3.23 / 3.27,
byte-identical vectors; sentence-transformers 3.00 / 3.16). Cosine vs the CPU reference: 0.999816 with
flash, 0.999614 without. Runs: `results/encoder-step5/`; recipe: `.okf/recipes/adding-an-encoder-family.md`.

## What the numbers mean

**Quality is the same everywhere.** Every impl reaches the same objective within seed noise. At
k = 64 and k = 1024 the impl means differ by at most 0.4%. At k = 256 they differ by up to 3% (Bend
lowest at all three n); the per-seed sd there is 0.4-2.5%, so with 5 seeds this is at most a weak
signal. It comes from different random streams (initial rows, shuffles, revival donors), not from
arithmetic: Rust on Accelerate, the blocked kernel, candle, Metal and CUDA give the same objective for a
seed, and Bend gives the same objective on CPU, Metal and CUDA and on both hosts. BF16 operands on CUDA
(the faiss default) match FP32 within noise up to n = 200k; at n = 1M they ended 0.8% *lower*
(0.3006 vs 0.3034-0.3041, about 2.5 seed sd), which is a single case, not an explanation.

**ARI vs py-numpy reads against the noise floor.** Bend and the torch `Emeans` draw their own random
numbers, and their ARI equals the seed-to-seed floor in every case: they behave like another seed of
the same algorithm. The Rust port draws numpy's MT19937 stream, so it repeats faiss draw for draw:
ARI 0.99-1.00 at k = 64, and above the floor at larger k, where a near-tie in the dot products
(different summation order) eventually flips one label and the runs drift apart. At k = 1024 even
equally good clusterings agree at ARI ~0.2-0.45, because the data has 200-1000 true centers and many
equivalent ways to split them.

**Mac: the Rust CPU path wins.** `rs-cpu` (Accelerate on the M1's AMX units) is the fastest impl in 8
of 9 cases: 1.2-2.5x faster than faiss NumPy and 1.5-2.4x faster than torch on the CPU. The fused Metal
kernel wins only at n = 200k, k = 64, and loses to the CPU path elsewhere, although it is the fastest
*assignment* kernel on this Mac (see [e-means algorithm](../../.okf/design/emeans-algorithm.md)): a
training run is thousands of small batches, and per-batch costs (upload of centroids and row ids,
a device sync, the host-side EMA update) eat the kernel's gain. Bend is 4-32x slower than `rs-cpu`
on the CPU and slower again on Metal (its GPU lanes each run a long serial dot product). Bend also
spends 0.5-6 s reading the file (byte lists). faiss on MPS does not run at all (float64).

**Ubuntu: faiss Triton wins on the GPU, torch wins on the CPU.** On the RTX 3060, faiss's fused
Triton kernel with BF16 operands trains 200k x 1024 in 0.15 s (0.31 s in FP32); the new `rs-cuda`
backend takes 1.20 s, 4x the FP32 Triton time. `rs-cuda` is unfused (it writes and re-reads the
`batch x k` matrix) and syncs to the host each batch to read labels; a fused CUDA kernel and keeping
the EMA update on the device are the follow-ups. On the 64-thread Xeon, `py-torch-cpu` (oneDNN/MKL
GEMM) is the fastest CPU impl, 3.7 s at 200k x 1024. faiss NumPy is 3x slower there (12 s) and
sensitive to threads: at 100k x 256 it took 7.96 s with 64 OpenMP threads, 4.75 s with 32 and 4.01 s
with 16 (`results/ubuntu-checks/`), so the default 64 hyperthreads oversubscribe its small per-batch
searches. The Rust CPU kernels are weak on x86: `CpuKernel::Blocked` falls back to portable scalar
code off aarch64 (16.5 s) and candle's GEMM is slower still (22.8 s); a BLAS kernel (OpenBLAS or MKL)
for Linux is a follow-up. Bend on 64 threads is 34 s, and its CUDA path is no faster than its CPU
path: the GPU runs in bursts (SM utilization mean 12%, max 99%) and a control run with
`BENDRTS="--gpu off"` was faster (4.4 s vs 5.2 s at 100k x 256, same objective).

**n = 1M, k = 1024 (box only).** Same ranking as at 200k: Triton BF16 0.60 s, native BF16 0.79 s,
Triton FP32 1.38 s, native FP32 1.51 s, `rs-cuda` 5.04 s (3.6x Triton FP32), torch CPU 16.8 s, faiss
NumPy 57.6 s, Rust blocked 76.2 s, Rust candle GEMM 111.5 s, Bend CPU 167.6 s and Bend CUDA 175.3 s,
plus 50 s for Bend to read the 2.9 GB file. ARI rises to ~0.68-0.70 (floor 0.681): with 1000 true
centers and k = 1024 the clusters are close to the truth. Peak RSS: 3.4-4.1 GiB for faiss and Bend,
5.8 GiB for Rust (the second copy while reading); GPU 3.4-4.5 GiB.

**Memory.** Every impl stays under 2 GiB at n = 200k. Rust keeps a second copy of the data while
reading the file (RSS ~2x the 614 MB data set at 200k); `rs-metal-fused` adds the GPU copy (footprint
1.75 GiB). On CUDA, faiss holds the data plus BF16 or FP32 operands (~1 GiB at 200k); `rs-cuda` ~0.8
GiB. Bend's CUDA heap is CUDA managed memory, which `nvidia-smi` does not count per process, so its
0.10 GiB is a floor, not a measurement.

**Embedding.** On the Mac, the candle Metal port runs 1.43-1.45 images/s in f32, 1.5x
sentence-transformers on MPS (0.98) and 7-9x sentence-transformers on the CPU (0.16-0.21). All
outputs agree at cosine 1.000000 with the CPU f32 reference. Batch 8 does not speed up either
implementation; on MPS it does not fit in 16 GB. On the RTX 3060, gemma-embed (now `crates/encoder`) builds for CUDA with
only a feature flag (candle-nn's `cuda`, for the fused RMSNorm and RoPE kernels; its `sdpa` call falls
back to plain attention off Metal). In f32 candle is 4-14% faster than sentence-transformers (3.24 vs
2.85 images/s at batch 1, 3.28 vs 3.17 at batch 8), at cosine 1.000000. In bf16 sentence-transformers
wins clearly (5.84 images/s at batch 1, 9.27 at batch 8, vs 3.80-3.87 for candle). Likely cause
(not profiled): candle's non-Metal attention path builds the full vision attention matrix
(`nn.rs` falls back from `sdpa`), while PyTorch can use its fused SDPA kernels. bf16 costs cosine 0.9996 (candle) and 0.9998 (sentence-transformers)
against f32. Peak GPU memory 1.1-3.4 GiB.

**Follow-ups.** A fused CUDA assignment kernel and device-side EMA update for `rs-cuda`; a BLAS CPU
kernel for x86 Linux; read the data file without a second copy; cap faiss OpenMP threads on the box
for a fairer `py-numpy` row; run faiss on MPS only after upstream supports float32 statistics.

## Rerun

```bash
# Mac
bench/compare/fetch_upstream.sh
(cd bench/compare/py && uv sync)
CARGO_TARGET_DIR=target/bench cargo build --release --manifest-path bench/compare/rs/Cargo.toml --features metal,embed
uv run --python 3.12 bench/py/gen_data.py --name s768_16k --n 16000 --d 768 --centers 200 --seed 17   # and the others above
python3 bench/compare/run.py --host mac --impls rs-metal-fused,bend-metal --datasets s768_16k,s768_100k,s768_200k --ks 64,256,1024
python3 bench/compare/run.py --host mac --impls py-numpy,py-torch-cpu,rs-cpu,bend-cpu --datasets s768_16k,s768_100k,s768_200k --ks 64,256,1024
bench/compare/py/.venv/bin/python bench/compare/embed_run.py --host mac --cases candle:metal:f32,st:mps:f32,st:cpu:f32 --max-footprint-gb 7
bench/compare/py/.venv/bin/python bench/compare/evaluate.py bench/compare/results/mac
python3 bench/compare/report.py

# Ubuntu box (in tmux; see box_*.sh)
bash bench/compare/box_setup.sh      # Bend + clang via `mise install`, venv, builds, data
bash bench/compare/box_bench.sh      # the sweep
bash bench/compare/box_embed.sh      # embedding
```

Results: `results/<host>/runs/<data>_k<k>/<impl>/s<seed>/run.json` (centroids and labels are
gitignored), `results/<host>/summary.json`, `results/<host>/table.md` (long table with IQRs),
`results/<host>/embed/`.
