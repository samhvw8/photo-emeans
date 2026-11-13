---
type: Design
title: e-means core — the op contract (layer 1) and the device-resident training loop (layer 2)
description: crates/emeans/src/core, the layer every e-means backend runs on — the Device op trait with its exactness contract and f64 reference, the training loop written once against it, which family uses which block, and the rule that a family names any block it skips.
resource: crates/emeans/src/core
tags: [emeans, core, architecture, metal, cuda, exactness]
status: deprecated
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T21:00:00Z'
sources:
  - id: tensorfold
    resource: https://github.com/ashhart/TensorFold/blob/main/docs/recipes/adding-a-zig-family.md
    title: TensorFold, "Adding a Zig family" — blocks built once in the core, families compose them, kernels per backend under an exactness contract
  - id: code
    resource: crates/emeans/src/core
    title: mod.rs (Device trait, host blocks), cpu.rs, metal.rs, cuda.rs, cuda_src.rs, reference.rs, contract.rs, train.rs
  - id: triton
    resource: bench/compare/.upstream/faiss_e_means/triton_assignment.py
    title: faiss contrib/e_means `_assign_kernel` (MIT), the fused Triton kernel the CUDA top-1 ports
  - id: tests
    resource: crates/emeans/tests/contract.rs
    title: Contract tests (every op on every backend) and the cross-backend training test
---

> **Moved (2026-11-12).** The code left this repo in the library split. The maintained version of this
> concept is `github.com/samhvw8/emeans/.okf/design/core.md` (paths and APIs updated there). This copy is kept as history and is not updated.

# Overview

The core keeps the e-means engine expandable the way TensorFold keeps its model engine:[^tensorfold]
each block is built once, the families compose the blocks, kernels exist per backend under one
exactness contract, and the [recipe](/recipes/adding-an-emeans-backend.md) records each speed step
with its measured gain and a "Don't bother" list.[^code]

| Layer | What | Where |
|---|---|---|
| 1. Op contract | `core::Device`: the ops a backend implements, on its own buffers | `core/mod.rs`; `core/cpu.rs` (`CpuAssigner`), `core/metal.rs` (`MetalCore`), `core/cuda.rs` + `core/cuda_src.rs` (`CudaCore`) |
| Reference | One f64 CPU implementation of every op, plain loops | `core/reference.rs` |
| Contract check | Runs every op on a backend against the reference and the CPU float order; `check_production` adds the production batch shape (16,384 x 768, k 1024 and 1000) | `core/contract.rs` (`check`, `check_production`), `tests/contract.rs` |
| 2. Device loop | The e-means loop written once against layer 1 | `core/train.rs`, called by `Emeans::train` |
| Host blocks | Decisions that stay on the host: revival plan (MT19937 draws), closing means | `core/mod.rs` (`plan_revival`, `apply_splits`, `closing_means`) |

Adding a backend = implementing `Device` and passing `contract::check`. The training loop, the
families and the tests then run on it unchanged. An `Assigner` hands its engine to training through
`Assigner::with_engine` (`Engine::Cpu`, `Engine::Metal`, `Engine::Cuda`).

# Layer 1: the op contract

| Op | Does | CPU | Metal | CUDA | Exactness |
|---|---|---|---|---|---|
| `top1` | Fused score + argmax over k centroids (inner product or squared L2), rows by id or dense; ties keep the lower id | macOS: Accelerate sgemm per 16k-row block + 8-lane argmax; x86-64: `cpu-fused` (packed 6 x 16 AVX2+FMA microkernel, argmax fused per tile, each dot one sequential FMA chain, physical-core pool); also blocked NEON, candle GEMM ([CPU kernels](/results/cpu-kernels.md)) | fused simdgroup-matrix kernel, software-pipelined, 64 x 128 x 32 tile (64 x 64 x 16 for small k or batches), centroid parts when a batch fills fewer than 48 threadgroups; rows chosen by id are first gathered into one contiguous block | exact FP32 through an FP16 tensor-core screen: candidates kept under a rigorous error bound, scored with the FP32 kernel's arithmetic (bit-equal to it), the rest on the fused FP32 register-tile kernel (faiss Triton `_assign_kernel` idea in CUDA C, NVRTC; centroid split); d % 8 == 0 and k >= 128, else the FP32 kernel alone ([CUDA kernels](/results/cuda-kernels.md)) | label = f64 reference label except near-ties (reference scores within `TIE_EPS` = 1e-5 of the score magnitude); score within `SCORE_EPS` = 1e-5; exact duplicates (bit-identical centroids) must give the lowest id |
| `topk` | Best kk rows by inner product (kNN, dedup) | Accelerate GEMM + rayon | fused GEMM + top-k kernel (`KNN_SOURCE`) | one warp per query (correct, not tuned) | same as `top1`, per rank; padding id `u32::MAX`, score -inf; a lower-id exact duplicate ranks first |
| `scatter_add` | Batch sums (`k x d`) and counts per label, rows added in row order; optional `acc += sums` (the closing pass's segmented sums) | counting sort, then one rayon pass per centroid (zero, add rows in row order, add into `acc`); columns split when k is small; small-op tasks of ~16k floats on the physical-core pool | stable counting sort by blocks of 256 rows (block ranks and counts, one scan of blocks x k, place), then one thread per (cluster, column) | segmented stable counting sort (a warp per 8 clusters and 2,048-row segment: count, scan of segments x k, place the data rows), then one thread per (cluster, column) with 8 row loads in flight | counts exact; sums bit-equal to adding the rows in row order in f32 |
| `ema_update` | faiss `update`: masses decay, `N += alpha n_j`, `c += (alpha n_j / N)(s_j / n_j - c)` | f32 loop | one kernel, a threadgroup per cluster (positions, then its mass) | two kernels (positions, then masses) | bit-equal to the CPU (no FMA contraction) |
| `split` | Apply a revival plan: donor and child get `c (1 -/+ 1e-4 s)`, half the mass each | `apply_splits` | one threadgroup per pair | one block per pair | bit-equal to the CPU, the zero-row nudge included |
| `row_norms`, `normalize_rows` | Squared norms (L2 readout); faiss `normalize_into` (spherical readout) | sequential f32 sums | one simdgroup per row; every lane adds the row's rounded squares in column order through `simd_shuffle` | one warp per row, lane 0 adds the staged squares | equal to the CPU (sequential f32 sum, IEEE divide and sqrt); f64 reference within 1e-5 |
| `bind`, `upload`, `zeros`, `write`, `read`, `flush` | Buffers; ops queue; only `read` waits | `Vec` (the data set is read in place) | shared-storage `MTLBuffer`s on its own command queue, hazard tracking on; `bind_shared` wraps the data's pages without a copy; the buffer itself holds the `Arc` (and every clone of it does), and the engine holds it for queued work until `sync`. `bind_data` always copies, also for a pointer and length already bound; `bind` copies unless the bound buffer holds an `Arc` (a copy is never served by address) | `CudaSlice` on one stream, event tracking off; `bind_data` always uploads a fresh copy, and `bind` serves it by pointer and length only while bound (the caller must not change the data until `unbind_data`) | — |

Why the exact ops are bit-equal and not only "close": a 1-ulp difference in a unit centroid flips
near-tie labels, and one flipped label changes a later revival draw. With tree-summed norms the Metal
run moved off the CPU's trajectory and the worst faiss gap grew from 0.43% to 0.52%; with the CPU's
float order it is back to the CPU table exactly (see [results](/results/core-engine.md)).

The kernels compile at runtime from source: Metal with `newLibraryWithSource` (the exact library with
`MTLMathModeSafe`, `#pragma METAL fp contract(off)`, `precise::divide`/`sqrt`), CUDA with NVRTC through
candle's cudarc (the exact module with `--fmad=false`).

**Opt-in BF16 (CUDA).** `Backend::CudaBf16` (`cuda-bf16`) runs `top1` on tensor cores
(`mma.sync.m16n8k16`, BF16 operands, FP32 accumulate; `cp.async` 3-stage pipeline, `ldmatrix`,
128 x 256 tile for k >= 512) on a BF16 copy of the data set made once per bound buffer and BF16
centroids converted per call.
It is not exact and does not pass `contract::check`; `tests/contract.rs`
`cuda_bf16_is_close_to_the_reference` bounds it instead (a label may differ only when the two f64
scores are within 1e-2). faiss's CUDA default is BF16; ours is FP32.

# Layer 2: the device loop

`core::train::run` runs one training run on any `Device`. The data set, the epoch's row order, labels
and scores, centroids, masses, batch sums and counts, and the closing sums stay on the device.

Per logical batch, queued without a host wait: readout (normalize the positions, or their norms for
L2) → `top1` on the batch's rows (in `assignment_chunk_size` pieces, as the streaming API does) →
`scatter_add` (with `acc` in the last epoch) → `ema_update`. The host computes `alpha` from the
cosine schedule (f64, cast like numpy) and passes it as a kernel parameter.

| Host sync | When | What moves |
|---|---|---|
| Revival check | every `CHECK_UNUSED_EVERY` (5) updates | masses (`k` floats) down; the (donor, dead) pairs up when clusters split |
| Epoch end | each epoch | labels and scores of the epoch down; a helper thread adds up the objective (the streaming API's f64 order), imbalance and last-pass labels while the next epoch is queued |
| Run end | once | positions, masses, closing sums down; closing means on the host |
| Epoch start | each epoch | the MT19937 permutation up (computed one epoch ahead on a helper thread; epoch 0's is drawn while `model.rs` draws the init permutation) |

The recipe is unchanged: batch size, alpha0, schedule, row order and revival draws are the streaming
API's. On the CPU backend the loop reproduces the streaming API bit for bit
(`tests/model.rs` `streaming_api_reproduces_train_bit_for_bit`) and the faiss fixtures
(`tests/reference.rs`).

# Families and the blocks they use

A family names any core block it skips, and why.

| Family | Entry | Uses | Skips |
|---|---|---|---|
| Plain training | `Emeans::train`, `train_from`, `train_resume` | layer 2, all layer-1 ops but `topk` | — |
| Automatic k | `auto::auto_cluster`, `Hierarchy` | layer 2 per model (shared engine); `top1` through `assign_rows` | host: k-means++ init, silhouette and the other metrics, average linkage of leaves (CPU, small) |
| Incremental | `Hierarchy::add`, `add_photos`, `full_retrain` | `top1` (`assign_rows`), layer 2 (`train_resume` from saved EMA state) | — |
| Near-duplicates | `dedup::dedup`, `knn`, `knn_self` | `topk` (GPU engines; CPU path for kk > 16) | layer 2 (no training) |
| Streaming API | `begin` / `accumulate` / `update` / `end_epoch` / `finalize` | the CPU blocks of layer 1 (`add_rows`, `ema_host`, `plan_revival`, `apply_splits`, `closing_means`) | the device loop: the caller supplies each batch, so the state is on the host |
| Unfused baselines | `Backend::MetalUnfused`, `Backend::CudaUnfused` | their own candle `assign` | the engine: no `with_engine`, training uses the streaming API (one sync per batch). Kept only as the comparison point |

Related: [e-means algorithm](/design/emeans-algorithm.md), [automatic k, incremental, dedup](/design/autok-incremental-dedup.md).

[^tensorfold]: TensorFold's family recipe; this concept maps its core/family split onto e-means.
[^code]: `crates/emeans/src/core/` and its tests, 2026-11-12.
