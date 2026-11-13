---
type: Decision
title: Library split — reusable e-means crates in their own repo
description: The e-means engine moves out of the photo tool into its own repo, split into a candle-free core, per-backend crates and an algorithms crate, for faster builds and reuse by other projects.
tags: [architecture, crates, build-time, reuse, decision]
status: stable
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T05:00:00Z'
verified:
  - by: human:samhvw8
    at: '2026-11-12T13:30:00Z'
---

# Decision

The e-means engine becomes its own library repo (one repo per reusable library). This repo (the
photo tool) first depended on it by local path; it now uses git dependencies pinned by commit.

| Crate | Holds | Heavy dependencies |
|---|---|---|
| `emeans-core` | `core::Device` op contract, CPU backend and f64 reference, device-resident training loop, recipe, RNG, save/load | none (`rayon`, `serde`, `anyhow`) |
| `emeans-metal` | Metal backend | `objc2-metal` |
| `emeans-cuda` | CUDA backend | `cudarc` (no candle) |
| `emeans-cluster` | auto-k, dedup, incremental, kNN, metrics, synthetic data | none |
| `emeans` | facade with `metal` / `cuda` features, today's public API | — |

The candle-based baselines (`CpuKernel::CandleGemm`, `MetalMode::Unfused`, `cuda-unfused`) move to the
benchmark harness only. `crates/encoder` (needs candle) and the photo crates stay in this repo.

# Why

`crates/emeans` pulls about 163-180 dependencies, nearly all through candle, which only three side
paths use. A CPU-only consumer needs about 20. Splitting also isolates rebuilds: editing an algorithm
no longer recompiles backends.

# Checks

Measured before and after through `scripts/quiet-run.py`: clean build time and dependency count of
`emeans-core`, rebuild time after touching one algorithm file, and the full test suite plus the
contract tests on every backend unchanged. The Ubuntu box syncs both folders.

# Publishing (decided 2026-11-12)

| Repo | Holds |
|---|---|
| `github.com/samhvw8/emeans` (public) | workspace: `emeans-core`, `emeans-metal`, `emeans-cuda`, `emeans-cluster`, `emeans` facade |
| `github.com/samhvw8/candle-embeddings` (public) | `Encoder` trait, shared blocks, EmbeddingGemma 2 family |

One repo per library, not per crate: crates that change together share one version and one place for
issues. License `MIT OR Apache-2.0`; the faiss-derived parts keep faiss's MIT notice. No model weights
in either repo. The photo tool (`photos`, `photo-emeans`) stays in its own repo (published later as
`github.com/samhvw8/photo-emeans`).

Before the first push: no personal data (no `data/`, no photo counts or asset ids from the user's
library, no absolute home-directory paths), benchmark results only from synthetic data,
and a scan of the full history being pushed.

# Language bindings (planned, after publishing)

Python (PyO3 + maturin; inputs through the Python buffer protocol, `PyBuffer<f32>`, so NumPy is not a dependency: NumPy arrays, `array.array`, `memoryview` and `bytes` all work without a copy; outputs as plain buffers, with an optional NumPy helper when NumPy is installed), Node.js (napi-rs, `Float32Array`) and Go (a C ABI with a
cbindgen header, wrapped by cgo) over one small facade API: `train`, `assign`, `auto_cluster`, `dedup`,
the incremental model (`add`, `save`, `load`), and `encoder`'s `embed_images` / `embed_texts`. Design
the `emeans` facade for that surface: plain slices in, plain structs out, no generics in the public
API. Prebuilt packages first for macOS arm64 (CPU + Metal) and Linux x86 (CPU); CUDA builds from
source. Binding versions to be checked when the work starts; publishing to PyPI or npm needs the
user's go-ahead.

# Outcome (2026-11-12)

Done. Two local repos, each with fresh git history, not pushed yet:

| Repo | Holds |
|---|---|
| `emeans` | `emeans-core`, `emeans-metal`, `emeans-cuda`, `emeans-cluster`, the `emeans` facade, `emeans-bench` (not published; candle baselines behind `baselines`) |
| `encoder` | the `encoder` crate (since renamed `candle-embeddings`, repo `github.com/samhvw8/candle-embeddings`); `hf-hub` behind the off-by-default `download` feature |

Measured payoff (a local build-speed study, not published): a CPU-only consumer builds
clean in 7.7 s instead of 32.7 s, and the facade with Metal in 6.0–8.6 s with 29 dependencies instead
of 28.6–58.4 s with 151. The photo CLI's own edit loop does not get faster (every library edit still
pays ~7.5 s in `photo-emeans`), so the split is for reuse and other consumers, not for this app's
build time. The CLI's edit loop gains from the incremental `release-dev` profile and from keeping the
mbx cache above its free-disk minimum instead.

Published 2026-11-13 after the CUDA tests passed on the RTX 3060: https://github.com/samhvw8/emeans and https://github.com/samhvw8/candle-embeddings (public). Push gate (user, 2026-11-13): both repos stay local until the CUDA contract tests pass on the new
layout on the Ubuntu box (task `test:cuda` in the emeans repo's `mise.toml`). Commits
use the author `samhvw8 <12538214+samhvw8@users.noreply.github.com>` (set per repo), not the personal
email.

Changes against the plan above:

- `encoder` moved out too (the publishing decision), so `crates/` here keeps only `photos` and
  `photo-emeans`. Both libraries are dependencies in the root `Cargo.toml` (git, pinned by commit).
- The closed engine enum became an object-safe `core::Engine` trait with a blanket impl for every
  `Device`, so a backend in another crate gets the device loop. `Backend` lives in `emeans-cluster`
  (which depends on the backend crates) and has a `Custom` variant.
- `CpuKernel::CandleGemm` became `CpuKernel::Gemm` (the `gemm` crate, called as candle calls it; not
  built on macOS). `Backend::MetalUnfused` and `Backend::CudaUnfused` are bench baselines now; the
  CLI lost one `backend_name` match arm.
- Metal is chosen by target in both libraries; neither has a `metal` feature any more.
- `bench/compare/rs` depends on the library repos and on `emeans-bench` (baselines on by default), so
  every backend name of `run.py` still works. The fixture generators moved to the library.
- Since 2026-11-13 the candle CUDA baseline (`rs-cuda-unfused`) needs `--features cuda-baselines`.
  candle's CUDA links the CUDA libraries at load time, and that made `cpu-fused` 8-12% slower in the
  same binary, which first looked like a regression from this split. `box_build.sh` builds
  `emeans-compare` with `cuda` only and `embed-compare` with `cuda,embed`; see the library's
  `.okf/results/cpu-kernels.md` ("CUDA libraries in the process slow the CPU path").

Measured (plain cargo, fresh target dirs, two interleaved rounds, shared Mac):

| | Before | After |
|---|---|---|
| Clean debug build, `emeans` (Metal) → facade | 28.6-58.4 s, 151 deps | 6.0-8.6 s, 29 deps |
| Clean debug build, `emeans` no features → `emeans-core` | 20.7-22.7 s, 139 deps | 4.8-5.1 s, 19 deps |
| `encoder` dependencies | 395 | 220 (395 with `download`) |
| `photo-emeans` dependencies | 424 | 252 |

Re-cluster of the real data (about 16k rows, `cluster --auto --dedup --dedup-max-gap 600 --backend
cpu`) with the pre-split CLI (aabfef0) and the split one: `labels.i32`, `labels_l2.i32`, `dedup.i32`,
`dedup.json`, `outlier.u8`, `sim.f32` and both centroid files byte-identical; one level-1
Davies-Bouldin diagnostic differs by 1 ulp (f64 summation order); steady-state cluster time 28.0 s
both. The CUDA tests were not run (the box was off); they are compiled and linted for Linux.

# When

After the cpu-kernel, cuda-kernel and fable-fix work is reported, verified and committed, so the move
does not collide with their edits.
