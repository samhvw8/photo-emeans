---
type: Design
title: Pipeline
description: The fetch, embed, dedup, cluster, gallery and albums steps for photos and videos, what each reads and writes, and the improvements over faiss e-means.
tags: [pipeline, architecture]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T12:40:00Z'
sources:
  - id: faiss-emeans
    resource: https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means
    title: faiss contrib/e_means at the pinned commit
  - id: eg2-config
    resource: https://huggingface.co/google/embeddinggemma-2/blob/main/config.json
    title: EmbeddingGemma 2 config
---

# Steps

| Step | Crate | Reads | Writes |
|---|---|---|---|
| fetch | `photos` | Photos library via PhotoKit | `photos/*.jpg`, `photos/manifest.jsonl` |
| fetch-videos | `photos` (PhotoKit + AVFoundation) | Photos library: a medium rendition per video | `videos/<uuid>/fNN.jpg` (poster + up to 7 sampled frames), `videos/manifest.jsonl` |
| embed | `candle-embeddings`, imported as `encoder` (family `embeddinggemma2`) | `photos/`, `videos/` | `embed/embeddings.f32`, `rows.jsonl` (`kind` photo or video), `meta.json`; frame vectors in `embed/frames/` |
| dedup | `emeans` | `embed/` | near-duplicate groups (one representative per group is clustered) |
| cluster | `emeans` (Rust) and `bend/` | `embed/` | `clusters/<run>/` |
| gallery | `photo-emeans` | `clusters/<run>/`, `photos/`, `videos/` | a local HTML page |
| albums | `photos` | approved cluster list | Photos albums (only after the user approves) |

# Videos

A video becomes one row of `embed/`, next to the photos, so `cluster`, `dedup`, `gallery` and
`albums` treat photos and videos alike. Its vector is the L2-normalized mean of the vectors of
its poster frame and up to 7 frames sampled evenly over its duration (fewer under 4 s). The frame
vectors stay in `embed/frames/` for later use (for example, per-frame search). Details:
[PhotoKit videos](/design/photokit.md#videos), [CLI videos](/design/cli.md#videos).

# Model

EmbeddingGemma 2 is a `gemma4_vision` encoder (16 layers, 768 wide, patch 16, 3x3 pooling, at most
280 soft tokens per image). Its output goes into a 24-layer Gemma text tower (512 wide), then mean
pooling and L2 normalization give a 768-d vector.[^eg2-config] The port is written from the
transformers source; candle's `gemma4` module cannot load this checkpoint as is. See
[EmbeddingGemma 2 family](/design/embeddinggemma2.md) of the [encoder core](/design/encoder-core.md).

# e-means and the additions

The faiss version keeps two exponential moving averages per cluster (count and sum). It takes its
batch size and step size from scaling laws, decays the step on a cosine curve, revives dead clusters
by splitting busy ones, and closes with an exact mean. Spherical mode keeps the centroids on the unit
sphere.[^faiss-emeans] Its fast kernels are Triton and CuTe, so they run on NVIDIA only.

| Addition | Why |
|---|---|
| Fused Metal assignment kernel | faiss has no fast Apple GPU path; assignment is the main cost |
| Automatic k (two levels) | The user does not know how many groups their library has |
| Incremental assignment | New photos join existing clusters without retraining |
| Near-duplicate groups | Bursts and repeated shots would otherwise form clusters of their own |

The last three are described, with their defaults and synthetic validation, in
[automatic k, incremental updates and near-duplicate groups](/design/autok-incremental-dedup.md).

[^faiss-emeans]: faiss `contrib/e_means/README.md`, sections "How the centroid updates work" and "Spherical e-means".
[^eg2-config]: `config.json` of `google/embeddinggemma-2`: `vision_config`, `text_config`, `vision_soft_tokens_per_image`.
