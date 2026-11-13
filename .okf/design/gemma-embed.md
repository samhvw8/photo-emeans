---
type: Design
title: gemma-embed (deprecated) — moved into the encoder core
description: The crates/gemma-embed crate was folded into crates/encoder on 2026-11-12; its content now lives in the encoder-core and EmbeddingGemma 2 family concepts.
tags: [embedding, embeddinggemma, deprecated]
status: deprecated
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T16:00:00Z'
---

# Overview

`crates/gemma-embed` no longer exists. On 2026-11-12 it became the first family of `crates/encoder`,
with byte-identical output:

- [Encoder core](/design/encoder-core.md): the `Encoder` trait, the shared blocks, the exactness harness.
- [EmbeddingGemma 2 family](/design/embeddinggemma2.md): input sequence, preprocessing, towers, dtype,
  parity and throughput (everything this concept held).
- [Adding an encoder family](/recipes/adding-an-encoder-family.md): the ordered steps with their gains.

API mapping: `gemma_embed::GemmaEmbedder` -> `encoder::families::embeddinggemma2::EmbeddingGemma2`
(with the `encoder::Encoder` trait in scope); `embed_patches` -> `embed_preprocessed`; results are
`Vec<Vec<f32>>`; `gemma_embed::open_image` -> `encoder::open_image`; `Patches` -> `encoder::Patches`
(now carries its patch size); `MODEL_ID`, `MODEL_REVISION`, `EMBED_DIM`, `DEFAULT_BATCH_SIZE`,
`prompts`, `preprocess` -> the family module.
