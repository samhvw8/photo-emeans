---
type: Playbook
title: Adding an encoder family, and taking it to full speed
description: The ordered steps to put an embedding model on the encoder core — stage 0 exactness against transformers, then the speed steps with their measured gains — and what a family change must justify.
resource: crates/encoder/src/families
tags: [embedding, encoder, recipe, parity, candle, metal, cuda, benchmark]
status: deprecated
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T16:00:00Z'
sources:
  - id: tensorfold
    resource: https://github.com/ashhart/TensorFold/blob/main/docs/recipes/adding-a-zig-family.md
    title: TensorFold, "Adding a Zig family, and taking it to full speed" (the format this recipe follows)
  - id: eg2
    resource: /design/embeddinggemma2.md
    title: EmbeddingGemma 2 family — parity, dtype and throughput tables (M1 Pro)
  - id: fa-research
    resource: https://crates.io/crates/candle-flash-attn/0.11.1
    title: candle-flash-attn 0.11.1 source (build.rs, src/lib.rs, kernels/flash_fwd_launch_template.h), candle-nn 0.11.1 ops.rs, huggingface/candle issues 2275, 3541, 3515, 3605 (read 2026-11-12)
  - id: step5
    resource: bench/compare/results/encoder-step5/
    title: Step 5 measurements on the RTX 3060 (interleaved idle-box session, and the earlier loaded run)
  - id: compare
    resource: /results/compare-mac-ubuntu.md
    title: Mac vs Ubuntu embedding throughput, candle vs sentence-transformers (RTX 3060)
---

> **Moved (2026-11-12).** The code left this repo in the library split. The maintained version of this
> concept is `github.com/samhvw8/candle-embeddings/.okf/recipes/adding-an-encoder-family.md` (paths and APIs updated there). This copy is kept as history and is not updated.

# How to use it

This is how a model goes onto the [encoder core](/design/encoder-core.md) and gets every speed-up we
know. The steps are in order. Each one is a known win (or a known loss) with its measured result, so
a new family goes through them without measuring each one again. It comes from one bring-up:
EmbeddingGemma 2 on an M1 Pro (Metal) and an RTX 3060 (CUDA), 12 Nov 2026.[^eg2] On CUDA in bf16 it
went from 3.86 to 10.31 images/s at batch 8, ahead of sentence-transformers (9.25).

1. Do stage 0 first. No speed step goes in before the family is exact.
2. Do the steps in order. Build each one in the core, then call it from the family.
3. After each step, run the family's exactness gate (`full_parity`). An f32 step must keep min cosine
   1.000000; a lossy step reports its cosine against the f32 reference.
4. Before a speed claim, measure with `bench/compare/embed_run.py` (end to end, one warm-up batch,
   fresh process per case) against the previous state and sentence-transformers on the same machine.
5. When a new step proves out, add it here with its result. When one loses, add it to "Don't bother".

# What a family writes, and what it must justify

A family composes the [core blocks](/design/encoder-core.md#core-blocks) and writes only its config,
weight map, layer graph, input sequence, the parts of its image processor that are its own, and
kernels for layers only it has. For each core block it does not use, the family's module doc and the
change that adds it name the block and say why its own code is needed, or refuse that feature by
name until the block covers it.[^tensorfold] EmbeddingGemma 2 skips none.

# Stage 0: exact on day one

1. **Write the reference script.** Copy `bench/gemma/reference.py` (uv, PEP 723, pinned torch,
   torchvision, transformers, sentence-transformers). Change the model id and pinned revision, the
   loading kwargs and the prompts. Run the official sentence-transformers (or transformers) path on
   the **CPU in f32**. It writes the fixtures the core harness reads: `ref/images.json` (name, file,
   grid, input_ids, caption), `ref/images.f32`, `ref/pixels/<name>.f32` (the processor's output),
   `ref/texts.json`, `ref/texts_<prompt>.f32`. Use public images (scikit-image sample data, CC0 or
   public domain) plus aspect-ratio edge cases (4:1, 13:1 strip, 1:13 tall, a small upscale), and at
   least one text longer than any sliding window.
2. **Port from the transformers source, not from another candle model.** candle's `gemma4` module
   could not load EmbeddingGemma 2: its RMSNorm multiplies by `1 + weight`, it expects `<name>.weight`
   where the checkpoint has `<name>.linear.weight`, and its text model is a causal decoder.
3. **Make preprocessing bit-exact.** The `image` crate's Catmull-Rom resize differed by up to 27/255 per
   pixel and cost up to 0.0044 cosine (0.9956 on an upscaled 64x48 thumbnail). The core's
   `resize_bicubic_aa` is PyTorch's uint8 antialiased bicubic kernel (what torchvision runs) and gives
   0/255. Rescale as the reference does (`x * (1/255)` as one f32 product).
4. **Keep the reference's dtype casts.** Pooling and scale steps that transformers runs in f32 run in f32
   here too (Gemma 4's vision pooler). When a mean is followed by a scale, fold both into one product
   (`pool::grid_sum` returns the sum for this reason).
5. **Implement `ExactCheck` and add the gate.** A `smoke` test (two inputs, runs in `cargo test` when
   the model is cached) and an ignored `full_parity` test that calls `exact::run` and `Report::check_f32` (the f32 gate: cosine >= 0.99999, max element difference <= 1e-4). The test device must fail when the production device is missing, never fall back, and a skipped test must say so on stderr outside the harness capture.
   The gate: token ids identical, pixels 0/255, min cosine 1.000000 in f32 on every image (model only
   and end to end) and text, image -> caption retrieval equal to the reference's own score.
   EmbeddingGemma 2 on Metal f32: min cosine 1.000000, max element difference 1.23e-6, retrieval 13/15
   (the reference also gets 13/15).
6. **Run the gate at batch 1 and at a batch that mixes input lengths.** Batch 1 never builds a padded
   row. EmbeddingGemma 2 at batch 8 put the 1238-token text next to short ones: a padding query more
   than 512 positions past its text's end had every key blocked in the sliding layers, its softmax
   gave NaN, and `0 * NaN` in the next matmul spread it to valid rows (`non-finite embedding`). The
   core's `padding_mask` now gives padding queries no window; valid rows are unchanged (image vectors
   byte-identical). Image inputs (at most 284 tokens) never reached the window, so photo embeddings
   were never affected.

# Stage 1: speed, in order

| Step | What | Measured (EmbeddingGemma 2) |
|---|---|---|
| 1. The stride fix | `attention::heads_first`: rebuild canonical strides after the head transpose. candle 0.11.1's Metal matmul miscomputes a size-1 dim with a non-canonical stride (a one-kv-head tensor). This is correctness first: without it the full-attention layers gave cosine 0.87 on Metal (CPU was right) | Metal f32: cosine 0.87 -> 1.000000 |
| 2. Contiguous norms, fused kernels | Norms on the `(b, s, heads, d)` layout before the transpose, candle's fused `rms_norm` and `rope` kernels (2-D RoPE moves the axis to the batch dim), and the fused `sdpa` for unmasked vision attention (never builds the 12 x 2304² scores, 255 MB f32 per image) | M1 Pro f32: 0.73 -> 1.53 images/s (2.1x), exact |
| 3. dtype | f32 is the default (exact). bf16 is the low-memory option on Apple GPUs and the fast option on CUDA | M1 Pro: bf16 1.59 vs f32 1.55 images/s (no faster: M1 has no bf16 math), footprint 1.39 vs 3.07 GB, cosine >= 0.9997. RTX 3060: bf16 3.87 vs f32 3.28 (+18%), cosine 0.9996[^compare] |
| 4. Batch size | Default 1. Batches help only when the per-image work is small; a ViT at ~2300 tokens per image already fills the GPU | M1 Pro: 1.52 / 1.55 / 1.53 images/s at batch 1 / 4 / 8, peak footprint 2.74 / 3.07 / 3.75 GB. RTX 3060 bf16: 3.80 -> 3.87 (+2%) at batch 8. sentence-transformers bf16 on CUDA gains 59% at batch 8 (5.84 -> 9.27): its fused attention scales with batch, our plain path did not (with step 5, candle gains 7%: 9.67 -> 10.31) |
| 5. Fused attention on CUDA | `attention_fused` picks FlashAttention-2 (`candle-flash-attn` 0.11.1, `flash-attn` cargo feature) for bf16/f16 on CUDA when head_dim is a multiple of 8 up to 256; f32 stays on the exact plain path. The vision tower (head_dim 64, no mask) uses it; the text encoder (masks, head_dim 512) still runs the plain path. Build once with `CUDA_COMPUTE_CAP=86`, `RAYON_NUM_THREADS=16` and an existing `CANDLE_FLASH_ATTN_BUILD_DIR` (the build script panics when the directory is missing); it took 18 min on the box (53 nvcc files) and clones CUTLASS, so it needs network.[^fa-research] Exactness: CUDA f32 output byte-identical to before, gate min cosine 1.000000; bf16 image cosine min 0.999816 vs 0.999614 for the plain bf16 path (FlashAttention keeps its scores in f32; the plain path rounds them to bf16), retrieval 13/15 | RTX 3060 bf16, idle box, median of 3 interleaved rounds: 3.79 -> 9.67 images/s at batch 1 (2.55x), 3.86 -> 10.31 at batch 8 (2.67x); peak GPU memory 3.35 -> 0.98 GiB at batch 8. sentence-transformers bf16 on the same session: 5.41 / 9.25, so candle now leads 1.8x / 1.11x[^step5] |

# Next

- **Text encoder on FlashAttention:** pack the batch without padding and call
  `flash_attn_varlen_windowed` (window 512 each side for sliding layers); check the window bounds
  against the transformers mask (`|q - k| <= 512`) and the head_dim 512 kernels (small tiles on
  sm_86). Images spend most of their time in the vision tower, so this matters for text batches.
- **Profile before the next step.** The 2.6x came from removing the score matrix; where the rest of the
  time goes on CUDA has not been measured.

# Don't bother


- **f16 as the default.** 17% faster than f32 on the M1 Pro (1.81 images/s) and it did not overflow on
  the test inputs, but the model card warns that activations exceed the f16 range. Keep it opt-in.
- **bf16 for speed on M1-M4.** No native bf16 math; it saves memory only.
- **Batches above 1 on Metal for a ViT family.** No gain, more memory (+1 GB from batch 1 to 8).
- **Probing a candle model module first.** Check the norms, weight names and attention mask against
  the transformers source before reusing a candle module; port from transformers when they differ.

[^tensorfold]: TensorFold `docs/recipes/adding-a-zig-family.md`, "What a family PR must justify".
[^eg2]: [EmbeddingGemma 2 family](/design/embeddinggemma2.md), sections Parity, Dtype, Throughput, and the Metal pitfalls under Model.
[^fa-research]: candle-flash-attn 0.11.1 crate source and candle issues on build memory and Hopper-only v3.
[^step5]: `bench/compare/results/encoder-step5/` (`ubuntu-idle-r{1,2,3}-{flash,st,noflash}`, load average 1.6-4.7 before each run). A first same-day run while another job loaded the box (load average up to 59) gave candle-flash 7.22 / 8.27, sentence-transformers 4.06 / 7.01, candle without flash 3.28 / 3.42: the load slowed sentence-transformers more than candle, so those numbers are not used (`ubuntu-flash`, `ubuntu-noflash`).
[^compare]: [Mac vs Ubuntu comparison](/results/compare-mac-ubuntu.md), embedding throughput table.
