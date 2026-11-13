---
type: Design
title: EmbeddingGemma 2 family — image and text embeddings in candle
description: The first encoder family (crates/encoder/src/families/embeddinggemma2) — how it turns a photo into a 768-d vector with google/embeddinggemma-2 on Metal (input sequence, preprocessing, the vision and text towers, dtype), and its measured parity with sentence-transformers and throughput on the M1 Pro.
resource: crates/encoder/src/families/embeddinggemma2
tags: [embedding, embeddinggemma, encoder-family, candle, metal, parity, benchmark]
status: deprecated
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T16:00:00Z'
sources:
  - id: hf-repo
    resource: https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b
    title: google/embeddinggemma-2 at the pinned revision (config.json, processor_config.json, modules.json, 1_Pooling, config_sentence_transformers.json, tokenizer.json, chat_template.jinja)
  - id: model-card
    resource: https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md
    title: EmbeddingGemma 2 model card (task prefixes, "pass images without any prefix", "do not use float16")
  - id: tf-eg2
    resource: https://github.com/huggingface/transformers/tree/v5.22.2/src/transformers/models/embedding_gemma2
    title: transformers 5.22.2 embedding_gemma2 (modeling, configuration, processing)
  - id: tf-gemma4
    resource: https://github.com/huggingface/transformers/tree/v5.22.2/src/transformers/models/gemma4
    title: transformers 5.22.2 gemma4 (vision tower, image processors)
  - id: torch-upsample
    resource: https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp
    title: PyTorch 2.11 CPU upsample kernel (uint8 antialiased bicubic)
  - id: candle-gemma4
    resource: https://docs.rs/crate/candle-transformers/0.11.1/source/src/models/gemma4/
    title: candle-transformers 0.11.1 models/gemma4
  - id: runs
    resource: bench/gemma/
    title: reference.py output and the parity / throughput example runs of 2026-11-12 (ref/ and images/ are gitignored)
---

> **Moved (2026-11-12).** The code left this repo in the library split. The maintained version of this
> concept is `github.com/samhvw8/candle-embeddings/.okf/design/embeddinggemma2.md` (paths and APIs updated there). This copy is kept as history and is not updated.

# Overview

The EmbeddingGemma 2 family of the [encoder core](/design/encoder-core.md) runs
`google/embeddinggemma-2` (pinned revision `914f7f89`) on candle 0.11.1 with Metal (CUDA and CPU
also build). It was the crate `gemma-embed` until 2026-11-12; the move kept its output bit for bit.
It owns its config, weight map, input sequence, the two towers and the aspect-ratio target size;
everything else is a core block, and it skips none. It loads only the text tower, the vision
tower and `embed_vision` from the 1.4 GB safetensors file; the audio tower's 752 tensors are never read. The output matches the
sentence-transformers pipeline of the model repo (`Transformer -> Pooling(mean, include_prompt) ->
Normalize`).[^hf-repo]

```rust
use encoder::Encoder;
use encoder::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID, prompts};
let emb = EmbeddingGemma2::load(MODEL_ID, &Device::new_metal(0)?)?;  // f32, batch 1
let v: Vec<Vec<f32>> = emb.embed_image_paths(&paths, None)?;          // 768-d unit vectors
let t = emb.embed_texts(&["a cat"], Some(prompts::SEARCH_QUERY))?;
```

Other entry points: `load_with(model, device, dtype)`, `with_batch_size(n)`, and from the `Encoder`
trait `embed_images(&[DynamicImage], prompt)`, `embed_preprocessed(&[Patches], prompt)` (already
preprocessed). Family functions: `preprocess(&DynamicImage)`, `soft_tokens(&Patches)`,
`resolve_model_dir` / `cached_model_dir` (HF cache via hf-hub 1.0, cache first, network only on a
miss, pinned revision). `encoder::open_image(path)` applies EXIF orientation.

# Input sequence

An image without a prompt becomes, token by token:[^tf-eg2]

| position | id | token |
|---|---|---|
| 0 | 2 | `<bos>` |
| 1 | 255999 | `<\|image>` (boi) |
| 2 .. N+1 | 258880 | `<\|image\|>` x N, replaced by the N soft tokens of the vision tower |
| N+2 | 258882 | `<image\|>` (eoi) |
| N+3 | 1 | `<eos>` |

N = (grid_h / 3) x (grid_w / 3) <= 280, so N depends on the aspect ratio (a 512x512 image gives 256, a
451x300 one 260). `EmbeddingGemma2Processor.replace_image_token` expands the placeholder; the
tokenizer's template adds bos and eos. The mean pool runs over all N + 4 positions. This crate
builds the same string and tokenizes it, and the parity run checks the ids against the reference for
all 15 test images.

**Prompt for photos: none.** The model card says "Prefixes apply to text only. Pass images, video,
and audio without any prefix", and the clustering prefix is defined as a text instruction.[^model-card]
`embed_images(.., Some(p))` puts `p` before boi for experiments. Text gets `prompt + text`; the
prompts are in `gemma_embed::prompts` (`task: clustering | query: `, `task: search result | query: `, ...).

# Preprocessing

A port of `Gemma4ImageProcessor` (the torchvision backend, which transformers picks when torchvision
is installed):[^tf-gemma4]

1. RGB, alpha dropped. EXIF orientation applied by `open_image`.
2. Target size from `get_aspect_ratio_preserving_size`: scale to `280 x 9` patches of 16 px, floor each
   side to a multiple of 48 px; a side that floors to 0 becomes 48 and the other `floor(ratio) x 48`,
   capped at 13440 px.
3. Bicubic resize with antialias **on uint8**. torchvision runs PyTorch's integer kernel for this case:
   Keys cubic a = -0.5, the kernel widened by the scale when shrinking, per-pixel normalized weights
   quantized to int16 with a computed precision, width pass then height pass, uint8 between passes.[^torch-upsample]
   The core's `preprocess::resize_bicubic_aa` reproduces it bit for bit. The `image` crate's Catmull-Rom resize differed by up
   to 27/255 per pixel and cost up to 0.0044 cosine (0.9956 on an upscaled 64x48 thumbnail).
4. `x / 255`, no mean/std; patches row-major, each flattened as (py, px, channel). The vision tower
   maps to [-1, 1] itself.

# Model

Written from the transformers source, not from candle's `gemma4` module: candle's vision tower
multiplies RMSNorm by `1 + weight` (the stored q_norm weights are ~1.0, so this checkpoint wants
`weight`), expects `<name>.weight` where this checkpoint has `<name>.linear.weight`
(`Gemma4ClippableLinear`), and its text model is the causal Gemma 4 decoder.[^candle-gemma4] The vision
code keeps candle's structure (credited in `vision.rs`). The layer graphs are the family's own; the
norms, RoPE, attention, MLP, pooling and weight loading are [core blocks](/design/encoder-core.md).

| part | what it does |
|---|---|
| vision | patches -> `2(x-0.5)` -> linear 768 + x/y position rows -> 16 layers (q/k RMSNorm, unscaled v norm, 2-D RoPE theta 100 on halves of each 64-d head, scale 1.0, gated GELU MLP, 4 norms) -> 3x3 average pool in f32 -> x sqrt(768) in f32 -> unscaled RMSNorm -> linear 768->512 |
| text | token rows x sqrt(512), soft tokens merged; per-layer inputs `RMSNorm(W x / sqrt(512))` (24 x 512); 24 bidirectional layers: sliding ones see \|q-k\| <= 512 with head_dim 256, 2 kv heads, theta 1e4; layers 5/11/17/23 are full with head_dim 512, 1 kv head, theta 1e6; each layer ends with the gated PLE bottleneck and `x layer_scalar` |
| head | final RMSNorm -> `embedding_projection` 512 -> 768 on every token -> mean over the sequence -> L2 |

`embed_tokens` (262144 x 512, 268 MB) is never loaded: rows are read from the memory-mapped file per
batch. Other weights are copied into Metal buffers at load. Images in a batch that share a patch grid
run through the vision tower together.

Two Metal pitfalls cost time and are handled in code:

- **candle 0.11.1 Metal matmul with a size-1 dim of non-canonical stride gives wrong numbers** (CPU is
  right). `transpose(1, 2).contiguous()` of a one-kv-head tensor `(b, s, 1, d)` counts as contiguous,
  keeps stride 512 on the head dim, and the full-attention layers then produced cosines of 0.87.
  The core's `attention::heads_first` reshapes after the copy to rebuild canonical strides.
- Norms and RoPE on transposed tensors ran as strided kernels and took 38 of ~90 ms per vision layer.
  Norms now run on the `(b, s, heads, d)` layout with candle's fused `rms_norm` (f32 accumulation), and
  RoPE uses the fused `rope` kernel (the 2-D vision RoPE moves the axis to the batch dim). Vision
  attention uses the fused `sdpa` kernel, which never builds the 12 x 2304^2 score matrix (255 MB f32
  per image). Together: 0.73 -> 1.53 images/s in f32.

# Dtype

The model card says to run bf16 or f32, never f16, because activations exceed the f16 range.[^model-card]
Measured on the 15 test images and 26 text inputs (cosine vs the f32 CPU reference):

| dtype on Metal | image cos min / mean | text cos min | images/s (batch 4) | peak footprint |
|---|---|---|---|---|
| f32 | 1.000000 / 1.000000 | 1.000000 | 1.55 | 3.07 GB |
| f16 | 0.999950 / 0.999992 | 0.999998 | 1.81 | 2.02 GB |
| bf16 | 0.999784 / 0.999915 | 0.999854 | 1.59 | 1.39 GB |

M1 has no native bf16 math, so bf16 is the slowest half type and the least accurate. On CUDA (RTX 3060) bf16 is the fast option: with the
`flash-attn` feature the vision tower runs FlashAttention-2 and bf16 reaches 9.67 / 10.31 images/s at
batch 1 / 8 (f32: 3.23 / 3.27), image cosine min 0.999816 vs the f32 reference; see the
[recipe](/recipes/adding-an-encoder-family.md), step 5. f16 did not
overflow on these inputs, but the card's warning is about inputs we have not seen; a non-finite
output fails the batch (`run` checks every value). **Default: f32** — exact parity; f16 is 17%
faster but carries the card's overflow warning, and bf16 is no faster. bf16 is the low-memory
option (half the footprint, cosine >= 0.9997): `load_with(.., DType::BF16)`.

# Parity

`bench/gemma/reference.py` (uv, PEP 723: torch 2.11, torchvision 0.26, transformers 5.22.2,
sentence-transformers 6.3.0) embeds 15 images and 13 texts on CPU in f32 with
`SentenceTransformer("google/embeddinggemma-2", model_kwargs={"dtype": float32}, config_kwargs={"audio_config": None})`.
Images are scikit-image sample data (CC0 or public domain) plus four aspect-ratio edge cases (4:1
crop, 380x30 strip, 90x1200 tall, 64x48 upscale). It also writes the processor's patches, so the
model can be checked apart from preprocessing.[^runs]

| check (Metal, f32) | result |
|---|---|
| token ids vs processor, 15 images | identical |
| pixels after preprocessing vs torchvision | max diff 0 / 255 on all 15 |
| image cosine, reference pixels in | min 1.000000, mean 1.000000 |
| image cosine, Rust preprocessing | min 1.000000, mean 1.000000 |
| text cosine, 13 texts x 2 prompts (one text is 1238 tokens, so the 512 sliding window is exercised) | min 1.000000 |
| image -> caption top-1 (search-query prompt) | 13/15, same as the reference; both miss the logo and the 90x1200 rocket |
| CPU f32 | same as Metal |

Commands: `uv run --script bench/gemma/reference.py`, then
`cargo run --release -p encoder --example parity -- metal f32 4`, or the ignored test
`cargo test --release -p encoder --test embeddinggemma2 -- --ignored` (the core
[exactness harness](/recipes/adding-an-encoder-family.md#stage-0-exact-on-day-one)).

The gate runs at batch 1 and batch 8 (a batch that mixes the 1238-token text with short ones).
Batched texts of mixed length longer than the 512 window gave a non-finite vector before the core's
`padding_mask` fix of 2026-11-12 (see the [recipe](/recipes/adding-an-encoder-family.md#stage-0-exact-on-day-one),
step 6); images were never affected.

After the move into `crates/encoder` (2026-11-12): the same gate passes (min cosine 1.000000, max
element difference 1.23e-6 vs the reference, pixels 0/255, ids identical, retrieval 13/15), and the
15 test vectors are byte-identical to the old crate's stored Metal f32 output at batch 1 and 8
(`cmp` against `bench/compare/results/mac/embed/candle_metal_f32_b{1,8}/embeddings.f32`).

# Throughput

`crates/encoder/examples/throughput.rs`: the 15 test images cycled to 32, preprocessing then model, after one
warm-up batch; `/usr/bin/time -l` for memory. The GPU was shared (other processes held it at
70-99% utilization between runs), so the rates are a floor.

| dtype | batch | model images/s | max RSS | peak footprint |
|---|---|---|---|---|
| f32 | 1 | 1.52 | 1.34 GB | 2.74 GB |
| f32 | 4 | 1.55 | 1.45 GB | 3.07 GB |
| f32 | 8 | 1.53 | 1.29 GB | 3.75 GB |
| f16 | 1 | 1.79 | 1.64 GB | 1.79 GB |
| f16 | 4 | 1.81 | 1.62 GB | 2.02 GB |
| f16 | 8 | 1.77 | 1.63 GB | 2.40 GB |
| bf16 | 1 | 1.57 | 1.64 GB | 1.23 GB |
| bf16 | 4 | 1.59 | 1.57 GB | 1.39 GB |
| bf16 | 8 | 1.57 | 1.56 GB | 1.73 GB |

Preprocessing runs at ~68 images/s on one core, so the model is the bottleneck. Batch size does not
change the rate (the vision tower already runs 2000+ tokens per image), but it raises peak memory.
**Default batch: 1** (`DEFAULT_BATCH_SIZE`). Max RSS counts the memory-mapped checkpoint pages the
loader touched, so it is higher for the half types, which read and convert every weight; the peak
footprint is the number to watch. About 1.5 images/s means ~3 hours per 15k photos.

Where the time goes (f32, profiled per stage with device syncs): the vision tower is ~90% of it.
Its matmuls run at ~3.6 TFLOP/s on this GPU when it is idle, so ~1 TFLOP per image puts the floor near
0.3 s per image; the rest is attention and elementwise work, plus the GPU share other processes take.

[^hf-repo]: `modules.json`, `1_Pooling/config.json`, `config.json` of the pinned revision.
[^model-card]: Model card, "Task Instruction Prefixes" and "Numerical Precision".
[^tf-eg2]: `processing_embedding_gemma2.py` (`replace_image_token`) and `modeling_embedding_gemma2.py`.
[^tf-gemma4]: `image_processing_gemma4.py` (`Gemma4ImageProcessor._preprocess`, `get_aspect_ratio_preserving_size`).
[^torch-upsample]: `_compute_indices_min_size_weights_aa`, `_compute_index_ranges_int16_weights`, `basic_loop_aa_horizontal<uint8_t>`.
[^candle-gemma4]: `vision.rs` (`RmsNorm::forward` adds 1.0 to the weight), `text.rs` (causal decoder with KV cache).
[^runs]: `bench/gemma/reference.py`; outputs in `bench/gemma/ref/`.
