---
type: Decision
title: Runtime choice — candle on Metal, plus a Bend twin of e-means
description: The model and the main e-means run in Rust/candle on this Mac's GPU; a Bend 2 e-means runs on the same vectors for comparison.
tags: [decision, candle, metal, bend]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T00:00:00Z'
sources:
  - id: bend-readme
    resource: https://github.com/HigherOrderCO/Bend/blob/main/README.md
    title: Bend 2 README (v2.0.40), limitations section
  - id: busy-mac
    resource: local note (not published)
    title: Local ML on a busy Apple Silicon Mac
---

# Decision

- The user asked for candle. Candle 0.11.1 has a Metal backend and already ships `gemma4`, so the
  embedding model and the main e-means are Rust/candle, on the M1 Pro GPU.
- The user also asked for a Bend version, kept next to the Rust one and compared. Bend 2 has F32,
  in-place arrays and a Metal backend, but only U32/F32 numbers and no F64.[^bend-readme] Bend
  runs the e-means core only. The 740M-parameter model stays in candle.
- Jobs run on this Mac. The home GPU server was offline on 2026-11-12, and the photos live here.
  Weights are memory-mapped and batches stay small, because the Mac is a busy 16 GB machine.[^busy-mac]

# Rejected

| Option | Why not |
|---|---|
| llama.cpp GGUF of EmbeddingGemma 2 | The user named candle; the vision path in GGUF was not checked |
| MLX | A second runtime for ~10-20% speed, measured on the text model only |
| Bend for the model | No tensor library; F32 only; the model would be a rewrite of every kernel |

# Runs

Nothing that keeps running: a CLI started by hand. Bend is pinned in `mise.toml` with telemetry off.

[^bend-readme]: Bend README, "Limitations": "Numbers are Nat, U32 and F32 only"; targets C, Metal, CUDA, JavaScript.
[^busy-mac]: Memory, not CPU, causes lag; mmap the weights.
