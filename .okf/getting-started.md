---
type: Reference
title: Getting started — emeans
description: What emeans is, how its steps fit together, and where each piece of knowledge lives.
tags: [getting-started, photos, clustering, e-means, embeddinggemma-2]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T12:40:00Z'
---

# Overview

emeans clusters the user's whole Photos library (local and iCloud) by image content. It has four
steps: fetch resized copies through PhotoKit, embed each photo with EmbeddingGemma 2 in candle,
cluster the vectors with e-means, then review the clusters in a local gallery before any Photos album
is created.

The e-means algorithm comes from faiss `contrib/e_means` (Meta FAIR, MIT). The project ports it to
Rust/candle, adds four improvements, and also implements it in Bend 2 to compare the two.

# Running heavy jobs

| Where | How |
|---|---|
| Mac (the user's workstation; a heavy job freezes the UI) | `scripts/quiet-run.py --lock /tmp/emeans-mac-gpu.lock -- <cmd>`: waits until CPU busy (user + sys from `top`) is at most 40% for 3 samples, then runs at nice 15 holding the GPU lock. Runs of minutes, not hours. The load average is not used: on macOS it also counts threads waiting on disk, and it read 86 while the CPU was 85% busy with the user's own apps |
| Linux GPU box (optional) | Named tmux session whose script ends with a `=== <time> DONE` line (wait on that line, not on a bare `DONE`: `run.py` prints its own), `flock /tmp/emeans-box-gpu.lock` around GPU work, `tmux ls` and `nvidia-smi` first |

Personal data stays on these two machines only ([data location](decisions/data-location.md)).

# Where to read next

| Topic | Concept |
|---|---|
| Steps and data files | [Pipeline](/design/pipeline.md) |
| Why candle, Metal on this Mac, and a Bend twin | [Runtime choice](/decisions/runtime-choice.md) |
| How photos are read from Mac and iCloud | [Photo source](/decisions/photo-source.md) |
| Gallery first, albums after review | [Output](/decisions/output.md) |

The data contract (file formats) is in `AGENTS.md` at the repo root, next to the commands and rules.

# Libraries

Since the [library split](/decisions/library-split.md) the e-means engine and the encoder are their
own repos, and this tool depends on them as git dependencies pinned by commit. Their design notes, recipes and synthetic results
live in their own OKF bundles:

| Library | Repo | Bundle |
|---|---|---|
| e-means (`emeans` facade over `emeans-core`, `-metal`, `-cuda`, `-cluster`) | [github.com/samhvw8/emeans](https://github.com/samhvw8/emeans) | its `.okf/` |
| Embedding encoders (`candle-embeddings`, imported as `encoder`; EmbeddingGemma 2) | [github.com/samhvw8/candle-embeddings](https://github.com/samhvw8/candle-embeddings) | its `.okf/` |

This bundle keeps what is specific to the photo tool: the pipeline, PhotoKit, the CLI, decisions,
and every result measured on the real library.
