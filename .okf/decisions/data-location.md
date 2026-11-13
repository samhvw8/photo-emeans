---
type: Decision
title: Data location — the user's own machines only
description: Personal data (photos, video frames, embeddings, labels) may live only on the user's own machines, the Mac that runs the tool and optionally the user's own Linux GPU box, and nowhere else.
tags: [privacy, data, ubuntu, decision]
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T11:35:00Z'
verified:
  - by: human:samhvw8
    at: '2026-11-12T11:35:00Z'
---

# Decision

Personal data from the Photos library (resized photos, video frames, embeddings, cluster labels,
`approved.json`) may be stored and processed on:

| Machine | Where |
|---|---|
| This Mac | `$EMEANS_DATA` (`data/` in the repo, gitignored) |
| The user's own Linux GPU box (optional, for CUDA benchmarks and embedding) | `data/` in its checkout of the repo (gitignored), same layout |

It never goes to any other host or service: no cloud storage, no hosted model APIs, no public
artifacts. Copy only what a job needs (for a CUDA benchmark, `embed/embeddings.f32` alone; no ids or
file names), and delete scratch copies when the job ends. The Photos library itself is still only
read through PhotoKit on the Mac, and only `albums --yes` after the user's approval writes to it.

# History

Until 2026-11-12 the rule was "personal data stays on this Mac". The user widened it to their own Linux
box too, so real embeddings can drive the CUDA benchmarks and video frames can be embedded on the
box's GPU.
