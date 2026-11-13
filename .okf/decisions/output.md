---
type: Decision
title: Output — local gallery first, Photos albums only after review
description: Clusters are shown in a local HTML gallery; albums are created in Photos only for the clusters the user approves.
tags: [decision, output, photos-albums]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-12T00:00:00Z'
verified:
  - by: human:samhvw8
    at: '2026-11-12T00:00:00Z'
---

# Decision

The user chose "both, albums after review" on 2026-11-12.

- `gallery` writes a local HTML page: one section per cluster, thumbnails from `$EMEANS_DATA/photos`.
  It does not change the library.
- `albums` creates one Photos album per approved cluster, under one folder. Albums sync to iCloud
  and show on all the user's devices, so this step needs an explicit list of approved clusters.
- The tool never deletes or edits assets.
