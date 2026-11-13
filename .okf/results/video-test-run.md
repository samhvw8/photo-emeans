---
type: Result
title: 'Video test run — download size, speed, and how videos cluster with photos'
description: A test set of about 500 fetched and embedded videos, with measured rendition size and fetch time per video, Mac embed speed, how the videos land in an automatic-k run next to about 16k photos, and the extrapolation to a whole library for a decision on a full run.
tags: [video, photokit, results, extrapolation, cluster]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T05:00:00Z'
sources:
  - id: fetch-log
    resource: data/fetch-videos.log
    title: fetch-videos runs of 2026-11-12 (per-video duration, rendition MB, request and decode ms; personal data, gitignored)
  - id: embed-log
    resource: data/embed-videos.log
    title: embed --media videos runs of 2026-11-12 on Metal (shared GPU lock, then quiet-run chunks)
  - id: av-run
    resource: data/clusters/auto-dedup-av-test/report.json
    title: cluster --auto --dedup --dedup-max-gap 600 --backend cpu on about 16k photos + a test set of about 500 videos
---

# Scope

The user asked for a small test set, not the whole library. Fetching stopped at a test set of about
500 videos. All of them are embedded. No fetch runs any more; the full-library commands below are
documented, not run.

# Fetch (medium rendition, concurrency 4, M1 Pro)

| Measure | Value |
|---|---|
| Videos fetched | About 500, about 7 frames per video (0.92 MB of JPEG frames per video), 1 failure (AVFoundation -11821 "Cannot Decode") |
| Rendition downloaded into the Photos cache | 0.35 MB per second of video[^fetch-log] |
| Per video (median) | request + download 2.4 s, frame decode 0.08 s, total 4.0 s per video in flight |
| Wall time per video | 2.07 s (276 videos of all lengths in 570 s); 1.26 s for videos up to 60 s |
| Free disk effect | Free disk fell by about the download size; the Photos derivatives folder grew by the same amount (purgeable cache) |

MB per video by length (first ~300 videos): under 4 s 0.7, 4-15 s 3.4, 15-60 s 9.3, 1-5 min 46.5,
over 5 min 597. Videos over a minute are about 85% of the bytes. See
[PhotoKit videos](/design/photokit.md#videos).

# Embed (Mac, Metal, f32, batch 1)

| Phase | Frames | Rate |
|---|---|---|
| First runs (GPU lock shared with benchmark jobs, 500 frames per hold) | about 1.2k | 1.18-1.33 frames/s |
| `scripts/quiet-run.py` chunks of 300 frames (CPU gate at 40% busy, nice 15, GPU lock) | about 2.3k in 8 chunks | 1.40 frames/s while running (~215 s per chunk); 71 min wall with the gate's waits |

Result: `embed/` holds every photo and every test video, `embed/frames/` every frame, 0 frame
failures.[^embed-log]

# How the videos cluster

`cluster --auto --dedup --dedup-max-gap 600 --backend cpu` (the settings of `auto-dedup`), run
`auto-dedup-av-test`: about 16k rows → about 11k dedup groups → about 280 leaves (the photo-only run
has about 300), 55.7 s on the CPU.[^av-run] An earlier version of the run, with about a third of the
videos embedded, gave the same picture: every video in a leaf with photos, about 15% stacked with a
photo.

| Question (test set) | Answer |
|---|---|
| Videos in leaves that also hold photos | all of them (about 40% of the leaves); no video-only leaf |
| Video share of those leaves | median 5%, max 81% |
| Videos flagged outlier | under 1% (photos: about 2%) |
| Videos in a near-duplicate stack with a photo | about a fifth |
| Cosine to the nearest photo | median 0.860, p10 0.785, p90 0.976 |
| Nearest photo in the same leaf | about two thirds |
| Nearest photo taken within 10 min / 1 h / 1 day | 31% / 38% / 42% |
| Videos with any photo within 10 min | about half; best cosine to such a photo: median 0.926, about a fifth of all videos at or above the 0.95 dedup threshold |

So a video's mean-of-frames vector sits among photos of the same scene: no video-only cluster
forms, and almost a fifth of the videos stack with a photo shot within 10 minutes. The gallery of
the run shows the photo and video counts (headless render, no console errors).

# Extrapolation to a whole library (about 1.7k videos)

| Step | Estimate |
|---|---|
| Download into the Photos cache, medium | Tens of GB (0.35 MB per second of video); the per-length table gives about a third more. With `--fast-over 60`: under half |
| Fetch time, concurrency 4 | ~1 h (about 13 MB/s; 2.07 s per video wall) |
| Frames on disk | up to 8 per video, ~0.9 MB per video (~1.6 GB) |
| Embed on the Mac | ~2.5 h of GPU time at 1.40 frames/s; longer in wall time, because quiet-run waits while the user works |
| Embed on the box (CUDA f32, 3.2 images/s) | ~67 min; needs `photo-emeans` to build on Linux and copying frames to the box |

On a Mac with little free disk, a full medium run does not fit without macOS purging the Photos
cache; `fetch-videos` stops below 8 GB truly free.

Full-library commands (not run; the user decides):

```
mise run pe -- fetch-videos --all [--fast-over 60]
# in chunks, each through the CPU gate (quiet-run holds the GPU lock; --nice 0 keeps its nice 15):
scripts/quiet-run.py --lock /tmp/emeans-mac-gpu.lock -- target/run/release/photo-emeans embed --media videos --limit 300 --nice 0
mise run pe -- cluster --auto --dedup --dedup-max-gap 600 --run auto-dedup-av
mise run pe -- gallery --run auto-dedup-av
```

[^fetch-log]: Sum over the logged test-set videos: rendition bytes divided by seconds of video.
[^embed-log]: Progress and summary lines of `embed-videos.log` and `embed/runs.jsonl`, 2026-11-12 19:02-20:50 local.
[^av-run]: Counts from `labels_l2.i32`, `dedup.i32`, `outlier.u8` and the embeddings, computed by a local script that prints counts only.
