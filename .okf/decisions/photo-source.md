---
type: Decision
title: Photo and video source — PhotoKit resized copies and video frames, iCloud download allowed
description: Photos are read through PhotoKit as copies of at most 1024 px, and videos as a poster plus sampled frames from a medium (or fast) rendition, which also covers assets stored only in iCloud; originals are never downloaded and no video file is kept.
tags: [decision, photokit, icloud, video]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T05:00:00Z'
---

# Context

The library holds about 16k photos and about 1.7k videos. With "Optimize Mac Storage", almost every
original is in iCloud only: the library folder holds small derivatives, not the originals.

# Decision

- Use PhotoKit (`PHImageManager`, network access allowed, target size 1024 px, aspect fit) through
  `objc2-photos`. It returns a local derivative when one is large enough and downloads a resized
  copy from iCloud when not. Expected transfer: about 3-4 GB.
- Do not read `Photos.sqlite` or the library folder directly: the schema is private and changes
  between macOS versions, and it cannot fetch iCloud-only photos.
- Videos were a later step; see the video decision below (2026-11-12).
- The terminal app that runs the CLI needs the Photos permission once (macOS asks).

# Measured full fetch (2026-11-12)

About 16k photos fetched, about a fifth of them downloaded from iCloud, 0 failed, in 49 min (0.19 s
per photo wall, 6 in flight). 3.4 GB in `$EMEANS_DATA/photos`, 223 KB per photo on average. See
[PhotoKit design](/design/photokit.md) for why the fetched count is lower than the library's total.

# Videos (2026-11-12)

The user chose "photo first", then asked for videos so that photos and videos cluster together.

- List videos with the same fetch options as photos: about 1.7k videos, all only in iCloud
  (no medium or fast rendition on the Mac); originals average about 160 MB per video.
- Per video, keep JPEG frames only: the poster (`requestImage`, as for photos) and up to 7 frames
  sampled evenly over the duration with `AVAssetImageGenerator` from the rendition that
  `requestAVAssetForVideo` hands out. Long side at most 1024 px. Never the original, never a video
  file of our own. See [PhotoKit videos](/design/photokit.md#videos).
- Rendition: medium (576 x 1024 for phone video, about the model's input size). Fast (360 x 640,
  ~0.4x the bytes) only where disk space requires it (`--fast-over`).
- One vector per video (the normalized mean of its frame vectors) goes into the same store as the
  photos; frame vectors are kept separately.
- Rejected: decoding the original (hundreds of GB of downloads for a large library); `requestExportSession` (writes a video
  file); `requestPlayerItem` plus frame grabs from playback (a stream, slower, and Photos may still
  cache it); one frame per video (the poster alone misses what happens in longer clips).

Cost: Photos stores each downloaded rendition in its library as purgeable cache. At medium, a large
library takes tens of GB (about 0.37 MB per second of video, measured on about 300 videos), mostly
videos over a minute. `fetch-videos` stops below 8 GB truly free disk.

# Runs

Nothing that keeps running. The copies take about 3.4 GB in `$EMEANS_DATA/photos`; video frames
take ~0.9 MB per video in `$EMEANS_DATA/videos` (~1.6 GB for about 1.7k videos), plus the Photos cache
above, which macOS can purge.
