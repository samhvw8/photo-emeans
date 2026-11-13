---
type: Design
title: PhotoKit access
description: How the photos crate lists the library, fetches 1024 px JPEG copies (local or from iCloud) of photos and sampled frames of videos, writes them resumably, and gates album creation and renaming; with the request options chosen, measured speed and size, and permission steps.
tags: [photokit, icloud, fetch, permissions, video, avfoundation]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T05:00:00Z'
sources:
  - id: apple-avasset
    resource: https://developer.apple.com/documentation/photos/phimagemanager/requestavasset(forvideo:options:resulthandler:)
    title: Apple, PHImageManager.requestAVAsset(forVideo:options:resultHandler:)
  - id: objc2-avf
    resource: https://docs.rs/objc2-av-foundation/0.3.2
    title: objc2-av-foundation 0.3.2 (AVAssetImageGenerator.rs, AVAsset.rs, AVTime.rs)
  - id: video-probe
    resource: crates/photos/examples/video_probe.rs
    title: Video count and pilot runs against the user's library, 2026-11-12
  - id: objc2-photos
    resource: https://docs.rs/objc2-photos/0.3.2
    title: objc2-photos 0.3.2 (PHImageManager.rs, PHFetchOptions.rs, PHError.rs, PhotosTypes.rs)
  - id: probe-run
    resource: crates/photos/examples/probe.rs
    title: Probe runs against the user's library, 2026-11-12
---

# Code

`crates/photos` ([photo source decision](/decisions/photo-source.md)):

| Function | Does |
|---|---|
| `authorize(timeout) -> Access` | Read/write status; prompts once when not determined |
| `ensure_authorized(timeout) -> Result<()>` | Errors with the exact System Settings steps unless full access |
| `list_photos() -> Result<Vec<AssetInfo>>` | Image assets, oldest first |
| `fetch_resized(&AssetInfo, &FetchConfig) -> Result<Fetched>` | One JPEG, long side <= 1024 |
| `is_local(&AssetInfo, max_side) -> Result<bool>` | Network-off check: can Photos render it without iCloud |
| `fetch_all(dir, &[AssetInfo], &FetchAllConfig, on_event) -> Result<FetchReport>` | Resumable pool writing `*.jpg` + `manifest.jsonl` |
| `create_albums(folder, &[AlbumRequest], confirm) -> Result<String>` | One folder of albums, one transaction; refuses unless `confirm` |
| `find_folder(title) -> Result<FolderListing>` | Read-only: the one top-level folder with that title and its children |
| `rename_collections(&Rename, &[Rename], confirm) -> Result<()>` | Titles of a folder and its albums, one transaction, checked before and read back after; refuses unless `confirm` |

| `list_videos() -> Result<Vec<AssetInfo>>` | Video assets, oldest first (`is_video`, `duration` set) |
| `fetch_video_frames(&AssetInfo, &VideoConfig) -> Result<VideoFrames>` | Poster + sampled frames of one video as JPEGs |
| `video_is_local(&AssetInfo, &VideoConfig) -> Result<bool>` | Network-off `requestAVAsset`: is the rendition on this Mac |
| `frames_from_file(path, times, max_side, quality)` | Same frame path on a local video file (tests, sample media) |
| `fetch_all_videos(dir, &[AssetInfo], &VideoFetchAllConfig, on_event)` | Resumable pool writing `<uuid>/fNN.jpg` + `manifest.jsonl` + `failures.jsonl` |

Probe: `cargo run -p photos --example probe -- --limit 20 --out <dir> [--check-local] [--pick spread|first|newest]`.
Video count (read-only): `cargo run -p photos --example video_probe -- [--check-local] [--fast]`.

# What is listed

`PHAsset fetchAssetsWithMediaType:Image` with source type `UserLibrary`, sorted by `creationDate`,
`includeHiddenAssets = false`, `includeAllBurstAssets = false`.[^objc2-photos]

The listed set is the user library including the iCloud Shared Library, burst representatives
only. It leaves out non-representative burst frames (the dedup step would group them anyway) and
iCloud Shared Album photos (`CloudShared`, other people's albums); hidden photos are excluded too.
On a real library the listed count was about 7% below the photo count Photos shows, almost all of
it burst frames.[^probe-run]

`mediaSubtypes` become names (`panorama`, `hdr`, `screenshot`, `live`, `depthEffect`, `spatial`).
Undocumented bits stay as `bit<N>`: `bit9` is set on almost half the photos and `bit6` on a handful.

# Request options

`requestImageForAsset:targetSize:contentMode:options:resultHandler:`, target 1024 x 1024 aspect-fit:

| Option | Value | Why |
|---|---|---|
| API | `requestImage`, not `requestImageDataAndOrientation` | The data call returns the full original bytes, which downloads originals from iCloud |
| `deliveryMode` | `HighQualityFormat` | One final callback; no degraded preview |
| `resizeMode` | `Exact` | Result fits the 1024 box. `Fast` may return a bigger derivative we would scale again |
| `version` | `Current` | Edits and crops applied, as the user sees the photo |
| `networkAccessAllowed` | `false` first, then `true` | See below |
| `synchronous` | `false` | Allows a timeout and `cancelImageRequest`; a sync request cannot be stopped when it stalls |

Each fetch asks with network off first. A local copy comes back at no extra cost. A photo that
is only in iCloud at this size fails with `PHPhotosErrorNetworkAccessRequired` (3164), and only
then the fetch asks again with network on. This makes `Fetched::from_icloud` exact. The progress
handler cannot tell: it also fires (with 1.0) for purely local renders. It only feeds the idle
timeout (60 s without progress, 300 s hard, 3 attempts, pause 2 s then 4 s).

The `NSImage` is already upright. It is drawn into an 8-bit sRGB bitmap (flattens HDR, wide
gamut and alpha onto white) and encoded by ImageIO at quality 0.88. The JPEG has no EXIF
orientation tag. HEIC, JPEG, PNG and RAW (`com.adobe.raw-image`) originals all came out as valid JPEGs.

# Main-thread rule

PhotoKit delivers asynchronous image results on the **main queue**, even for requests made on
other threads. Measured: a background-thread request with the main thread blocked timed out at
60 s; with the main thread running its run loop it finished in 0.33 s.[^probe-run]

- `fetch_resized` on the main thread runs the main run loop while it waits.
- `fetch_all` must run on the main thread (it errors otherwise). Workers wait on a condvar; the
  main thread receives their results and runs the run loop in 20 ms slices.
- A caller that blocks the main thread some other way gets a timeout on every photo.

# Files and resume

- `<uuid>.jpg` is written as `<uuid>.jpg.tmp`, fsynced, then renamed.
- Only after the rename, the single writer appends the manifest line in one write.
- Manifest `width`/`height` are the JPEG's pixel size; lines are in completion order.
- On start: torn last line dropped; lines whose file is missing or not a JPEG of the recorded
  size dropped; duplicate ids keep the first; the manifest is rewritten atomically when
  anything was dropped. A bad line in the middle stops the run. `*.tmp` files are deleted.
- Failed assets get no line, so the next run retries them.

Checked on real data: deleting one JPEG and tearing the last line gave `repaired 2`, two refetches,
21 unique lines, no `.tmp` left.

# Measured (M1 Pro, concurrency 6, 2026-11-12)

| Sample | Local | From iCloud | Wall | Avg size |
|---|---|---|---|---|
| 21 spread across the library | 16, 21-76 ms each | 5, 2.9-5.1 s each | 5.3 s (0.25 s/photo) | 238 KB |
| 12 newest, concurrency 1 | 7, 55-210 ms | 5, 0.9-3.0 s | 9.3 s | 125 KB |

About a quarter of sampled photos needed iCloud at 1024 px; the rest render from local
derivatives. A downloaded copy stays cached: the same photo is local on the next run. Rough
estimate for about 16k photos: ~230 KB each ≈ 3.6 GB on disk, under an hour at concurrency 6.
This is an extrapolation from 33 photos.

# Videos

## What a library holds

`PHAsset fetchAssetsWithMediaType:Video`, same options as photos (user library, no hidden).[^video-probe]
On a real library of about 1.7k videos, most clips were short (median about 15 s) with a long tail
of clips over 5 minutes; originals averaged about 160 MB per video (undocumented `PHAssetResource`
key `fileSize`); no video had a medium or fast rendition on the Mac (network-off request); and K = 8
(fewer under 4 s) gives about 7 frames per video. Subtypes seen: the undocumented bit 20 (almost
half), screen recordings, cinematic, high frame rate (slo-mo) and time-lapse.

## Request

Per video ([`fetch_video_frames`](/design/photokit.md#code)):

1. **Poster**: `fetch_resized` (the photo path, `requestImage`). It works on video assets and
   returns the frame Photos shows. Stored as `f00`, `t = null`.
2. **Rendition**: `requestAVAssetForVideo:options:resultHandler:` with
   `deliveryMode = MediumQualityFormat`, `version = Current` (edits and slo-mo applied; a slo-mo
   video may come back as an `AVComposition`, which the generator also reads), network off first,
   then on after `PHPhotosErrorNetworkAccessRequired` or `PHImageResultIsInCloudKey`. Idle timeout
   90 s (progress handler), hard timeout 900 s. A remote `AVURLAsset` (a stream) is refused. Photos
   downloads the rendition into its own library and hands out a `file:` URL; this crate writes no
   video file.[^apple-avasset]
3. **Frames**: `AVAssetImageGenerator` with `appliesPreferredTrackTransform = true` (rotated
   phone video comes out upright), `maximumSize = 1024 x 1024`, tolerance before and after a quarter
   of the frame spacing (0.02-1 s), default dynamic range policy `ForceSDR`. All sample times go to
   one `generateCGImagesAsynchronouslyForTimes` call (the batch, in-time-order mode); the
   synchronous `copyCGImageAtTime` is deprecated.[^objc2-avf]
4. **Times**: `n` = 8 frames (poster included) from 4 s on, else `round(2 x duration)` clamped to
   2..8. The `n - 1` sampled times are the middles of `n - 1` equal slices of `PHAsset.duration`.
5. **Encode**: the photo JPEG path (8-bit sRGB, quality 0.88).

Apple's page does not name the queue of the `requestAVAsset` result handler.[^apple-avasset] The
code waits the same way as for images: on the main thread it runs the run loop; on a worker it
waits on a condvar while `fetch_all_videos` (main thread) runs the run loop. The generator's
callbacks arrive off the main thread: `frames_from_file` finishes from a test thread with no run
loop.

Whole-video retries: 3 attempts, pause 2 s then 4 s. After the last, the video goes to
`videos/failures.jsonl`, and later runs skip it unless `--retry-failed` (a retry may download
again).

## Files and resume

- Frames go to `<uuid>.tmp/fNN.jpg` (each fsynced), the directory is fsynced and renamed to
  `<uuid>/`, then the single writer appends the manifest line. A `<uuid>/` without a line (kill
  between rename and append) is replaced on the next run; `*.tmp` directories are deleted.
- On start, the manifest gets the photo repair rules: torn last line, lines whose frames are
  missing or not JPEGs of the recorded size, and duplicate ids are dropped; a bad middle line stops.
- Free-disk guard: workers stop starting videos when the data volume has less than
  `min_free_bytes` (8 GB) free (`NSFileSystemFreeSize`); the run then fails with the reason.

Tested in `video_store.rs` (field order, atomic directory write, orphan replace, repair, failures
file). A kill of the first full run (SIGTERM at video 30) was resumed: 30 lines kept, the orphan
directory replaced.

## Renditions and the Photos cache

Pilot of 12 videos spread across the library, concurrency 4:[^video-probe]

| Rendition | Frame size (9:16) | Bytes per second of video | Wall per video |
|---|---|---|---|
| Medium | 576 x 1024 | 0.25 MB/s (10.8 MB per video) | 2.1 s |
| Fast | 360 x 640 | ~0.15 MB/s | 1.9 s |

Medium is the default: 576 x 1024 is about the model's input budget (280 soft tokens = 2,520
patches of 16 px), while fast frames would be upscaled. A request for the fast rendition of a
video whose medium rendition is cached returns the cached medium file.

Photos keeps every downloaded rendition in `Photos Library.photoslibrary/resources/derivatives/`
(`<uuid>_2_3_o.mp4`, `_2_4_o.mp4`, `_2_201_o.mov`) with the `com.apple.cscachefs` attribute, so
CacheDelete can purge it under disk pressure. The library grew by the sum of the renditions.

The pilot under-estimated the size. The first ~300 videos of the full run (oldest first) downloaded
0.37 MB per second of video:

| Duration | MB per video (medium) | MB/s |
|---|---|---|
| < 4 s | 0.7 | 0.42 |
| 4-15 s | 3.4 | 0.40 |
| 15-60 s | 9.3 | 0.30 |
| 1-5 min | 46.5 | 0.34 |
| > 5 min | 597 | 0.39 |

So a large library at medium takes tens of GB of cache, and videos over a minute are ~85% of it.
After those videos the volume's truly free space (`NSFileSystemFreeSize`) was about half of what it
reported as "available for important usage" (`NSURLVolumeAvailableCapacityForImportantUsageKey`):
the difference is purgeable cache. `fetch-videos --fast-over SECS` takes the fast rendition (~0.4x the bytes) for
longer videos, and `--max-duration SECS` leaves longer videos for a later run.

# Permissions

The process's responsible app (the terminal app) needs Photos "Full Access". In the
development runs it already had it (status `Authorized`, no prompt). When it does not:

1. Run the CLI. macOS shows "... would like to access your Photos". Click "Allow Full Access".
2. If no dialog appears, or access is Limited or Denied: System Settings > Privacy & Security >
   Photos > set the terminal app to "Full Access", quit and reopen it, run again.

`Limited` is refused: only hand-picked photos would be visible.

# Albums

`create_albums` checks the arguments (confirm, non-empty and unique titles, non-empty unique ids),
resolves every id before any change, then creates the folder and all albums in one
`performChangesAndWait`. Only the argument checks are unit tested; it created the `auto-dedup`
folder "emeans" (about 300 albums) on a real library. See the [output decision](/decisions/output.md).

`find_folder(title)` (read-only) returns the one top-level folder with that title and its direct
children (local identifier, title, asset count for an album, `None` for a sub-folder), and fails on
zero or several matches. `rename_collections(folder, albums, confirm)` changes titles only:
`check_renames` (confirm, ids unique, titles non-empty and trimmed, new album titles unique), then a
fresh read that the folder and every album still carry their old title, that each album is
directly in the folder, that `canPerformEditOperation(.rename)` holds and that no other child has a
new title; then one `performChangesAndWait` with `PHAssetCollectionChangeRequest setTitle:` per
album and `PHCollectionListChangeRequest setTitle:` for the folder (skipped when the title stays);
then `verify_renamed` reads every title back. No asset is added or removed. Used by
[`albums --rename-from`](/design/cli.md#album-names).

[^objc2-photos]: objc2-photos 0.3.2 generated bindings: `PHFetchOptions`, `PHImageRequestOptions`, `PHPhotosError::NetworkAccessRequired = 3164`.
[^probe-run]: Probe and count runs on 2026-11-12; logs not kept (they name asset ids).
[^apple-avasset]: Apple developer documentation, `requestAVAsset(forVideo:options:resultHandler:)`, Discussion ("Photos downloads the video data (if necessary) and creates AVFoundation objects"), read 2026-11-12.
[^objc2-avf]: objc2-av-foundation 0.3.2 generated bindings: `copyCGImageAtTime:actualTime:error:` is marked deprecated ("Use generateCGImageAsynchronouslyForTime:completionHandler: instead"); `dynamicRangePolicy` default `ForceSDR`.
[^video-probe]: `video_probe --check-local [--fast]` and two 12-video `fetch-videos --pick spread|first` pilots into scratch data dirs (deleted), 2026-11-12.
