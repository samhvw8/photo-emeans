# emeans

Clusters every photo and video in the user's Photos library (this Mac and iCloud) by content.

1. `photos` fetches a resized copy of each photo through PhotoKit, and the poster plus up to 7
   sampled frames of each video (from a medium rendition), downloading from iCloud when needed.
2. `candle-embeddings` (imported as `encoder`) turns each image into a 768-d unit vector with its EmbeddingGemma 2 family (`google/embeddinggemma-2`), in candle on Metal.
   A video's vector is the L2-normalized mean of its frame vectors.
3. `emeans` clusters the vectors with e-means, a Rust/candle port of faiss `contrib/e_means`, plus
   improvements: a fused Metal assignment kernel, automatic k (two levels), incremental assignment
   of new photos, and near-duplicate grouping.
4. `bend/` holds a second e-means in Bend 2. It reads the same vector file, and `bench/` compares the two versions.
5. `photo-emeans` is the CLI that runs the steps, writes a local HTML gallery, and creates Photos
   albums only for clusters the user approves. The gallery's "Clean up" tab suggests a keeper per
   near-duplicate group; the marked rest can become one album to review and delete from in Photos.

## Layout

| Path | What |
|---|---|
| [`emeans`](https://github.com/samhvw8/emeans) (own repo) | e-means library, a git dependency pinned by commit in the root `Cargo.toml`: `emeans-core` (op contract on CPU, device loop, model), `emeans-metal`, `emeans-cuda`, `emeans-cluster` (auto-k, incremental, dedup, kNN), the `emeans` facade, `emeans-bench`. Its design, recipes and synthetic results are in its own `.okf/`; `.okf/design/core.md` here is the photo-tool view |
| [`candle-embeddings`](https://github.com/samhvw8/candle-embeddings) (own repo) | Embedding encoders in candle, a git dependency pinned by commit and renamed to `encoder` in `Cargo.toml` (`package = "candle-embeddings"`), so the code says `use encoder::`: the `Encoder` trait and shared blocks, one module per family under `src/families/` (EmbeddingGemma 2). Reads models from the local Hugging Face cache; `hf-hub` only behind its `download` feature, which this tool does not enable. Its own `.okf/` has the encoder design and the recipe for a new family |
| `crates/photos` | PhotoKit access via `objc2-photos`: list photos and videos, fetch resized JPEGs, fetch video frames (`video.rs`: `requestAVAsset` + `AVAssetImageGenerator`; `video_store.rs`: resumable `fetch_all_videos`), create albums, list a folder and rename its albums (titles only) |
| `crates/photo-emeans` | CLI binary: `fetch`, `fetch-videos`, `embed` (photos and video frames), `cluster` (`--k` / `--auto`, `--dedup`), `gallery` (albums tab, clean-up tab), `albums` (`--approved`, `--cleanup` or `--rename-from`), `names` (content titles from `labels.txt`), `add`; `cleanup.rs`: keeper rule and `cleanup.json` reader |
| `bend/` | e-means in Bend 2 |
| `bench/` | Parity and comparison harness (Python reference via `uv`, Rust vs Bend) |
| `reference/faiss_e_means/` | Upstream Python source, MIT, pinned commit in `UPSTREAM_COMMIT`. Read-only |
| `data/` | Personal data (gitignored). Path comes from `EMEANS_DATA` |

## Data contract

Every step reads and writes these files under `$EMEANS_DATA`. Change them only together with all readers.

| File | Format |
|---|---|
| `photos/<asset-uuid>.jpg` | JPEG, long side at most 1024 px, aspect kept, EXIF orientation applied |
| `photos/manifest.jsonl` | One JSON object per photo: `id` (PHAsset `localIdentifier`), `file`, `created` (ISO 8601), `width`, `height`, `favorite`, `burst_id` (or null), `subtypes` |
| `videos/<asset-uuid>/fNN.jpg` | Frames of one video: `f00` = poster frame, `f01..` sampled at the middles of equal slices of the duration. JPEG, long side at most 1024 px, upright. 8 frames from 4 s on, else two per second (at least 2) |
| `videos/manifest.jsonl` | One JSON object per video: `id`, `created`, `duration` (s), `width`, `height` (of the poster JPEG), `favorite`, `subtypes`, `frames: [{file, t, width, height}]` (`file` relative to `videos/`, `t` in seconds, null for the poster) |
| `videos/failures.jsonl` | `{id, error, at}` per video that `fetch-videos` could not fetch; skipped by later runs unless `--retry-failed` |
| `embed/embeddings.f32` | Raw little-endian f32, row-major, `n x 768`, each row L2-normalized. Photo rows and video rows (a video row = L2-normalized mean of its frame rows) |
| `embed/rows.jsonl` | Line i = `{"row": i, "id": ..., "kind": "photo" \| "video"}` for row i of `embeddings.f32`. Lines without `kind` (written before videos) are photos |
| `embed/frames/` | Frame store, same format as `embed/` (`embeddings.f32`, `rows.jsonl` with `kind: "frame"` and id `<asset id>#fNN`, `meta.json` equal to `embed/meta.json`) |
| `embed/runs.jsonl` | Per `embed` run an `event: start` line (`started`, `host`, `os`, `device` `metal`/`cpu`/`cuda`, `compute_dtype`, `media`) and, when it finishes, an `event: end` line that adds `ended`, `seconds`, counts of photos, frames and videos written, `lock_wait_seconds`. A start without an end = the run was killed |
| `embed/meta.json` | `{model, revision, d, n, dtype: "f32", compute_dtype, prompt, created, updated}`. `compute_dtype` (`f32`/`bf16`) and `prompt` (null for images) must match to append |
| `embed/failures.jsonl` | `{id, file, error, at}` per photo, or per video (with the frame `file`), that could not be decoded; skipped by later runs |
| `clusters/<run>/` | `labels.i32` (raw LE i32 per embed row, photo or video, index into `centroids.f32`, -1 = none), `centroids.f32` (`k x d`), `sim.f32` (cosine to own centroid, per row), `report.json` (mode, `n`, `n_videos`, `ids_fnv` of the first `n` embed ids, sizes, timings, recipe) |
| `clusters/<run>/` with `--auto` | `labels.i32` = level-1 index; `labels_l2.i32` (leaf index per row) + `centroids_l2.f32` (leaves x d); `outlier.u8` (1 byte per row); `model/` (saved `Hierarchy`); `report.json` `auto.groups` / `auto.leaves` map indices to stable ids, `history` lists `add` calls |
| `clusters/<run>/` with `--dedup` | `dedup.i32` (each row's representative row), `dedup.json` (`emeans::DedupResult`) |
| `clusters/<run>/gallery/` | `index.html` (data inline) + `thumbs/<file>` (256 px, hard links into `thumbs/256/`; a video's thumbnail is `v-<asset-uuid>.jpg`, made from its poster) |
| `approved.json` (from the gallery) | `[{cluster, title, asset_ids}]` (photo and video ids), input of `albums` |
| `cleanup.json` (from the gallery's clean-up tab) | `[{asset_id, file, group, reason}]`, one entry per photo or video marked for removal in a reviewed group: `file` = its image relative to `$EMEANS_DATA` (`photos/<uuid>.jpg` or `videos/<uuid>/f00.jpg`), `group` = index in the run's `dedup.json`, `reason` free text. Input of `albums --cleanup`, which requires every id and `file` to match the manifests |
| `clusters/<run>/album-names.json` (from `names`) | `{version: 1, created, approved, model, prompt, labels_source, labels_fnv, labels, method: {center, min_share, min_margin, second_share}, folder: {local_id, old_title, new_title}, unchanged: [title], albums: [{cluster, local_id, old_title, new_title, count, mixed, margin, top: [{label, share, mean, lift}]}]}`. The rename plan and its undo record: `albums --rename-from` fills `local_id` (PhotoKit local identifiers) on its first run and renames old -> new; `--reverse` renames new -> old. Do not overwrite it after a rename (`names` refuses without `--force`) |
| `names/label-vectors.jsonl` | Cache of label text vectors: `{key, text, v}` per line, `key` = model id, revision, dtype and prompt. Safe to delete (re-embedded on the next `names`) |

## Commands

| Task | Command |
|---|---|
| Format, lint, test | `mise run check` |
| Tests only | `mise run test` |
| Encoder exactness gate (Metal GPU; in a checkout of the `candle-embeddings` repo, needs `bench/gemma/ref` from `uv run --script bench/gemma/reference.py`) | `mise run exact` in that checkout |
| Fetch frames of a test set of videos (resumable; downloads medium renditions into Photos' own cache, ~0.35 MB per second of video; stops below 8 GB free disk; Ctrl-C stops cleanly) | `mise run pe -- fetch-videos --limit 60 --sample spread` |
| Fetch every video (tens of GB of Photos cache for a large library; only when the user asks; see `.okf/results/video-test-run.md`) | `mise run pe -- fetch-videos --all [--fast-over 60]` |
| Embed everything not embedded yet (photos, then video frames; resumable, safe to kill) | `mise run embed` |
| Embed on the shared Mac GPU in chunks | `mise run pe -- embed --gpu-lock /tmp/emeans-mac-gpu.lock --lock-rows 500` |
| Any CLI command | `mise run pe -- cluster --auto --dedup`, `mise run pe -- gallery --run auto-dedup` |
| Album plan (dry run; only `--yes` writes, after the user approves the exact list) | `mise run pe -- albums --approved approved.json` |
| Clean-up album plan (dry run: one album of the marked near-duplicates; nothing is deleted) | `mise run pe -- albums --cleanup cleanup.json` |
| Album lists from a run without the gallery (writes the list, prints the dry-run plan; create with `albums --approved <file> --yes` after approval) | `mise run pe -- albums --run auto-dedup --all-clusters --duplicates` |
| Content names for created albums (label text embedding on the GPU: run through quiet-run with the lock; writes `album-names.json`) | `mise run pe -- names --approved data/clusters/auto-dedup/approved-from-run.json` |
| Rename plan (dry run; records local ids) / rename / undo | `mise run pe -- albums --rename-from data/clusters/auto-dedup/album-names.json [--yes]`; undo: add `--reverse` |

CLI details: `.okf/design/cli.md`.

## Rules

- Personal data (photos, video frames, embeddings, labels) stays on your own machines: the Mac that runs the tool (`$EMEANS_DATA`, `data/` by default, gitignored) and, if you use one, your own Linux GPU box (same layout, gitignored). Never upload it to any other host or service, and never commit it. See `.okf/decisions/data-location.md`.
- Never write to the Photos library without the user's approval of the exact albums (or exact renames). Album creation and renames sync to iCloud.
- Never delete or edit a photo or video. Clean-up only collects marked photos into an album; the user deletes them in the Photos app.
- Fetch resized copies, never originals. For videos, fetch a medium rendition and keep only the frames; never write a video file. Never change or delete assets.
- The Mac is a 16 GB M1 Pro that is often busy. Memory-map weights, keep batches small, and run long jobs in the background.
- The Mac is the user's workstation: a heavy CPU or GPU job freezes their UI. Start every Mac benchmark, Metal test run or embedding through `scripts/quiet-run.py --lock /tmp/emeans-mac-gpu.lock -- <cmd>` (waits until CPU busy is at most 40%, runs at nice 15, holds the GPU lock). Keep each run short: chunks of minutes, not hours. Long benchmarks belong on a Linux GPU box.
- Correctness first: the Rust e-means must match the faiss reference within tolerance (`bench/`) before you optimize it.
- Crashing beats corrupting: a partial embedding run must resume or fail loudly, never write a short or misaligned file.

Project knowledge lives in `.okf/`. Update it in the same change as the code it describes, and run the validator.
