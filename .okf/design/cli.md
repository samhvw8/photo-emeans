---
type: Design
title: photo-emeans CLI — commands, crash-safe embedding, gallery approval, clean-up and album naming flows
description: How crates/photo-emeans runs the pipeline (fetch, fetch-videos, embed, cluster, gallery, albums, add) for photos and videos, why the embed stores survive a kill at any point, what a cluster run writes, how a cluster becomes an album only after the user marks it in the local gallery, how the gallery's clean-up tab collects near-duplicates into one review album (nothing is ever deleted), and how `names` titles albums by content and `albums --rename-from` renames them with an undo record.
resource: crates/photo-emeans
tags: [cli, embed, resume, crash-safety, cluster, gallery, albums, incremental, video, cleanup, dedup, naming, rename]
status: draft
generated:
  by: claude-code/claude-opus-5-5
  at: '2026-11-13T05:00:00Z'
sources:
  - id: cli-tests
    resource: crates/photo-emeans/tests/
    title: embed_store.rs (resume and alignment), pipeline.rs (cluster, gallery, add, albums dry run, clean-up), video_embed.rs and video_pipeline.rs (videos)
  - id: apple-wait
    resource: https://developer.apple.com/documentation/photos/phphotolibrary/performchangesandwait(_:)
    title: Apple, PHPhotoLibrary.performChangesAndWait(_:) (Discussion)
  - id: apple-async
    resource: https://developer.apple.com/documentation/photos/phphotolibrary/performchanges(_:completionhandler:)
    title: Apple, PHPhotoLibrary.performChanges(_:completionHandler:) (Discussion)
  - id: cleanup-check
    resource: crates/photo-emeans/src/gallery.html
    title: Headless Chrome (CDP) check of the clean-up tab on a synthetic 233-photo library and on the real auto-dedup gallery, 2026-11-12 (scratch scripts, not kept)
  - id: names-tests
    resource: crates/photo-emeans/src/names.rs
    title: Unit tests in names.rs (vocabulary, votes, centering, top-k, repeated words, titles, cache) and albums.rs (rename plan forward, reverse, refusals)
  - id: kill-run
    resource: crates/photo-emeans/src/embed.rs
    title: SIGKILL-and-resume run on 30 real photos on Metal, 2026-11-12 (scratch data, not kept)
---

# Commands

All paths are under `$EMEANS_DATA` (`mise.toml` sets it to `data/`; `--data` overrides it). The
file formats are the data contract in `AGENTS.md`.

| Command | Does |
|---|---|
| `fetch [--concurrency 6] [--limit N]` | `photos::fetch_all` into `photos/`, from `main()` on the main thread ([PhotoKit](/design/photokit.md): results arrive on the main queue). Last line `fetched N (...)` marks the end for `embed --follow` |
| `fetch-videos (--limit N [--pick\|--sample spread\|first] \| --all) [--concurrency 4] [--rendition medium\|fast] [--fast-over SECS] [--max-duration SECS] [--frames 8] [--retry-failed] [--min-free-gb 8]` | `photos::fetch_all_videos` into `videos/` (poster + sampled frames, [PhotoKit videos](/design/photokit.md#videos)), on the main thread. Refuses to run without `--limit N` or `--all`: every video is an iCloud download into the Photos cache. `--limit N --pick spread` (default) takes N videos evenly spaced by creation date, the same set every time; already fetched ones count. Last line `fetched N videos (...)`. Measured: [video test run](/results/video-test-run.md) |
| `embed [--media all\|photos\|videos] [--dtype f32\|bf16] [--batch 1] [--chunk 32] [--limit N] [--follow] [--retry-failed] [--device metal\|cpu] [--gpu-lock FILE --lock-rows 500] [--nice 10]` | Embeds every manifest photo not yet in `embed/`, then every video frame not yet in `embed/frames/`, with the [EmbeddingGemma 2 family](/design/embeddinggemma2.md), no prompt. Renices itself to 10. `--limit` counts photos, then frames |
| `cluster (--k K \| --auto) [--dedup] [--backend cpu] [--run NAME] [--force]` | Writes `clusters/<run>/`. `--k`: plain spherical e-means. `--auto`: the two-level hierarchy of [automatic k](/design/autok-incremental-dedup.md). `--dedup`: one representative per near-duplicate group is clustered, members follow it |
| `gallery --run NAME [--per-cluster 40]` | Writes `clusters/<run>/gallery/index.html` and thumbnails |
| `albums --approved approved.json [--folder emeans] [--dry-run \| --yes]` | Prints the album plan; only `--yes` calls `photos::create_albums` |
| `albums --run NAME (--all-clusters \| --duplicates \| both) [--out FILE] [--duplicates-title T] [--folder emeans]` | Builds the lists from a run, without the gallery ([lists from a run](#lists-from-a-run)): writes them in the `approved.json` format to `--out` (default `clusters/<run>/approved-from-run.json`) and prints the dry-run plan. Refuses `--yes`; `albums --approved FILE --yes` creates them after review |
| `albums --cleanup cleanup.json [--title "emeans · to delete (review)"] [--folder emeans] [--dry-run \| --yes]` | Prints the plan for one album of the photos marked in the [clean-up tab](#clean-up-tab); only `--yes` creates it. Nothing is deleted |
| `names --approved FILE [--labels FILE] [--out FILE] [--force] [--folder emeans] [--folder-to "Photo groups"] [--top-k 3] [--min-share 0.4] [--second-share 0.4] [--min-margin 0] [--no-center] [--device metal\|cpu]` | Names every cluster album of FILE by content ([album names](#album-names)) and writes `album-names.json` (default: next to FILE). Reads embeddings and metadata only; never touches Photos. Refuses to overwrite an existing file without `--force` |
| `albums --rename-from album-names.json [--reverse] [--dry-run \| --yes]` | Finds the folder in Photos (read-only), matches every album by its current title, records the local identifiers in the file and prints the plan; only `--yes` renames (titles only). `--reverse` puts the old titles back |
| `add --run NAME [--no-embed] [--backend cpu]` | Runs `embed`, then `Hierarchy::add_photos` (with dedup) or `Hierarchy::add` on an `--auto` run, and rewrites the run |

`fetch` and `fetch-videos` stop cleanly on SIGINT or SIGTERM (`photos::stop`): the first signal
stops new downloads, the assets in flight finish and get their manifest lines, and the command
fails with `stopped by signal N`. A second signal exits at once. Files are renamed into place
before their manifest line, so even that leaves no partial entry. Tested in
`crates/photos/tests/stop_signal.rs` (raises SIGTERM and SIGINT in-process) and
`crates/photo-emeans/tests/cli_guard.rs` (the `--limit`/`--all` refusal).

mise: `mise run embed` and `mise run pe -- <command ...>` build into `target/run` first, so a
development build never replaces the binary of a running job.

Default backend for `cluster` and `add` is `cpu` (Accelerate): the `--auto` sweep runs hundreds
of small e-means runs, where the Metal assigner's per-batch overhead dominates, and the GPU stays
free for a running `embed`. macOS can kill every GPU command buffer when one process faults.

# Embed store: crash safety

One write order makes every crash state repairable:

1. Append the chunk's f32 rows to `embeddings.f32`, fsync.
2. Append the matching `rows.jsonl` lines (`{"row": i, "id": ...}`), fsync.
3. Rewrite `meta.json` (with the new `n`) through a temporary file, fsync, rename, fsync the directory.

A kill can therefore leave only: a torn last line in `rows.jsonl`, whole f32 rows without lines,
or a partial f32 row. On start, `EmbedStore::open` takes the shorter of the two files, truncates
both to that many whole rows, fsyncs, and prints a line starting with `!!!` that names what it
dropped. The next pass re-embeds the dropped photos.

Everything else stops the run instead of being "repaired", because it is not a crash artifact:

| Condition | Result |
|---|---|
| A bad line in the middle of `rows.jsonl`, a row number out of order, a duplicate id | error |
| A stored row that is not finite or not unit length (tolerance 1e-3) | error |
| `meta.json` from another model, revision, compute dtype or prompt | error: vectors from two models would cluster as if comparable |
| Data files but no `meta.json` | error: the model that made them is unknown (`meta.json` is written before the first row) |
| A second `embed` while one runs | error: exclusive `flock` on `embed/.lock` |

Readers (`store::load`, used by `cluster`, `gallery`, `add`) do not repair. They take the
complete `rows.jsonl` lines as `n` and need at least `n` f32 rows. f32 is written first, so a
reader gets a consistent prefix while an embed run appends.

Other embed rules:

- The manifest is read without writing (`photos::load_manifest` would rewrite it and read every
  JPEG): complete lines only, first of duplicate ids, a corrupt middle line stops the run. So
  `embed` can run while `fetch` still appends.
- One thread decodes and preprocesses the next photos while the GPU runs the model.
- A photo that cannot be opened or preprocessed goes to `embed/failures.jsonl` and is skipped by
  later runs unless `--retry-failed`. A model error is fatal: the rows computed before it are
  committed, then the run stops.
- `--follow` polls the manifest every 10 s; it stops when the fetch log's last line (after any
  `=== ... DONE` lines a wrapper adds) is the `fetched ...` summary or an `Error: ...` line, or
  after `--idle-secs` (300) without new lines. Photos watch `fetch.log`, videos
  `fetch-videos.log`.
- On the user's Mac, run `embed` through `scripts/quiet-run.py --lock /tmp/emeans-mac-gpu.lock --
  ... embed --limit 300 --nice 0` (`AGENTS.md`): quiet-run waits for a quiet CPU, holds the GPU
  lock and sets nice 15. Do not also pass `--gpu-lock` with the same file: a second `flock` in the
  child waits forever on the parent's lock. `--nice 0` keeps nice 15 (the default renice to 10
  would need privileges and fail).
- `--gpu-lock FILE` takes an exclusive `flock(2)` on FILE before the first model call and releases
  it after every `--lock-rows` committed rows (then pauses 2 s, so a waiting job gets it). The
  shared Mac lock is `/tmp/emeans-mac-gpu.lock`, the same file `/tmp/emeans-flock.py` locks.
  Waits over 5 s are logged. The model stays loaded between chunks.
- Every run appends a `start` line to `embed/runs.jsonl` (device, dtype, host) and an `end` line
  with counts and lock wait when it finishes, so the device of each part of the store is on record,
  also for a killed run.

## Videos

`embed` handles a video in two stores (data contract in `AGENTS.md`):

1. Each frame `videos/<uuid>/fNN.jpg` is embedded like a photo and appended to the frame store
   `embed/frames/` (same format and crash rules as `embed/`; id `<asset id>#fNN`,
   `kind: frame`). Its `meta.json` must equal the main store's (model, revision, compute dtype,
   prompt).
2. In the same commit callback, every video whose frames are now all in the frame store gets one
   row in `embed/`: the L2-normalized mean of its frame vectors (f64 accumulation), `kind: video`.
3. On start, a video whose frames are all stored but which has no main row (a kill between the two
   writes) gets its row without a model call.

`rows.jsonl` lines gain `"kind": "photo" | "video"`. Lines written before videos have no `kind`
and read as photos; nothing is rewritten, so `ids_fnv` of existing runs stays valid, and a run
made before videos were added still builds its gallery (it covers the first `n` rows).

A frame that cannot be read fails its whole video: one line in `embed/failures.jsonl` with the
video id and the frame file; later runs skip the video unless `--retry-failed`.

Measured on the Mac (Metal, f32, batch 1, the GPU lock shared with benchmark jobs): 1.18-1.20
frames/s.

Measured:[^kill-run] 30 photos, Metal, f32, batch 1, load average ~40: 36 s (0.83 photos/s), peak
footprint 2.8 GB. A run killed with SIGKILL at 16 rows and resumed gave 30/30 vectors bitwise
equal to a clean run. The unit tests cover each repair path with a fake model.[^cli-tests]

# Cluster run

`cluster` builds `clusters/.<run>.tmp/` and renames it into place; `--force` swaps an existing
run out to `.<run>.old` and deletes it. That delete retries 4 times; if it still fails (seen once:
"Directory not empty" while Spotlight indexed the old gallery's thumbnails), the command logs a
warning and succeeds, because the new run is already in place, and the next commit deletes the
old version first (unit test in `cluster.rs`). `add` rebuilds the same way.

| File | `--k` | `--auto` |
|---|---|---|
| `labels.i32` | cluster index per photo | level-1 index per photo |
| `centroids.f32` | `k x d` | level-1 centroids, `k1 x d` |
| `labels_l2.i32`, `centroids_l2.f32` | | leaf index per photo, leaf centroids |
| `sim.f32` | cosine of each photo to its own (leaf) centroid | same |
| `outlier.u8` | | 1 = below its leaf's outlier floor (median minus 3 robust deviations of the leaf's member cosines; [outlier floor](/design/autok-incremental-dedup.md#outlier-floor)) |
| `dedup.i32`, `dedup.json` | with `--dedup`: each photo's representative row, and the `DedupResult` (with each group's `anchors` row) that `add` extends | same |
| `model.emeans` / `model/` | trained e-means model | saved `Hierarchy` (EMA state, stable ids) |
| `report.json` | mode, `n`, `n_videos`, `ids_fnv`, embed meta, backend, recipe, iteration stats, cluster sizes and mean similarity, timings, dedup stats | plus `auto`: config, chosen k, groups and leaves with their stable ids, diagnostics; `history` of `add` calls |

Labels are indices into the centroid files; the stable ids that survive `add` are in
`report.json` (`auto.groups[].id`, `auto.leaves[].l1_id/l2_id`). `ids_fnv` is FNV-1a 64 of the
first `n` embed ids; every reader checks it, so a run never pairs with different rows.

`sim` is recomputed for every photo against its centroid, including photos behind a dedup
representative.

# Gallery and the approval flow

`index.html` is self-contained: data inline as JSON, no network, no CDN, no web fonts.
Thumbnails (long side 256 px, JPEG q80) are made once into `$EMEANS_DATA/thumbs/256/` with rayon
and hard-linked into `gallery/thumbs/`. The viewer loads the 1024 px copy from
`../../../photos/` (or the video's poster from `../../../videos/`).

Videos are rows like photos. Each page row is `[thumb, id, date, favorite, outlier, image path,
null | [duration, [frame paths]]]`. A video tile shows its poster (thumbnail `v-<uuid>.jpg`) with a
play badge and its duration (`m:ss`). The viewer shows the poster with a strip of the sampled
frames; a click on a frame shows it large. Card counts read "N photos · M videos", and the summary
bar counts both.

- One section per level-1 group (for `--auto`), one card per leaf; flat runs show one card per
  cluster. Sort by size, date or likeness; filter all, marked or unmarked.
- A card shows size, date range, mean cosine to the centroid ("alike") and the first 40
  thumbnails, most typical first. "Show all" adds 200 at a time. Grids are only built when the card
  comes within 1200 px of the viewport, and images load lazily, so 16k photos stay fast.
- Near-duplicate stacks show one thumbnail with a count; the count spreads the members. Outliers
  have a dotted frame.
- Mark a card (or a whole group) with "Album". The title field is prefilled with the date range;
  a marked card gets a red grease-pencil loop around its title.
- "Download approved.json" writes `[{cluster, title, asset_ids}]`, with every photo and video of
  the cluster (stack members included); the status line counts photos and videos. It refuses empty or duplicate titles and focuses the card.
  "Load approved.json" restores the marks. Marks are also kept in localStorage per run and
  `ids_fnv`; the file is the record.
- Light and dark follow `prefers-color-scheme`; layout works down to 390 px.

`albums` reads that file, applies the same checks as `photos::albums::check_request`, requires
every id to be in the photo or video manifest, and prints the plan (with the number of videos per
album and in total; PhotoKit adds a video to an album by its local identifier like a photo) (folder, albums, counts, date ranges, photos in
more than one album). Without `--yes` nothing is written. `--yes` created the `auto-dedup`
albums on a real library (folder "emeans", about 300 albums).

## Clean-up tab

A run with `dedup.json` gets a second tab, "Clean up" (`#clean` in the URL keeps it across
reloads). The tab lists every near-duplicate group of 2 or more photos (`crates/photo-emeans/src/cleanup.rs`),
largest first. The page data has `cleanup: {anchors_stored, groups: [{g, anchor, shots, rows,
keep, px, shot}]}`: `rows` ranked by the keeper rule, `keep` the rows that start as Keep, `px` the
stored copy's `[width, height]`.

Keeper rule, in order:

1. A favorite.
2. The larger pixel area of the stored copy (manifest `width x height`). Copies are at most 1024 px
   on the long side, so this ranks originals smaller than that, and crops. The original's size is
   not in the manifest.
3. The group's anchor (`DedupResult::anchor`; a `dedup.json` written before anchors were stored
   has none, and the representative stands in, as in `auto-dedup`).
4. The lower row.

The keeper and every other favorite start as Keep; the rest start as Remove. A favorite is never
pre-marked for removal.

| Key | Does |
|---|---|
| `j` / `↓`, `k` / `↑` | Next group (marks the current one reviewed), previous group |
| `h` `l` / `←` `→` | Select a photo in the group |
| `x` / Space, or a click on the photo or its label | Switch Keep / Remove |
| `1`-`9`, Enter | Keep only that photo (Enter: the selected one), go to the next group |
| `a` | Keep all, go to the next group |
| `s` | Back to the suggested marks |
| `v`, or the ⤢ button | Large view (1024 px copy); arrows step through the group, `x` switches, Esc closes |

- Filters: section Photos or Screenshots (a group is a screenshot group when every member has the
  `screenshot` subtype; mixed groups stay under Photos with a SCREEN badge), size (2, 3-5, 6+),
  and All or To review. "Accept the suggestions of every shown group" marks the shown groups
  reviewed after a confirm.
- The summary counts reviewed groups, photos marked (all, and in reviewed groups), kept photos,
  and reviewed groups that keep nothing. There is no space-freed estimate: the manifest has no
  original file sizes, and the stored copies are resized.
- Marks and reviewed state stay in localStorage per run and `ids_fnv` (convenience only).
- "Download cleanup.json" writes the Remove marks of **reviewed** groups only:
  `[{asset_id, file, group, reason}]` (data contract in `AGENTS.md`). `reason` is
  `near-duplicate of <asset id of the first kept photo>`, or `whole group marked, nothing kept`.
- "Load cleanup.json" sets every group named in the file to the file's marks, reviewed. Other
  groups keep their marks. A group with nothing marked is not in the file, so after a load in a
  fresh browser it shows its suggestion again, unreviewed.

`albums --cleanup` parses the file (refuses an empty list, unknown fields, an empty or repeated
id), requires every id to be in the photo or video manifest with the same `file`, runs
`photos::albums::check_request`, and prints one line per plan plus "Nothing is deleted. The album
only collects the marked photos; review it in Photos and delete there." Video ids are allowed.
`--yes` goes through the same `create` path as `--approved` (one folder, one album). Nobody has run
`--yes` on the library.

Checked in headless Chrome over CDP on a synthetic library (every key, click toggles, filters,
accept, reload persistence, load of a downloaded file, the albums tab still marks clusters) and on
the real `auto-dedup` gallery with images blocked, no console errors.[^cleanup-check] On that
library of about 16k photos, almost half of the photos sit in a near-duplicate group of 2 or more;
about two thirds of the groups are pairs or triples (largest 40); about 6% of the groups are all
screenshots; and about a third of all photos start pre-marked for removal (extra favorites in a
group stay Keep).

## Lists from a run

`albums --run NAME` writes album lists without the gallery (`albums::run_lists`):

- `--all-clusters`: one album per cluster (`--k`) or leaf (`--auto`), the same keys as the gallery
  cards, every member included (stack members follow their representative's label; videos only
  when the run has them). Oldest first. Title: the members' capture months and the count,
  `2019-06 → 2019-08 · 46`, `2019-06 · 12` for one month, `undated · 3` without dates. A repeated
  title gets ` #2`, ` #3` (`albums::unique_titles`).
- `--duplicates`: one album, "emeans · duplicates to review", with every member of every
  near-duplicate group of 2 or more that does not start as Keep under the [keeper rule](#clean-up-tab).
  So it never holds a keeper, nor another favorite of the group; `run_lists` checks this before it
  returns.

Both lists go into one folder (`--folder`, default `emeans`) through one `create_albums`
transaction when the written file is passed to `albums --approved FILE --yes`. Test:
`cleanup_gallery_and_albums_dry_run` (every row in exactly one cluster album, unique titles, the
duplicates album equals the gallery's pre-marked set, the written file passes `--approved`).

## Album names

`names` (`src/names.rs`) gives each cluster album a content title such as
`Beach & Sunset · 2019–2024 · 65`. It works from the stored image vectors only; it never opens a
photo.

1. **Vocabulary.** `crates/photo-emeans/labels.txt` (built into the binary; `--labels` overrides
   it): one everyday concept per line, `Name | text to embed`. Lines with the same name are
   averaged (prompt ensemble). Edit it freely.
2. **Label vectors.** Each text goes through the EmbeddingGemma 2 text side with the search-query
   prompt `task: search result | query: `, the prompt the encoder's image-to-caption check uses
   (images are embedded without a prompt). Vectors are cached in
   `$EMEANS_DATA/names/label-vectors.jsonl` under model, revision and prompt; the model loads only
   for texts not in the cache. Run it through `scripts/quiet-run.py --lock /tmp/emeans-mac-gpu.lock`.
3. **Votes.** Every embedded row (photos and videos) is scored against every label (cosine). Each
   label's library mean is subtracted (`--no-center` turns this off), so a label that scores high on
   everything does not win every vote. Each member of an album votes for its `--top-k` best
   labels, so near-synonyms ("Office", "Coworkers", "Meeting") do not split one theme's votes; a
   label's share is the fraction of members that have it among their `--top-k`.
4. **Name.** Labels are ranked by share, then by mean lift (mean cosine minus the library mean).
   The top label needs `--min-share` (0.4); below that the album is `Mixed (<top label>)`. A
   second label joins at `--second-share` (0.4), unless its name repeats the first's ("Sign" and
   "Signboard", "City" and "Night City"); then the next label that qualifies is used. Label names
   may not contain `&` or `·`. `--min-margin` (default 0, off) makes the named labels lead the best
   label left out by that share, else the name falls back to one label, then to Mixed. It is off
   because top-3 votes tie near-synonyms by design: at 0.1, coherent albums whose two or three
   near-synonym labels had shares within a few points of each other became Mixed.
5. **Title.** `<name> · <first year>–<last year> · <count>` (one year: `2021`; no dates: `<name> ·
   <count>`). A repeated title gets ` #2`, ` #3`; titles never repeat the album that stays
   unchanged (the duplicates album, cluster key `duplicates`).

`album-names.json` (data contract in `AGENTS.md`) is the plan and the undo record. `albums
--rename-from` then:

1. finds the one top-level folder titled `folder.old_title` (`new_title` with `--reverse`) and lists
   its albums (`photos::find_folder`, read-only, with the asset count of each);
2. matches every album by its current title, which must be unique in the folder, and checks any
   local identifier the file already records; a new title may not equal an album that is not
   renamed; a changed asset count is printed as a note;
3. writes the folder's and albums' local identifiers into the file (first run), prints the plan;
4. with `--yes`, calls `photos::rename_collections` on a worker (same main-thread pumping as
   `create`): it checks again that the folder and every album still carry the expected title, that
   each album is directly in the folder and that Photos allows renaming it, then sets every title
   in one `performChangesAndWait` (`PHAssetCollectionChangeRequest setTitle:`,
   `PHCollectionListChangeRequest setTitle:`), and reads every title back. No asset is added or
   removed.

Undo: `albums --rename-from <file> --reverse` (dry run), then the same with `--yes`.

Applied with the user's approval to `auto-dedup` on a real library: folder "emeans" became "Photo
groups", every cluster album (about 300) got a content title; the `--reverse` dry run matches all
of them by new title and recorded id. Checked with `osascript` (read-only): every album in "Photo
groups" has its planned title and media count, "emeans · duplicates to review" unchanged.

### Tuning on `auto-dedup`

Measured on the cluster albums of a real library (about 300 albums, about 16k photos) with the 433
shipped labels. Only metadata was
checked: the `screenshot` subtype and capture months; no photo was opened.

Shares are of all cluster albums, except the last two columns:

| Variant | Mixed | Two-label names | Albums ≥80% screenshots with a screen or document name | Non-screenshot albums named "Screenshot" |
|---|---|---|---|---|
| top-1, raw cosine, share 0.3, margin 0.1 | 32% | 25% | 78% | under 1% |
| top-1, centered, share 0.3, margin 0.1 | 31% | 20% | 78% | under 1% |
| top-3, centered, share 0.4, margin 0.1 | 24% | 45% | 78% | none |
| top-5, centered, share 0.5, margin 0.1 | 30% | 39% | 78% | none |
| **top-3, centered, share 0.4, margin 0 (default)** | **11%** | 61% | 78% | none |

The non-default rows were measured before the repeated-word rule for the second label. The few
screenshot albums without a screen name hold screenshots of everyday scenes, so their content
names fit. Default run: top-label share min 0.19, median 0.71, max 1.00; about 27% one-label, 61%
two-label and 11% Mixed albums (about 13% of the photos); about 140 distinct top labels; all titles
unique, no suffix.

## Threading of `albums --yes`

`photos::create_albums` uses `PHPhotoLibrary performChangesAndWait:error:`. Apple's page for
`performChangesAndWait(_:)` says: "Do not call this method from the main thread. [...] Photos may
need to prompt the user to perform changes, so this method can block execution
indefinitely."[^apple-wait] (The asynchronous `performChanges(_:completionHandler:)` runs its
blocks "on an arbitrary serial queue".[^apple-async]) PhotoKit delivers other results on the main
queue ([PhotoKit](/design/photokit.md)), so a main thread that blocks in `join` can starve it.

So `albums --yes`, from `main()` on the main thread:

1. checks it is on the main thread, then `photos::ensure_authorized`;
2. runs `create_albums` on a worker thread through `mainthread::run_pumping`, while the main
   thread runs its run loop in 20 ms slices (the `fetch_all` arrangement);
3. stops waiting after `--timeout-secs` (600). The worker cannot be cancelled, so the error says
   to look for the folder in Photos before running `--yes` again (the change is one transaction,
   all or nothing).

`tests/albums_threading.rs` (`harness = false`, so its `main()` is the process main thread) checks
the wrapper with a fake call that blocks its worker until a block it dispatched to the main queue
has run: with pumping it returns in ~30 ms; with a main thread that only sleeps it times out (the
control); off the main thread it is refused; a panic and a slow call are handled.

# Tests

`cargo test -p photo-emeans` (CPU only): embed resume after an f32-ahead crash, after a torn rows
line and after a model error (each resumes to the clean result); refusals (dtype mix, second
writer, non-unit row, bad middle line, missing meta); a growing manifest with `--follow`; `--k`
and `--auto --dedup` run files and their consistency; gallery data and thumbnails; `add` of 40
photos; `albums` plan parsing and dry run; the keeper rule, the clean-up groups from a dedup
result and the `cleanup.json` reader (`src/cleanup.rs`); `cleanup_gallery_and_albums_dry_run`
(a `--k --dedup` run: 12 stacks, keeper by favorite, by larger copy and by anchor, a screenshot
group, `albums --cleanup` dry run, a mismatched `file` refused).[^cli-tests]

Videos (`tests/video_embed.rs`, fake model): video rows equal the normalized mean of their stored
frames; resume after `--limit` and after a kill between the frame commit and the video row gives
the same rows as a clean run; a broken frame skips its video (and `--retry-failed`); old `rows.jsonl`
lines without `kind` read as photos; the joined photo/video view refuses an id in both manifests;
the model never runs without the GPU lock held, and the lock is free after the run.

`tests/video_pipeline.rs` runs on sample media made by `ffmpeg` (skipped with a message when
`ffmpeg` is missing): three clips (6 s, 3 s with a 90° display rotation, 1.5 s) go through
`photos::frames_from_file` (the real AVFoundation path: frames upright, at most 1024 px), the
`videos/` writer, `run_videos`, `cluster --dedup --dedup-max-gap 600`, `gallery` (video rows,
`v-<uuid>.jpg` thumbnails, frame paths that resolve) and `albums` dry run with video ids. A run
made before the videos were added still builds its gallery.

Related: [pipeline](/design/pipeline.md), [output decision](/decisions/output.md).

[^cli-tests]: `tests/embed_store.rs`, `tests/pipeline.rs`, `tests/albums_threading.rs`, unit tests in `src/albums.rs`, `src/cleanup.rs`, `src/manifest.rs`, `src/util.rs`.
[^cleanup-check]: `bun` script driving Chrome through the DevTools protocol (`Input.dispatchKeyEvent`, `DOM.setFileInputFiles`); for the real gallery `Network.setBlockedURLs` blocked every `.jpg`, so no photo was loaded.
[^kill-run]: Copies of 30 photos from `data/photos` in a scratch directory; the two stores compared by id in Python (alignment, row numbers, meta `n`, vectors).
[^apple-wait]: Apple developer documentation, `performChangesAndWait(_:)`, Discussion, read 2026-11-12.
[^apple-async]: Apple developer documentation, `performChanges(_:completionHandler:)`, Discussion, read 2026-11-12.
