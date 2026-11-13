# photo-emeans

Group every photo and video in your Apple Photos library by what is in it. photo-emeans looks at
each picture, puts similar ones together and turns each group into an album, such as
`Beach & Sunset · 2019–2024 · 65`, inside one **Photo groups** folder. Near-duplicates (bursts,
retakes, re-saved copies) go into one extra album, so you can review them and delete the ones you
do not want in Photos.

- **Your photos never leave your Mac.** Fetching, embedding and clustering all run locally on
  Apple Silicon. Nothing is uploaded; there is no account and no cloud API.
- **Nothing is deleted or edited.** The tool only creates albums and sets album titles. It
  prints the exact list first and writes to Photos only when you rerun the command with `--yes`.
- **Every step can be undone or resumed.** Album titles have an undo command. Fetching and
  embedding can be stopped at any point and pick up where they left off.

It builds on two libraries: [emeans](https://github.com/samhvw8/emeans) (fast k-means for
embeddings, a Rust port of faiss `contrib/e_means`) and
[candle-embeddings](https://github.com/samhvw8/candle-embeddings) (EmbeddingGemma 2 image
embeddings in Rust on Metal).

> Version 0.1, macOS only. Build from source. Command names and file formats may change.

## Contents

[How it works](#how-it-works) ·
[Requirements](#requirements) ·
[Quickstart](#quickstart) ·
[How long it takes](#how-long-it-takes) ·
[What it writes to Photos](#what-it-writes-to-photos) ·
[Limits](#limits) ·
[Docs](#docs) ·
[License](#license-and-credit)

## How it works

1. **Fetch.** A resized copy of every photo (long side at most 1024 px) comes from Photos through
   PhotoKit, downloaded from iCloud when the Mac has no large enough local copy. Originals are
   never downloaded. For videos, it keeps a poster frame and up to 7 sampled frames.
2. **Embed.** EmbeddingGemma 2 turns each image into a 768-number vector that describes its
   content, on the Mac's GPU. A video's vector is the mean of its frames.
3. **Cluster.** emeans groups near-duplicates first, then picks the number of groups by itself
   (automatic k in two levels), so you do not have to guess how many albums you want.
4. **Review.** A local HTML gallery shows every group and suggests which near-duplicate to keep.
5. **Albums.** One album per group plus one album of near-duplicates, created in one Photos
   transaction after you approve the list.
6. **Names.** Each album gets a content title from a fixed list of 433 everyday labels
   ([`labels.txt`](crates/photo-emeans/labels.txt)), scored against the stored vectors only.

All copies, vectors and cluster runs stay in `data/` (gitignored) inside your checkout.

## Requirements

- A Mac with Apple Silicon and the Photos library you want to group (local or in iCloud).
- [mise](https://mise.jdx.dev), which installs the pinned Rust toolchain (`mise install`).
  Without mise: Rust 1.99.1, and `EMEANS_DATA=data cargo run --release -p photo-emeans -- <command>`
  in place of `mise run pe -- <command>`.
- The `hf` CLI, to download the model once:

  ```sh
  hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b
  ```

  The tool reads the model from the local Hugging Face cache and never downloads it itself. The
  weights are not part of this repository and follow Google's
  [Gemma terms of use](https://ai.google.dev/gemma/terms).
- Photos access for your terminal app. On the first `fetch`, macOS asks; choose
  **Allow Full Access**. "Limited" access is refused, because the tool would see only hand-picked
  photos. To change it later: System Settings > Privacy & Security > Photos.
- Disk space for the copies: about 223 KB per photo on average (about 3.4 GB for 16k photos).

## Quickstart

Run everything from the repository root. `mise run pe -- <command>` builds the CLI in release mode
and runs it with `data/` as the data directory.

```sh
git clone https://github.com/samhvw8/photo-emeans
cd photo-emeans
mise install

# 1. Copy every photo (resumable; Ctrl-C stops cleanly, run it again to continue)
mise run pe -- fetch

# Optional: videos. Try a sample first; --all fetches every video
mise run pe -- fetch-videos --limit 60 --sample spread

# 2. Embed everything not embedded yet (resumable, safe to kill)
mise run embed

# 3. Group near-duplicates, then cluster with automatic k (writes data/clusters/auto-dedup/)
mise run pe -- cluster --auto --dedup

# 4. Look at the groups: open data/clusters/auto-dedup/gallery/index.html
mise run pe -- gallery --run auto-dedup

# 5. Album list: one album per group plus the near-duplicates album.
#    Dry run: writes data/clusters/auto-dedup/approved-from-run.json and prints the plan
mise run pe -- albums --run auto-dedup --all-clusters --duplicates

# 6. After you have read the plan: create the albums (folder "emeans")
mise run pe -- albums --approved data/clusters/auto-dedup/approved-from-run.json --yes

# 7. Content names: writes data/clusters/auto-dedup/album-names.json (never touches Photos)
mise run pe -- names --approved data/clusters/auto-dedup/approved-from-run.json

# 8. Rename: dry run first, then --yes. The folder becomes "Photo groups"
mise run pe -- albums --rename-from data/clusters/auto-dedup/album-names.json
mise run pe -- albums --rename-from data/clusters/auto-dedup/album-names.json --yes
```

**Undo the renames** (dry run, then apply):

```sh
mise run pe -- albums --rename-from data/clusters/auto-dedup/album-names.json --reverse
mise run pe -- albums --rename-from data/clusters/auto-dedup/album-names.json --reverse --yes
```

Keep `album-names.json` after a rename: it is the undo record, and `names` refuses to overwrite
it without `--force`.

The measured run below also passed `--dedup-max-gap 600` to `cluster`: two photos count as
near-duplicates only when they were taken at most 10 minutes apart, so similar scenes from
different days stay separate. Every command has `--help`; the full reference is in
[`.okf/design/cli.md`](.okf/design/cli.md).

**Picking albums by hand.** Instead of step 5, mark groups with "Album" in the gallery, set their
titles, and download `approved.json`; then run `albums --approved approved.json` (dry run) and
add `--yes`. The gallery's "Clean up" tab lets you review each near-duplicate group and download
`cleanup.json` for `albums --cleanup cleanup.json`, which collects the marked photos into one
album to delete from in Photos.

**New photos later.** `mise run pe -- add --run auto-dedup` embeds new photos and assigns them to
the existing groups without starting over.

## How long it takes

One measured run on an M1 Pro (16 GB), library of about 16,000 photos (15,894 fetched, 3,343 of
them downloaded from iCloud):

| Step | Time |
|---|---|
| Fetch | about 49 min |
| Embed (f32, Metal) | about 3 h 13 min, about 1.4 photos/s |
| Cluster (`--auto --dedup`) | about 35 s |

That is one run on one machine. Your times depend on how many photos are only in iCloud, your
network, and what else the Mac is doing. Embedding runs at a lower CPU priority; you can stop it
at any time and run it again later.

## What it writes to Photos

- **Only album and folder titles.** `albums --yes` creates one folder with the approved albums in
  a single transaction (all or nothing). `albums --rename-from ... --yes` changes titles only and
  reads every title back afterwards.
- **Never deletes or edits a photo or video.** The near-duplicates album only collects photos;
  you decide in Photos what to delete.
- **Nothing without `--yes`.** Without it, every `albums` command is a dry run that prints the
  plan.
- **Album creation syncs to iCloud**, so the albums show on all your devices.

Reading is just as limited: the tool fetches resized copies through PhotoKit and never reads
`Photos.sqlite` or the library folder.

## Limits

- **macOS only.** Photos access needs PhotoKit. The clustering and embedding libraries also run on
  Linux, but this app does not.
- **Videos are tested on a small sample.** A test set of about 500 videos was fetched, embedded and
  clustered with the photos ([results](.okf/results/video-test-run.md)). Fetching all videos
  downloads a medium rendition of each into Photos' own cache (about 0.35 MB per second of
  video); start with `--limit`.
- **Names come from a fixed label list.** A group whose photos do not agree on a label is titled
  `Mixed (<top label>)`. Edit [`labels.txt`](crates/photo-emeans/labels.txt) or pass
  `--labels FILE` to use your own words.
- **One model.** EmbeddingGemma 2 at a pinned revision; no other model is wired in.

## Docs

Project knowledge lives in the [`.okf/`](.okf/index.md) bundle:

| Topic | Where |
|---|---|
| Overview and where to read next | [`.okf/getting-started.md`](.okf/getting-started.md) |
| Every command and flag, gallery, album naming | [`.okf/design/cli.md`](.okf/design/cli.md) |
| How photos and videos are read from Photos and iCloud | [`.okf/design/photokit.md`](.okf/design/photokit.md) |
| Automatic k, incremental updates, near-duplicates | [`.okf/design/autok-incremental-dedup.md`](.okf/design/autok-incremental-dedup.md) |
| What emeans adds to the original e-means | [`.okf/results/improvements-over-faiss.md`](.okf/results/improvements-over-faiss.md) |
| Where personal data may live | [`.okf/decisions/data-location.md`](.okf/decisions/data-location.md) |
| File formats under `data/` | [`AGENTS.md`](AGENTS.md), "Data contract" |

Development: `mise run check` runs format check, clippy and the tests.

## License and credit

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

e-means comes from faiss [`contrib/e_means`](https://github.com/facebookresearch/faiss) by Meta
(MIT). The upstream Python source is kept unchanged in
[`reference/faiss_e_means/`](reference/faiss_e_means/) with its license. EmbeddingGemma 2 weights
are not included; they follow Google's Gemma terms. See [NOTICE](NOTICE).
