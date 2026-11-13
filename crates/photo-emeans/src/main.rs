//! photo-emeans: fetch, embed, cluster, gallery and albums for the user's Photos library.
//! All files live under `$EMEANS_DATA` (AGENTS.md, data contract).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use encoder::families::embeddinggemma2;

use photo_emeans::embed::{self, EmbedOpts, Gemma};
use photo_emeans::store::{EmbedStore, Kind, Meta};
use photo_emeans::util::now_iso;
use photo_emeans::{albums, cluster, gallery, names};

#[derive(Parser)]
#[command(
    name = "photo-emeans",
    about = "Cluster the Photos library by content (EmbeddingGemma 2 + e-means)"
)]
struct Cli {
    /// Data directory (photos/, embed/, clusters/). Default: $EMEANS_DATA, set by mise.toml.
    #[arg(long, env = "EMEANS_DATA", global = true)]
    data: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum DtypeArg {
    F32,
    Bf16,
}

#[derive(Clone, Copy, ValueEnum)]
enum RenditionArg {
    /// PhotoKit medium-quality rendition (default).
    Medium,
    /// PhotoKit fast (smallest) rendition.
    Fast,
}

#[derive(Clone, Copy, ValueEnum)]
enum PickArg {
    /// The oldest videos first.
    First,
    /// Evenly spaced across the library (for pilots with --limit).
    Spread,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MediaArg {
    /// Photos, then video frames.
    All,
    Photos,
    Videos,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    Metal,
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fetch 1024 px JPEG copies of every photo into photos/ (resumable).
    Fetch {
        /// Requests in flight.
        #[arg(long, default_value_t = 6)]
        concurrency: usize,
        /// Fetch only the first N photos (oldest first), for tests.
        #[arg(long)]
        limit: Option<usize>,
        /// Seconds to wait for the Photos permission prompt.
        #[arg(long, default_value_t = 300)]
        auth_timeout: u64,
    },
    /// Fetch poster + sampled frames of every video into videos/ (resumable). Downloads a
    /// medium rendition from iCloud when needed; never the original, never a video file kept.
    FetchVideos {
        /// Videos in flight.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Fetch only these N videos of the library (chosen by --pick; already fetched ones count).
        /// Required unless --all: every video is a download from iCloud into Photos' cache
        /// (~0.37 MB per second of video at medium).
        #[arg(long, required_unless_present = "all", conflicts_with = "all")]
        limit: Option<usize>,
        /// Fetch every video in the library (tens of GB of Photos cache at medium for a large library).
        #[arg(long)]
        all: bool,
        /// Which N videos --limit takes. `spread` is deterministic (evenly spaced by creation
        /// date), so the same N always gives the same set.
        #[arg(long, alias = "sample", value_enum, default_value_t = PickArg::Spread)]
        pick: PickArg,
        #[arg(long, value_enum, default_value_t = RenditionArg::Medium)]
        rendition: RenditionArg,
        /// Use the fast rendition for videos longer than this many seconds (smaller download).
        #[arg(long)]
        fast_over: Option<f64>,
        /// Only fetch videos up to this many seconds long (the rest wait for a later run).
        #[arg(long)]
        max_duration: Option<f64>,
        /// Frames per video, poster included (fewer for clips under 4 s).
        #[arg(long, default_value_t = 8)]
        frames: usize,
        /// Fetch videos listed in videos/failures.jsonl again.
        #[arg(long)]
        retry_failed: bool,
        /// Stop starting new videos below this much free disk (GB). Photos caches every
        /// downloaded rendition in its library.
        #[arg(long, default_value_t = 8.0)]
        min_free_gb: f64,
        /// Seconds to wait for the Photos permission prompt.
        #[arg(long, default_value_t = 300)]
        auth_timeout: u64,
    },
    /// Embed every fetched photo not embedded yet into embed/ (resumable, crash-safe).
    Embed(EmbedArgs),
    /// Cluster the embeddings into clusters/<run>/ (plain --k, or two-level --auto).
    Cluster {
        /// Run name (directory under clusters/). Default: k<K>, auto, with -dedup when deduped.
        #[arg(long)]
        run: Option<String>,
        /// Plain spherical e-means with this many clusters.
        #[arg(long, conflicts_with = "auto", required_unless_present = "auto")]
        k: Option<usize>,
        /// Two-level clustering with automatic k (level-1 groups, level-2 leaves).
        #[arg(long)]
        auto: bool,
        /// Assignment backend. cpu (Accelerate) is the default: faster than Metal for the many
        /// small runs of --auto, and it leaves the GPU to the embed run.
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Group near-duplicates first and cluster one representative per group.
        #[arg(long)]
        dedup: bool,
        /// --dedup: cosine at or above which two photos are near-duplicates.
        #[arg(long, default_value_t = 0.95)]
        dedup_threshold: f32,
        /// --dedup: largest time gap in seconds for a link (default: no time rule).
        #[arg(long)]
        dedup_max_gap: Option<f64>,
        /// --k: epochs.
        #[arg(long, default_value_t = 5.0)]
        nepochs: f64,
        #[arg(long, default_value_t = 1234)]
        seed: u32,
        /// --auto: level-1 k sweep, smallest k.
        #[arg(long, default_value_t = 2)]
        k1_min: usize,
        /// --auto: level-1 k sweep, largest k.
        #[arg(long, default_value_t = 40)]
        k1_max: usize,
        /// --auto: smallest cluster a person wants to browse.
        #[arg(long, default_value_t = 20)]
        min_cluster_size: usize,
        /// --auto: largest browsable cluster.
        #[arg(long, default_value_t = 200)]
        max_cluster_size: usize,
        /// --auto: e-means restarts per k.
        #[arg(long, default_value_t = 4)]
        restarts: usize,
        /// --auto: print the k sweep to stderr.
        #[arg(long)]
        verbose: bool,
        /// Replace an existing run.
        #[arg(long)]
        force: bool,
    },
    /// Write a static HTML gallery of a run to clusters/<run>/gallery/index.html.
    Gallery {
        #[arg(long)]
        run: String,
        /// Thumbnails shown per cluster before "show all".
        #[arg(long, default_value_t = 40)]
        per_cluster: usize,
    },
    /// Create Photos albums for approved clusters, or one album of the photos marked in the
    /// gallery's clean-up tab. Prints the plan; only --yes writes. Nothing is ever deleted.
    Albums {
        /// approved.json downloaded from the gallery (or written by --run).
        #[arg(long, required_unless_present_any = ["cleanup", "run", "rename_from"], conflicts_with_all = ["cleanup", "run", "rename_from"])]
        approved: Option<PathBuf>,
        /// Build the lists from this run instead of a file (with --all-clusters and/or
        /// --duplicates). Writes --out and prints the plan; never creates albums.
        #[arg(long, conflicts_with_all = ["cleanup", "yes"])]
        run: Option<String>,
        /// With --run: one album per cluster (leaf for --auto), stack members included.
        #[arg(long, requires = "run")]
        all_clusters: bool,
        /// With --run: one album of the non-keepers of every near-duplicate group.
        #[arg(long, requires = "run")]
        duplicates: bool,
        /// With --duplicates: title of that album.
        #[arg(long, default_value = albums::DUPLICATES_TITLE, requires = "duplicates")]
        duplicates_title: String,
        /// With --run: where to write the list (approved.json format).
        #[arg(long, requires = "run")]
        out: Option<PathBuf>,
        /// cleanup.json downloaded from the gallery's clean-up tab: one album to review in Photos.
        #[arg(long, conflicts_with = "rename_from")]
        cleanup: Option<PathBuf>,
        /// album-names.json written by `names`: rename the folder and its albums (titles only).
        /// Finds the folder in Photos, records the local identifiers in the file and prints the
        /// plan; only --yes renames.
        #[arg(long, conflicts_with = "run")]
        rename_from: Option<PathBuf>,
        /// With --rename-from: undo, new titles back to the old ones.
        #[arg(long, requires = "rename_from")]
        reverse: bool,
        /// With --cleanup: title of the album.
        #[arg(long, default_value = albums::CLEANUP_TITLE, requires = "cleanup")]
        title: String,
        /// Photos folder that holds the new albums.
        #[arg(long, default_value = "emeans")]
        folder: String,
        /// Only print the plan (the default without --yes).
        #[arg(long, conflicts_with = "yes")]
        dry_run: bool,
        /// Create the albums. They sync to iCloud on every device.
        #[arg(long)]
        yes: bool,
        /// With --yes: seconds to wait for Photos (it may show a permission prompt).
        #[arg(long, default_value_t = 600)]
        timeout_secs: u64,
    },
    /// Name cluster albums by content: score each album's photos against a label vocabulary
    /// (EmbeddingGemma 2 text side) and write album-names.json. Never touches Photos; rename with
    /// `albums --rename-from`.
    Names {
        /// The approved list the albums were created from.
        #[arg(long)]
        approved: PathBuf,
        /// Label vocabulary (`Name | text` per line). Default: the shipped labels.txt.
        #[arg(long)]
        labels: Option<PathBuf>,
        /// Where to write the plan. Default: album-names.json next to --approved.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Overwrite an existing --out (it may be the undo record of an applied rename).
        #[arg(long)]
        force: bool,
        /// Photos folder that holds the albums now.
        #[arg(long, default_value = "emeans")]
        folder: String,
        /// New title of that folder.
        #[arg(long, default_value = names::FOLDER_TO)]
        folder_to: String,
        /// Weakest top-label vote share that still names a cluster (below: "Mixed").
        #[arg(long, default_value_t = names::ScoreOpts::default().min_share)]
        min_share: f32,
        /// Lead in vote share the named labels need over the best label left out.
        #[arg(long, default_value_t = names::ScoreOpts::default().min_margin)]
        min_margin: f32,
        /// Vote share at which a second label joins the name.
        #[arg(long, default_value_t = names::ScoreOpts::default().second_share)]
        second_share: f32,
        /// Vote on raw cosines instead of cosines minus each label's library mean.
        #[arg(long)]
        no_center: bool,
        /// Each photo votes for its this many best labels.
        #[arg(long, default_value_t = names::ScoreOpts::default().top_k)]
        top_k: usize,
        /// Device for the text model (only used for label texts not in the cache).
        #[arg(long, value_enum, default_value_t = DeviceArg::Metal)]
        device: DeviceArg,
    },
    /// Embed new photos, then assign them with a saved --auto run (updates it in place).
    Add {
        #[arg(long)]
        run: String,
        /// Skip the embed step (assign what is already embedded).
        #[arg(long)]
        no_embed: bool,
        #[arg(long, default_value = "cpu")]
        backend: String,
    },
}

#[derive(clap::Args)]
struct EmbedArgs {
    /// Model compute dtype. f32 matches the reference exactly; bf16 halves GPU memory.
    #[arg(long, value_enum, default_value_t = DtypeArg::F32)]
    dtype: DtypeArg,
    /// Images per model call (1 is as fast as larger batches on the M1 Pro and uses least memory).
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Rows per fsynced commit.
    #[arg(long, default_value_t = 32)]
    chunk: usize,
    /// Embed at most N new photos in this run.
    #[arg(long)]
    limit: Option<usize>,
    /// Keep polling the manifest until the fetch log ends with its summary line or no new
    /// lines arrive for --idle-secs.
    #[arg(long)]
    follow: bool,
    /// With --follow: the photo fetch log to watch. Default: $EMEANS_DATA/fetch.log.
    #[arg(long)]
    fetch_log: Option<PathBuf>,
    /// With --follow: the video fetch log to watch. Default: $EMEANS_DATA/fetch-videos.log.
    #[arg(long)]
    fetch_videos_log: Option<PathBuf>,
    /// What to embed. Videos: each frame into embed/frames/, then one mean row per video.
    #[arg(long, value_enum, default_value_t = MediaArg::All)]
    media: MediaArg,
    /// Hold an exclusive flock on this file while embedding (shared with other GPU jobs).
    #[arg(long)]
    gpu_lock: Option<PathBuf>,
    /// With --gpu-lock: release the lock after this many rows, so other jobs can run.
    #[arg(long, default_value_t = 500)]
    lock_rows: usize,
    /// With --follow: stop after this many seconds without new manifest lines.
    #[arg(long, default_value_t = 300)]
    idle_secs: u64,
    /// Retry photos listed in embed/failures.jsonl.
    #[arg(long)]
    retry_failed: bool,
    #[arg(long, value_enum, default_value_t = DeviceArg::Metal)]
    device: DeviceArg,
    /// Model directory or Hugging Face id.
    #[arg(long, default_value = embeddinggemma2::MODEL_ID)]
    model: String,
    /// CPU niceness for this process (0 = unchanged).
    #[arg(long, default_value_t = 10)]
    nice: i32,
}

impl Default for EmbedArgs {
    fn default() -> Self {
        Self {
            dtype: DtypeArg::F32,
            batch: 1,
            chunk: 32,
            limit: None,
            follow: false,
            fetch_log: None,
            fetch_videos_log: None,
            media: MediaArg::All,
            gpu_lock: None,
            lock_rows: 500,
            idle_secs: 300,
            retry_failed: false,
            device: DeviceArg::Metal,
            model: embeddinggemma2::MODEL_ID.into(),
            nice: 10,
        }
    }
}

fn data_dir(cli: &Cli) -> Result<PathBuf> {
    cli.data
        .clone()
        .context("no data directory: set EMEANS_DATA (mise.toml does) or pass --data")
}

fn log(msg: &str) {
    println!("[{}] {msg}", &now_iso()[11..19]);
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let data = data_dir(&cli)?;
    match cli.cmd {
        Cmd::Cluster {
            run,
            k,
            auto,
            backend,
            dedup,
            dedup_threshold,
            dedup_max_gap,
            nepochs,
            seed,
            k1_min,
            k1_max,
            min_cluster_size,
            max_cluster_size,
            restarts,
            verbose,
            force,
        } => {
            let mode = if auto {
                let cfg = emeans::AutoConfig {
                    k1_range: (k1_min, k1_max),
                    min_cluster_size,
                    max_cluster_size,
                    restarts,
                    seed,
                    verbose,
                    ..Default::default()
                };
                cluster::Mode::Auto(Box::new(cfg))
            } else {
                cluster::Mode::K(k.context("pass --k K or --auto")?)
            };
            let dedup = dedup.then(|| emeans::DedupConfig {
                sim_threshold: dedup_threshold,
                max_gap_secs: dedup_max_gap,
                ..Default::default()
            });
            let opts = cluster::ClusterOpts {
                run: run.unwrap_or_else(|| cluster::default_run_name(&mode, dedup.is_some())),
                mode,
                backend: cluster::parse_backend(&backend)?,
                dedup,
                nepochs,
                seed,
                force,
            };
            cluster::cluster(&data, &opts, &mut |m| log(m))?;
            Ok(())
        }
        Cmd::Gallery { run, per_cluster } => {
            let p = gallery::build(
                &data,
                &run,
                &gallery::GalleryOpts { per_cluster },
                &mut |m| log(m),
            )?;
            log(&format!("open {}", p.display()));
            Ok(())
        }
        Cmd::Albums {
            approved,
            run,
            all_clusters,
            duplicates,
            duplicates_title,
            out,
            cleanup,
            rename_from,
            reverse,
            title,
            folder,
            dry_run: _,
            yes,
            timeout_secs,
        } => match (approved, cleanup) {
            (None, None) if rename_from.is_some() => albums::run_rename(
                &rename_from.unwrap(),
                reverse,
                yes,
                Duration::from_secs(timeout_secs),
                &mut |m| println!("{m}"),
            ),
            (None, None) if run.is_some() => {
                let run = run.unwrap();
                let out = out.unwrap_or_else(|| {
                    data.join("clusters")
                        .join(&run)
                        .join("approved-from-run.json")
                });
                albums::run_from_run(
                    &data,
                    &run,
                    &albums::RunLists {
                        all_clusters,
                        duplicates,
                        duplicates_title,
                    },
                    &folder,
                    &out,
                    &mut |m| println!("{m}"),
                )
            }
            (Some(approved), None) => albums::run(
                &data,
                &approved,
                &folder,
                yes,
                Duration::from_secs(timeout_secs),
                &mut |m| println!("{m}"),
            ),
            (None, Some(cleanup)) => albums::run_cleanup(
                &data,
                &cleanup,
                &folder,
                &title,
                yes,
                Duration::from_secs(timeout_secs),
                &mut |m| println!("{m}"),
            ),
            _ => bail!("give exactly one of --approved, --cleanup, --run and --rename-from"),
        },
        Cmd::Names {
            approved,
            labels,
            out,
            force,
            folder,
            folder_to,
            min_share,
            min_margin,
            second_share,
            no_center,
            top_k,
            device,
        } => {
            let out = out.unwrap_or_else(|| {
                approved
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("album-names.json")
            });
            let model = format!(
                "{}@{}/f32",
                embeddinggemma2::MODEL_ID,
                embeddinggemma2::MODEL_REVISION
            );
            let mut load = || -> Result<Box<dyn names::TextEmbedder>> {
                let dev = match device {
                    DeviceArg::Metal => candle_core::Device::new_metal(0)?,
                    DeviceArg::Cpu => candle_core::Device::Cpu,
                };
                let t = Instant::now();
                let m = embeddinggemma2::EmbeddingGemma2::load_with(
                    embeddinggemma2::MODEL_ID,
                    &dev,
                    candle_core::DType::F32,
                )?
                .with_batch_size(8);
                log(&format!(
                    "model loaded in {:.1}s",
                    t.elapsed().as_secs_f64()
                ));
                Ok(Box::new(names::GemmaText(m)))
            };
            names::run(
                &data,
                &names::RunOpts {
                    approved: &approved,
                    labels: labels.as_deref(),
                    out: &out,
                    folder: &folder,
                    folder_to: &folder_to,
                    score: names::ScoreOpts {
                        center: !no_center,
                        top_k,
                        min_share,
                        min_margin,
                        second_share,
                    },
                    model: &model,
                    force,
                },
                &mut load,
                &mut |m| println!("{m}"),
            )
        }
        Cmd::Add {
            run,
            no_embed,
            backend,
        } => {
            if !no_embed {
                embed_cmd(&data, EmbedArgs::default())?;
            }
            let s = cluster::add(&data, &run, cluster::parse_backend(&backend)?, &mut |m| {
                log(m)
            })?;
            log(&format!(
                "{run}: {} -> {} photos ({}); rerun `gallery --run {run}` to see them",
                s.n_before, s.n_after, s.action
            ));
            Ok(())
        }
        Cmd::Fetch {
            concurrency,
            limit,
            auth_timeout,
        } => fetch(&data, concurrency, limit, auth_timeout),
        Cmd::FetchVideos {
            concurrency,
            limit,
            all: _,
            pick,
            rendition,
            fast_over,
            max_duration,
            frames,
            retry_failed,
            min_free_gb,
            auth_timeout,
        } => fetch_videos(
            &data,
            &FetchVideosArgs {
                concurrency,
                limit,
                pick,
                rendition,
                fast_over,
                max_duration,
                frames,
                retry_failed,
                min_free_gb,
                auth_timeout,
            },
        ),
        Cmd::Embed(args) => embed_cmd(&data, args),
    }
}

struct FetchVideosArgs {
    concurrency: usize,
    limit: Option<usize>,
    pick: PickArg,
    rendition: RenditionArg,
    fast_over: Option<f64>,
    max_duration: Option<f64>,
    frames: usize,
    retry_failed: bool,
    min_free_gb: f64,
    auth_timeout: u64,
}

/// Must run on the main thread, like `fetch`.
fn fetch_videos(data: &Path, a: &FetchVideosArgs) -> Result<()> {
    photos::stop::install_signal_stop();
    photos::ensure_authorized(Duration::from_secs(a.auth_timeout))?;
    let mut assets = photos::list_videos()?;
    log(&format!("listed {} videos", assets.len()));
    if let Some(m) = a.max_duration {
        assets.retain(|v| v.duration <= m);
        log(&format!("{} videos are at most {m} s long", assets.len()));
    }
    if let Some(n) = a.limit {
        let n = n.min(assets.len());
        assets = match a.pick {
            PickArg::First => assets[..n].to_vec(),
            PickArg::Spread => (0..n)
                .map(|i| assets[i * assets.len() / n.max(1)].clone())
                .collect(),
        };
    }
    let cfg = photos::VideoFetchAllConfig {
        concurrency: a.concurrency,
        retry_failed: a.retry_failed,
        min_free_bytes: (a.min_free_gb * 1e9) as u64,
        video: photos::VideoConfig {
            frames: a.frames,
            fast_over: a.fast_over,
            rendition: match a.rendition {
                RenditionArg::Medium => photos::Rendition::Medium,
                RenditionArg::Fast => photos::Rendition::Fast,
            },
            ..Default::default()
        },
    };
    let total = assets.len();
    let mut i = 0usize;
    let report = photos::fetch_all_videos(&data.join("videos"), &assets, &cfg, |ev| match ev {
        photos::VideoFetchEvent::Done {
            line,
            bytes,
            from_icloud,
            rendition_bytes,
            request,
            decode,
            elapsed,
        } => {
            i += 1;
            println!(
                "{i:>6}/{total} {}  {:>7.1}s  {} frames  {bytes} B  rendition {} MB  request {} ms  decode {} ms  total {} ms icloud={from_icloud}",
                &line.id[..8.min(line.id.len())],
                line.duration,
                line.frames.len(),
                rendition_bytes.map_or("?".into(), |b| format!("{:.1}", b as f64 / 1e6)),
                request.as_millis(),
                decode.as_millis(),
                elapsed.as_millis()
            );
        }
        photos::VideoFetchEvent::Failed { id, error } => {
            println!("  FAILED {}: {error}", &id[..8.min(id.len())]);
        }
    })?;
    // `embed --follow` treats a last line starting with "fetched " as the end of the fetch.
    println!(
        "fetched {} videos ({} frames, from iCloud {}, renditions {:.2} GB, frames {:.1} MB), skipped {} (already done), {} (failed before), failed {}, repaired {}; {:.1}s total",
        report.fetched,
        report.frames,
        report.from_icloud,
        report.rendition_bytes as f64 / 1e9,
        report.bytes as f64 / 1e6,
        report.skipped,
        report.skipped_failed,
        report.failed.len(),
        report.repaired,
        report.elapsed.as_secs_f64()
    );
    if let Some(sig) = report.interrupted {
        bail!(
            "stopped by signal {sig}: started no new video after it; the videos in flight were \
             finished and recorded. Run fetch-videos again to continue"
        );
    }
    if let Some(free) = report.stopped_low_disk {
        bail!(
            "stopped early: free disk space fell to {:.1} GB (limit {:.1} GB). Free space (Photos \
             keeps downloaded renditions as a purgeable cache), then run fetch-videos again",
            free as f64 / 1e9,
            a.min_free_gb
        );
    }
    if !report.failed.is_empty() {
        bail!(
            "{} videos failed (listed in videos/failures.jsonl); run fetch-videos --retry-failed to retry them",
            report.failed.len()
        );
    }
    Ok(())
}

fn embed_cmd(data: &Path, a: EmbedArgs) -> Result<()> {
    let EmbedArgs {
        dtype,
        batch,
        chunk,
        limit,
        follow,
        fetch_log,
        fetch_videos_log,
        media,
        gpu_lock,
        lock_rows,
        idle_secs,
        retry_failed,
        device,
        model,
        nice,
    } = a;
    if nice != 0 {
        photo_emeans::renice(nice)?;
    }
    let (dt, dt_name) = match dtype {
        DtypeArg::F32 => (candle_core::DType::F32, "f32"),
        DtypeArg::Bf16 => (candle_core::DType::BF16, "bf16"),
    };
    let revision = if model == embeddinggemma2::MODEL_ID {
        embeddinggemma2::MODEL_REVISION
    } else {
        "local"
    };
    let want = Meta::new(&model, revision, dt_name);
    let (mut store, _repair) =
        EmbedStore::open(&data.join("embed"), want.clone(), &mut |m| log(m))?;
    log(&format!(
        "store has {} rows ({} photos, {} videos)",
        store.n(),
        store.n() - store.count(Kind::Video),
        store.count(Kind::Video)
    ));
    let do_videos =
        media != MediaArg::Photos && data.join("videos").join("manifest.jsonl").exists();
    let mut frames = if do_videos {
        let (f, _) = EmbedStore::open(&data.join("embed").join("frames"), want, &mut |m| log(m))?;
        log(&format!("frame store has {} rows", f.n()));
        Some(f)
    } else {
        None
    };
    let (dev, dev_name) = match device {
        DeviceArg::Metal => (candle_core::Device::new_metal(0)?, "metal"),
        DeviceArg::Cpu => (candle_core::Device::Cpu, "cpu"),
    };
    let t = Instant::now();
    let emb = embeddinggemma2::EmbeddingGemma2::load_with(&model, &dev, dt)?.with_batch_size(batch);
    log(&format!(
        "model loaded in {:.1}s ({dt_name}, batch {batch}, device {dev_name})",
        t.elapsed().as_secs_f64()
    ));
    let mut lock = match &gpu_lock {
        Some(p) => photo_emeans::embed::GpuLock::new(p, lock_rows)?,
        None => photo_emeans::embed::GpuLock::none(),
    };
    let opts = EmbedOpts {
        batch,
        chunk,
        limit,
        retry_failed,
        follow,
        fetch_log: Some(fetch_log.unwrap_or_else(|| data.join("fetch.log"))),
        idle_timeout: Duration::from_secs(idle_secs),
        ..EmbedOpts::default()
    };
    let mut gemma = Gemma(emb);
    let run_log = RunLog::start(data, dev_name, dt_name, media)?;
    let t = Instant::now();
    let mut photos_done = 0;
    if media != MediaArg::Videos {
        let s = embed::run_locked(
            &data.join("photos"),
            &mut store,
            &mut gemma,
            &opts,
            &mut lock,
            &mut |m| log(m),
        )?;
        photos_done = s.embedded;
        log(&format!(
            "photos done: embedded {}, failed {}, store has {} rows, {} photos in the manifest still pending; {:.1}s",
            s.embedded,
            s.failed,
            s.total,
            s.remaining,
            t.elapsed().as_secs_f64()
        ));
    }
    let mut frames_done = 0;
    let mut videos_done = 0;
    if let Some(frames) = frames.as_mut() {
        let vopts = EmbedOpts {
            fetch_log: Some(fetch_videos_log.unwrap_or_else(|| data.join("fetch-videos.log"))),
            ..opts.clone()
        };
        let t = Instant::now();
        let s = embed::run_videos(
            &data.join("videos"),
            &mut store,
            frames,
            &mut gemma,
            &vopts,
            &mut lock,
            &mut |m| log(m),
        )?;
        frames_done = s.frames;
        videos_done = s.videos;
        log(&format!(
            "videos done: {} frames embedded, {} videos added, {} failed; frame store {} rows, {} videos in the main store, {} videos still pending; {:.1}s",
            s.frames,
            s.videos,
            s.failed,
            s.frames_total,
            s.videos_total,
            s.remaining,
            t.elapsed().as_secs_f64()
        ));
    } else if media == MediaArg::Videos {
        log("no videos/manifest.jsonl: run fetch-videos first");
    }
    lock.release()?;
    run_log.finish(photos_done, frames_done, videos_done, lock.waited)?;
    Ok(())
}

/// This machine's host name (`hostname`), or `unknown`.
fn host_name() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Lines in `embed/runs.jsonl`: a `start` line when a run begins (so a killed run is on record
/// too) and an `end` line with the counts when it finishes.
struct RunLog {
    path: PathBuf,
    line: serde_json::Value,
    started: Instant,
}

impl RunLog {
    fn start(data: &Path, device: &str, dtype: &str, media: MediaArg) -> Result<Self> {
        let r = Self {
            path: data.join("embed").join("runs.jsonl"),
            line: serde_json::json!({
                "event": "start",
                "started": now_iso(),
                "host": host_name(),
                "os": std::env::consts::OS,
                "device": device,
                "compute_dtype": dtype,
                "media": match media { MediaArg::All => "all", MediaArg::Photos => "photos", MediaArg::Videos => "videos" },
            }),
            started: Instant::now(),
        };
        r.append(&r.line)?;
        Ok(r)
    }

    fn append(&self, v: &serde_json::Value) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(format!("{v}\n").as_bytes())?;
        f.sync_data()?;
        Ok(())
    }

    fn finish(
        mut self,
        photos: usize,
        frames: usize,
        videos: usize,
        waited: Duration,
    ) -> Result<()> {
        let o = self.line.as_object_mut().context("run log is an object")?;
        o.insert("event".into(), "end".into());
        o.insert("ended".into(), now_iso().into());
        o.insert(
            "seconds".into(),
            (self.started.elapsed().as_secs_f64().round()).into(),
        );
        o.insert(
            "lock_wait_seconds".into(),
            (waited.as_secs_f64().round()).into(),
        );
        o.insert("photos".into(), photos.into());
        o.insert("frames".into(), frames.into());
        o.insert("videos".into(), videos.into());
        self.append(&self.line)
    }
}

/// Must run on the main thread: PhotoKit delivers image results on the main queue
/// (.okf/design/photokit.md). No async runtime, no spawning this onto another thread.
fn fetch(data: &Path, concurrency: usize, limit: Option<usize>, auth_timeout: u64) -> Result<()> {
    photos::stop::install_signal_stop();
    photos::ensure_authorized(Duration::from_secs(auth_timeout))?;
    let mut assets = photos::list_photos()?;
    log(&format!("listed {} photos", assets.len()));
    if let Some(n) = limit {
        assets.truncate(n);
    }
    let cfg = photos::FetchAllConfig {
        concurrency,
        ..Default::default()
    };
    let total = assets.len();
    let mut i = 0usize;
    let report = photos::fetch_all(&data.join("photos"), &assets, &cfg, |ev| match ev {
        photos::FetchEvent::Done {
            line,
            bytes,
            from_icloud,
            elapsed,
            ..
        } => {
            i += 1;
            println!(
                "{i:>7}/{total} {}  {}x{}  {bytes} B  {} ms icloud={from_icloud}",
                &line.id[..8.min(line.id.len())],
                line.width,
                line.height,
                elapsed.as_millis()
            );
        }
        photos::FetchEvent::Failed { id, error } => {
            println!("  FAILED {}: {error}", &id[..8.min(id.len())]);
        }
    })?;
    // `embed --follow` treats a last line starting with "fetched " as the end of the fetch.
    println!(
        "fetched {} (from iCloud {}), skipped {} (already done), failed {}, repaired {}; {:.1}s total",
        report.fetched,
        report.from_icloud,
        report.skipped,
        report.failed.len(),
        report.repaired,
        report.elapsed.as_secs_f64()
    );
    if let Some(sig) = report.interrupted {
        bail!("stopped by signal {sig}; run fetch again to continue");
    }
    if !report.failed.is_empty() {
        bail!(
            "{} photos failed; run fetch again to retry them",
            report.failed.len()
        );
    }
    Ok(())
}
