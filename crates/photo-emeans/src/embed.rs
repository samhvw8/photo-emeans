//! The `embed` step: every photo and video frame in the manifests that is not embedded yet goes
//! through the image embedder and into an [`EmbedStore`] in fsynced chunks.
//!
//! - Photos ([`run`]) go to the main store `embed/` as `kind: photo` rows.
//! - Videos ([`run_videos`]): each frame goes to the frame store `embed/frames/` (id
//!   `<asset id>#fNN`, `kind: frame`). When every frame of a video is in the frame store, the
//!   L2-normalized mean of its frame vectors goes to the main store as one `kind: video` row. A
//!   run killed between the two writes finishes the video row on the next run from the stored
//!   frames.
//!
//! One thread decodes and preprocesses the next images while the GPU runs the model. An image that
//! cannot be opened or preprocessed is logged to `embed/failures.jsonl` and skipped (for a frame,
//! the whole video is skipped); later runs skip it too, unless `retry_failed`. Any error from the
//! model itself is fatal: the rows already computed are committed first, then the run stops.
//!
//! With a [`GpuLock`], the run holds an exclusive `flock(2)` on a shared lock file only while it
//! embeds, and releases it every `lock_rows` rows (at a commit) so other GPU jobs can run between
//! its chunks.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use encoder::families::embeddinggemma2::{self, EmbeddingGemma2};
use encoder::{Encoder, Patches};

use crate::manifest;
use crate::store::{self, D, EmbedStore, Failure, Kind};
use crate::util::{fmt_duration, now_iso};

/// What runs the model. The real one is [`Gemma`]; tests use a deterministic fake.
pub trait ImageEmbedder {
    /// One vector per input, unit length, in order.
    fn embed(&mut self, batch: &[Patches]) -> Result<Vec<[f32; D]>>;
}

/// EmbeddingGemma 2, images without a prompt (the model card's rule).
pub struct Gemma(pub EmbeddingGemma2);

impl ImageEmbedder for Gemma {
    fn embed(&mut self, batch: &[Patches]) -> Result<Vec<[f32; D]>> {
        self.0
            .embed_preprocessed(batch, None)?
            .into_iter()
            .map(|v| {
                v.try_into()
                    .map_err(|v: Vec<f32>| anyhow::anyhow!("embedding has {} dims", v.len()))
            })
            .collect()
    }
}

/// Decode, orient (EXIF) and preprocess one photo file.
pub fn prepare(path: &Path) -> Result<Patches> {
    embeddinggemma2::preprocess(&encoder::open_image(path)?)
}

#[derive(Debug, Clone)]
pub struct EmbedOpts {
    /// Images per model call.
    pub batch: usize,
    /// Rows per fsynced commit.
    pub chunk: usize,
    /// Print a progress line every this many embedded images.
    pub progress_every: usize,
    /// Embed at most this many new images (photos, or video frames) in this run.
    pub limit: Option<usize>,
    /// Retry items listed in `failures.jsonl`.
    pub retry_failed: bool,
    /// Keep polling the manifest for new lines.
    pub follow: bool,
    /// With `follow`: stop when this log's last line is the fetch summary.
    pub fetch_log: Option<PathBuf>,
    /// With `follow`: stop after this long without new manifest lines.
    pub idle_timeout: Duration,
    /// With `follow`: wait this long between manifest reads.
    pub poll: Duration,
}

impl Default for EmbedOpts {
    fn default() -> Self {
        Self {
            batch: embeddinggemma2::DEFAULT_BATCH_SIZE,
            chunk: 32,
            progress_every: 100,
            limit: None,
            retry_failed: false,
            follow: false,
            fetch_log: None,
            idle_timeout: Duration::from_secs(300),
            poll: Duration::from_secs(10),
        }
    }
}

/// An exclusive `flock(2)` on a lock file shared with other GPU jobs, held in chunks of rows.
pub struct GpuLock {
    file: Option<File>,
    path: PathBuf,
    held: bool,
    rows_since: usize,
    /// Release the lock after this many rows (at the next commit).
    pub lock_rows: usize,
    /// Pause after a release, so a waiting job gets the lock first.
    pub pause: Duration,
    /// Seconds spent waiting for the lock.
    pub waited: Duration,
    log: Vec<String>,
}

impl GpuLock {
    /// No lock: every call is a no-op.
    pub fn none() -> Self {
        Self {
            file: None,
            path: PathBuf::new(),
            held: false,
            rows_since: 0,
            lock_rows: usize::MAX,
            pause: Duration::ZERO,
            waited: Duration::ZERO,
            log: Vec::new(),
        }
    }

    /// Lock `path` (created if missing) in chunks of `lock_rows` rows.
    pub fn new(path: &Path, lock_rows: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .with_context(|| format!("opening GPU lock {}", path.display()))?;
        Ok(Self {
            file: Some(file),
            path: path.to_path_buf(),
            held: false,
            rows_since: 0,
            lock_rows: lock_rows.max(1),
            pause: Duration::from_secs(2),
            waited: Duration::ZERO,
            log: Vec::new(),
        })
    }

    pub fn is_held(&self) -> bool {
        self.held
    }

    /// Block until the lock is held.
    pub fn acquire(&mut self) -> Result<()> {
        if self.held {
            return Ok(());
        }
        let Some(f) = &self.file else { return Ok(()) };
        let t = Instant::now();
        f.lock()
            .with_context(|| format!("locking {}", self.path.display()))?;
        let w = t.elapsed();
        self.waited += w;
        if w > Duration::from_secs(5) {
            self.log.push(format!(
                "waited {} for the GPU lock",
                fmt_duration(w.as_secs_f64())
            ));
        }
        self.held = true;
        self.rows_since = 0;
        Ok(())
    }

    /// Count committed rows; release (and pause) when the chunk is full.
    fn committed(&mut self, rows: usize) -> Result<()> {
        self.rows_since += rows;
        if self.held && self.rows_since >= self.lock_rows {
            self.release()?;
            std::thread::sleep(self.pause);
        }
        Ok(())
    }

    pub fn release(&mut self) -> Result<()> {
        if let (true, Some(f)) = (self.held, &self.file) {
            f.unlock()
                .with_context(|| format!("unlocking {}", self.path.display()))?;
        }
        self.held = false;
        Ok(())
    }

    fn drain_log(&mut self) -> Vec<String> {
        std::mem::take(&mut self.log)
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct EmbedSummary {
    pub embedded: usize,
    pub failed: usize,
    /// Rows in the store at the end.
    pub total: usize,
    /// Manifest photos not embedded and not failed at the end.
    pub remaining: usize,
}

/// One image to embed.
#[derive(Debug, Clone)]
struct Job {
    /// Row id in the store it goes to.
    id: String,
    /// Asset id that a failure is recorded under (the photo, or the frame's video).
    asset: String,
    /// File name recorded in a failure.
    file: String,
    path: PathBuf,
}

fn failed_ids(store: &EmbedStore, retry: bool) -> Result<HashSet<String>> {
    Ok(if retry {
        HashSet::new()
    } else {
        store::read_failures(store.dir())?
            .into_iter()
            .map(|f| f.id)
            .collect()
    })
}

/// Photos in the manifest that still need a vector.
fn pending(
    photos_dir: &Path,
    store: &EmbedStore,
    failed: &HashSet<String>,
) -> Result<(Vec<Job>, usize)> {
    let lines = manifest::read(photos_dir)?;
    let total = lines.len();
    let todo = lines
        .into_iter()
        .filter(|l| !store.contains(&l.id) && !failed.contains(&l.id))
        .map(|l| Job {
            path: photos_dir.join(&l.file),
            id: l.id.clone(),
            asset: l.id,
            file: l.file,
        })
        .collect();
    Ok((todo, total))
}

/// Decide whether a `follow` loop should stop; logs why.
fn stop_following(
    opts: &EmbedOpts,
    budget: usize,
    last_new: Instant,
    log: &mut dyn FnMut(&str),
) -> bool {
    let done_fetch = opts
        .fetch_log
        .as_deref()
        .is_some_and(manifest::fetch_finished);
    let stop = !opts.follow || budget == 0 || done_fetch || last_new.elapsed() >= opts.idle_timeout;
    if stop && opts.follow {
        log(&format!(
            "follow: stopping ({})",
            if budget == 0 {
                "limit reached"
            } else if done_fetch {
                "fetch finished"
            } else {
                "no new manifest lines"
            }
        ));
    }
    stop
}

/// Embed pending photos into `store`. Prints progress through `log`.
pub fn run(
    photos_dir: &Path,
    store: &mut EmbedStore,
    emb: &mut dyn ImageEmbedder,
    opts: &EmbedOpts,
    log: &mut dyn FnMut(&str),
) -> Result<EmbedSummary> {
    run_locked(photos_dir, store, emb, opts, &mut GpuLock::none(), log)
}

/// [`run`] holding `lock` while embedding.
pub fn run_locked(
    photos_dir: &Path,
    store: &mut EmbedStore,
    emb: &mut dyn ImageEmbedder,
    opts: &EmbedOpts,
    lock: &mut GpuLock,
    log: &mut dyn FnMut(&str),
) -> Result<EmbedSummary> {
    let mut failed = failed_ids(store, opts.retry_failed)?;
    let mut sum = EmbedSummary::default();
    let started = Instant::now();
    let mut last_new = Instant::now();
    let mut budget = opts.limit.unwrap_or(usize::MAX);
    loop {
        let (mut todo, in_manifest) = pending(photos_dir, store, &failed)?;
        todo.truncate(budget);
        if todo.is_empty() {
            if stop_following(opts, budget, last_new, log) {
                lock.release()?;
                sum.total = store.n();
                let (rest, _) = pending(photos_dir, store, &failed)?;
                sum.remaining = rest.len();
                return Ok(sum);
            }
            lock.release()?;
            std::thread::sleep(opts.poll);
            continue;
        }
        last_new = Instant::now();
        log(&format!(
            "embedding {} photos ({} in manifest, {} already embedded, {} failed before)",
            todo.len(),
            in_manifest,
            store.n(),
            failed.len()
        ));
        let n = todo.len();
        let failures_dir = store.dir().to_path_buf();
        let base = store.n();
        let mut commit = |rows: &[(String, [f32; D])]| store.append(rows, Kind::Photo);
        let embedded_before = sum.embedded;
        let pass = embed_list(
            &todo,
            &failures_dir,
            emb,
            opts,
            lock,
            &mut commit,
            &mut |ev| match ev {
                Event::Failed(f) => {
                    log(&format!("skip {}: {}", short(&f.id), f.error));
                    failed.insert(f.id.clone());
                }
                Event::Progress { done: k } => {
                    let done = embedded_before + k;
                    let secs = started.elapsed().as_secs_f64();
                    let rate = done as f64 / secs.max(1e-9);
                    let left = n.saturating_sub(k) as f64 / rate.max(1e-9);
                    log(&format!(
                        "{k}/{n} this pass | {} in store | {rate:.2} photos/s | ETA {}",
                        base + k,
                        fmt_duration(left)
                    ));
                }
                Event::Note(m) => log(m),
            },
        );
        let (ok, bad) = pass?;
        sum.embedded += ok;
        sum.failed += bad;
        budget = budget.saturating_sub(ok + bad);
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct VideoSummary {
    /// Frames embedded in this run.
    pub frames: usize,
    /// Video rows added to the main store in this run.
    pub videos: usize,
    /// Videos skipped in this run because a frame could not be read.
    pub failed: usize,
    /// Rows in the frame store at the end.
    pub frames_total: usize,
    /// Video rows in the main store at the end.
    pub videos_total: usize,
    /// Manifest videos without a row and not failed at the end.
    pub remaining: usize,
}

/// Id of frame `i` of video `asset` in the frame store.
pub fn frame_id(asset: &str, i: usize) -> String {
    format!("{asset}#f{i:02}")
}

/// L2-normalized mean of `vs`. Fails on an empty list or a zero mean.
pub fn mean_unit(vs: &[&[f32; D]]) -> Result<[f32; D]> {
    ensure!(!vs.is_empty(), "no frame vectors");
    let mut m = [0f64; D];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v.iter()) {
            *a += f64::from(*b);
        }
    }
    let n = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    ensure!(n > 1e-6, "frame vectors cancel out (mean norm {n})");
    let mut out = [0f32; D];
    for (o, x) in out.iter_mut().zip(m) {
        *o = (x / n) as f32;
    }
    Ok(out)
}

/// Embed the frames of every fetched video not in `main` yet into `frames`, then add one row per
/// complete video to `main`. `videos_dir` is `$EMEANS_DATA/videos`.
pub fn run_videos(
    videos_dir: &Path,
    main: &mut EmbedStore,
    frames: &mut EmbedStore,
    emb: &mut dyn ImageEmbedder,
    opts: &EmbedOpts,
    lock: &mut GpuLock,
    log: &mut dyn FnMut(&str),
) -> Result<VideoSummary> {
    ensure!(
        main.meta().model == frames.meta().model
            && main.meta().revision == frames.meta().revision
            && main.meta().compute_dtype == frames.meta().compute_dtype
            && main.meta().prompt == frames.meta().prompt,
        "embed/frames/meta.json and embed/meta.json come from different models or dtypes"
    );
    // Every stored frame vector, by frame id (12k frames x 3 KB).
    let mut vecs: HashMap<String, [f32; D]> = HashMap::new();
    if frames.n() > 0 {
        let e = store::load(frames.dir())?;
        for (i, id) in e.ids.iter().enumerate() {
            vecs.insert(id.clone(), e.row(i).try_into().expect("row of D floats"));
        }
    }
    let failed = RefCell::new(failed_ids(main, opts.retry_failed)?);
    let mut sum = VideoSummary::default();
    let started = Instant::now();
    let mut last_new = Instant::now();
    let mut budget = opts.limit.unwrap_or(usize::MAX);
    loop {
        let lines = manifest::read_videos(videos_dir)?;
        let pend: Vec<&manifest::VideoLine> = lines
            .iter()
            .filter(|v| !main.contains(&v.id) && !failed.borrow().contains(&v.id))
            .collect();
        // Videos whose frames are all stored already (a run killed after the frame commit).
        let ready = finish_videos(&pend, &vecs, main)?;
        if ready > 0 {
            log(&format!(
                "added {ready} videos whose frames were already embedded"
            ));
            sum.videos += ready;
        }
        let mut todo = Vec::new();
        for v in &pend {
            if main.contains(&v.id) {
                continue;
            }
            for (i, f) in v.frames.iter().enumerate() {
                let id = frame_id(&v.id, i);
                if !vecs.contains_key(&id) {
                    todo.push(Job {
                        id,
                        asset: v.id.clone(),
                        file: format!("videos/{}", f.file),
                        path: videos_dir.join(&f.file),
                    });
                }
            }
        }
        todo.truncate(budget);
        if todo.is_empty() {
            if stop_following(opts, budget, last_new, log) {
                lock.release()?;
                sum.frames_total = frames.n();
                sum.videos_total = main.count(Kind::Video);
                sum.remaining = manifest::read_videos(videos_dir)?
                    .iter()
                    .filter(|v| !main.contains(&v.id) && !failed.borrow().contains(&v.id))
                    .count();
                return Ok(sum);
            }
            lock.release()?;
            std::thread::sleep(opts.poll);
            continue;
        }
        last_new = Instant::now();
        let n = todo.len();
        log(&format!(
            "embedding {n} frames of {} videos ({} videos in manifest, {} done, {} failed before)",
            pend.len(),
            lines.len(),
            main.count(Kind::Video),
            failed.borrow().len()
        ));
        let by_id: HashMap<&str, &manifest::VideoLine> =
            pend.iter().map(|v| (v.id.as_str(), *v)).collect();
        let failures_dir = main.dir().to_path_buf();
        let added = RefCell::new(0usize);
        let mut commit = |rows: &[(String, [f32; D])]| -> Result<()> {
            frames.append(rows, Kind::Frame)?;
            let mut touched: Vec<&manifest::VideoLine> = Vec::new();
            for (id, v) in rows {
                vecs.insert(id.clone(), *v);
                let asset = id.rsplit_once('#').map_or(id.as_str(), |(a, _)| a);
                if let Some(l) = by_id.get(asset)
                    && !touched.iter().any(|t| t.id == l.id)
                {
                    touched.push(l);
                }
            }
            let f = failed.borrow();
            let ok: Vec<&manifest::VideoLine> =
                touched.into_iter().filter(|l| !f.contains(&l.id)).collect();
            drop(f);
            *added.borrow_mut() += finish_videos(&ok, &vecs, main)?;
            Ok(())
        };
        let frames_before = sum.frames;
        let pass = embed_list(
            &todo,
            &failures_dir,
            emb,
            opts,
            lock,
            &mut commit,
            &mut |ev| match ev {
                Event::Failed(f) => {
                    log(&format!("skip video {}: {}", short(&f.id), f.error));
                    failed.borrow_mut().insert(f.id.clone());
                }
                Event::Progress { done: k } => {
                    let done = frames_before + k;
                    let secs = started.elapsed().as_secs_f64();
                    let rate = done as f64 / secs.max(1e-9);
                    let left = n.saturating_sub(k) as f64 / rate.max(1e-9);
                    log(&format!(
                        "{k}/{n} frames this pass | {} videos added | {rate:.2} frames/s | ETA {}",
                        sum.videos + *added.borrow(),
                        fmt_duration(left)
                    ));
                }
                Event::Note(m) => log(m),
            },
        );
        let (ok, bad) = pass?;
        sum.frames += ok;
        sum.videos += added.into_inner();
        sum.failed += bad;
        budget = budget.saturating_sub(ok + bad);
    }
}

/// Add a main-store row for every video in `videos` whose frames are all in `vecs`.
fn finish_videos(
    videos: &[&manifest::VideoLine],
    vecs: &HashMap<String, [f32; D]>,
    main: &mut EmbedStore,
) -> Result<usize> {
    let mut rows = Vec::new();
    for v in videos {
        if main.contains(&v.id) {
            continue;
        }
        let fv: Option<Vec<&[f32; D]>> = (0..v.frames.len())
            .map(|i| vecs.get(&frame_id(&v.id, i)))
            .collect();
        if let Some(fv) = fv {
            rows.push((
                v.id.clone(),
                mean_unit(&fv).with_context(|| format!("video {}", v.id))?,
            ));
        }
    }
    main.append(&rows, Kind::Video)?;
    Ok(rows.len())
}

fn short(id: &str) -> &str {
    &id[..8.min(id.len())]
}

enum Event<'a> {
    Failed(&'a Failure),
    Progress { done: usize },
    Note(&'a str),
}

/// Stores one chunk of `(row id, vector)`.
type Commit<'a> = dyn FnMut(&[(String, [f32; D])]) -> Result<()> + 'a;

/// Embed `todo` in order; `commit` stores each chunk. Returns (embedded, failed).
fn embed_list(
    todo: &[Job],
    failures_dir: &Path,
    emb: &mut dyn ImageEmbedder,
    opts: &EmbedOpts,
    lock: &mut GpuLock,
    commit: &mut Commit<'_>,
    on: &mut dyn FnMut(Event<'_>),
) -> Result<(usize, usize)> {
    let batch = opts.batch.max(1);
    let chunk_rows = opts.chunk.max(1);
    let every = opts.progress_every.max(1);
    let (tx, rx) = mpsc::sync_channel::<(usize, Result<Patches, String>)>(2 * batch + 2);
    std::thread::scope(|s| -> Result<(usize, usize)> {
        s.spawn(move || {
            for (i, j) in todo.iter().enumerate() {
                let r = prepare(&j.path).map_err(|e| format!("{e:#}"));
                if tx.send((i, r)).is_err() {
                    return; // the consumer stopped
                }
            }
        });
        let mut ok = 0usize;
        let mut bad = 0usize;
        let mut failed_assets: HashSet<&str> = HashSet::new();
        let mut wait: Vec<(usize, Patches)> = Vec::with_capacity(batch);
        let mut chunk: Vec<(String, [f32; D])> = Vec::with_capacity(chunk_rows + batch);
        let mut run_batch = |wait: &mut Vec<(usize, Patches)>,
                             chunk: &mut Vec<(String, [f32; D])>,
                             lock: &mut GpuLock|
         -> Result<()> {
            if wait.is_empty() {
                return Ok(());
            }
            lock.acquire()?;
            let patches: Vec<Patches> = wait.iter().map(|(_, p)| p.clone()).collect();
            let vs = emb.embed(&patches).with_context(|| {
                format!(
                    "model failed on {} (batch of {}); this is fatal (GPU or model error)",
                    todo[wait[0].0].id,
                    wait.len()
                )
            })?;
            anyhow::ensure!(
                vs.len() == wait.len(),
                "embedder returned {} vectors for {}",
                vs.len(),
                wait.len()
            );
            for ((i, _), v) in wait.drain(..).zip(vs) {
                chunk.push((todo[i].id.clone(), v));
            }
            Ok(())
        };
        let mut flush = |chunk: &mut Vec<(String, [f32; D])>,
                         lock: &mut GpuLock,
                         ok: &mut usize|
         -> Result<()> {
            commit(chunk)?;
            *ok += chunk.len();
            let k = chunk.len();
            chunk.clear();
            lock.committed(k)
        };
        for (i, r) in rx {
            if failed_assets.contains(todo[i].asset.as_str()) {
                continue; // another frame of this video already failed
            }
            match r {
                Ok(p) => wait.push((i, p)),
                Err(error) => {
                    let f = Failure {
                        id: todo[i].asset.clone(),
                        file: todo[i].file.clone(),
                        error,
                        at: now_iso(),
                    };
                    store::append_failure(failures_dir, &f)?;
                    failed_assets.insert(todo[i].asset.as_str());
                    bad += 1;
                    on(Event::Failed(&f));
                }
            }
            if wait.len() >= batch {
                let before = ok + chunk.len();
                if let Err(e) = run_batch(&mut wait, &mut chunk, lock) {
                    // Rows computed before the failure are valid: keep them.
                    commit(&chunk)?;
                    return Err(e);
                }
                let after = ok + chunk.len();
                if after / every > before / every {
                    on(Event::Progress { done: after });
                }
            }
            if chunk.len() >= chunk_rows {
                flush(&mut chunk, lock, &mut ok)?;
                for m in lock.drain_log() {
                    on(Event::Note(&m));
                }
            }
        }
        if let Err(e) = run_batch(&mut wait, &mut chunk, lock) {
            commit(&chunk)?;
            return Err(e);
        }
        flush(&mut chunk, lock, &mut ok)?;
        on(Event::Progress { done: ok });
        Ok((ok, bad))
    })
}
