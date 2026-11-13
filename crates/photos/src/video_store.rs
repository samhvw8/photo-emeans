//! Resumable bulk fetch of video frames into `$EMEANS_DATA/videos`.
//!
//! - The frames of one video go to `<uuid>.tmp/fNN.jpg`, each fsynced; the directory is fsynced
//!   and renamed to `<uuid>/`. A `<uuid>/` left without a manifest line (crash between rename and
//!   append) is replaced on the next fetch.
//! - Only after the rename, the single writer thread appends the manifest line in one `write_all`.
//! - On start the manifest is checked like `photos/manifest.jsonl`: a torn last line is dropped,
//!   lines whose frames are missing or not JPEGs of the recorded size are dropped (and refetched),
//!   duplicate ids keep their first line, and the manifest is rewritten atomically when anything
//!   was dropped. A bad line in the middle stops the run. Leftover `*.tmp` directories are deleted.
//! - A video that fails after all attempts gets a line in `failures.jsonl` and no manifest line.
//!   Later runs skip it unless `retry_failed` (each retry may download again).

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use objc2::rc::autoreleasepool;
use serde::{Deserialize, Serialize};

use crate::assets::{AssetInfo, iso8601_utc};
use crate::fetch::{is_main_thread, pump_main_run_loop};
use crate::store::{MANIFEST, jpeg_size};
use crate::video::{VideoConfig, fetch_video_frames};

pub const FAILURES: &str = "failures.jsonl";

/// One frame in a [`VideoLine`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameLine {
    /// Path relative to `videos/`: `<uuid>/fNN.jpg`.
    pub file: String,
    /// Time in the video in seconds; `null` for the poster frame (`f00`).
    pub t: Option<f64>,
    pub width: u32,
    pub height: u32,
}

/// One line of `videos/manifest.jsonl` (see AGENTS.md, data contract).
///
/// `width` and `height` are the pixel size of the poster JPEG (`frames[0]`), as for photos.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoLine {
    pub id: String,
    pub created: Option<String>,
    /// Seconds.
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub favorite: bool,
    pub subtypes: Vec<String>,
    /// Poster first, then the sampled frames in time order.
    pub frames: Vec<FrameLine>,
}

impl VideoLine {
    /// The poster frame's file (relative to `videos/`).
    pub fn poster(&self) -> &str {
        &self.frames[0].file
    }
}

/// One line of `videos/failures.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoFailure {
    pub id: String,
    pub error: String,
    pub at: String,
}

/// Options for [`fetch_all_videos`].
#[derive(Debug, Clone)]
pub struct VideoFetchAllConfig {
    /// Videos in flight at once.
    pub concurrency: usize,
    pub video: VideoConfig,
    /// Fetch videos listed in `failures.jsonl` again.
    pub retry_failed: bool,
    /// Stop starting new videos when the volume of `dir` has less free space than this. Photos
    /// keeps every downloaded rendition in its library (as a purgeable cache), so a full run grows
    /// the library by about the sum of the renditions.
    pub min_free_bytes: u64,
}

impl Default for VideoFetchAllConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            video: VideoConfig::default(),
            retry_failed: false,
            min_free_bytes: 8_000_000_000,
        }
    }
}

/// Progress events, delivered on the calling thread.
#[derive(Debug)]
pub enum VideoFetchEvent<'a> {
    Done {
        line: &'a VideoLine,
        bytes: usize,
        from_icloud: bool,
        rendition_bytes: Option<u64>,
        request: Duration,
        decode: Duration,
        elapsed: Duration,
    },
    Failed {
        id: &'a str,
        error: &'a str,
    },
}

/// Summary of a [`fetch_all_videos`] run.
#[derive(Debug, Default)]
pub struct VideoFetchReport {
    /// Videos already in the manifest with valid frames.
    pub skipped: usize,
    /// Videos skipped because they are in `failures.jsonl`.
    pub skipped_failed: usize,
    pub fetched: usize,
    pub from_icloud: usize,
    pub frames: usize,
    /// JPEG bytes written.
    pub bytes: u64,
    /// Sum of the rendition file sizes Photos handed out (when known).
    pub rendition_bytes: u64,
    pub failed: Vec<(String, String)>,
    /// Manifest lines dropped at start (torn, invalid frames, duplicate).
    pub repaired: usize,
    /// The run stopped early because free disk space fell below `min_free_bytes`.
    pub stopped_low_disk: Option<u64>,
    /// A stop was asked (SIGINT/SIGTERM, see [`crate::stop`]): the run started no new video.
    pub interrupted: Option<i32>,
    pub elapsed: Duration,
}

/// Free bytes on the volume that holds `path` (`NSFileSystemFreeSize`).
pub fn free_bytes(path: &Path) -> Result<u64> {
    use objc2_foundation::{NSFileManager, NSFileSystemFreeSize, NSNumber, NSString};
    let p = NSString::from_str(&path.to_string_lossy());
    // SAFETY: plain Foundation calls with valid arguments; the key is an extern constant.
    unsafe {
        let attrs = NSFileManager::defaultManager()
            .attributesOfFileSystemForPath_error(&p)
            .map_err(|e| {
                anyhow::anyhow!(
                    "free space of {}: {}",
                    path.display(),
                    e.localizedDescription()
                )
            })?;
        attrs
            .objectForKey(NSFileSystemFreeSize)
            .and_then(|v| v.downcast::<NSNumber>().ok())
            .map(|n| n.as_u64())
            .context("no NSFileSystemFreeSize")
    }
}

fn frames_ok(dir: &Path, line: &VideoLine) -> bool {
    !line.frames.is_empty()
        && line.frames.iter().all(|f| {
            fs::read(dir.join(&f.file))
                .ok()
                .and_then(|b| jpeg_size(&b))
                .is_some_and(|wh| wh == (f.width, f.height))
        })
}

/// Read and repair `dir/manifest.jsonl`. Returns the valid lines and how many were dropped.
pub fn load_video_manifest(dir: &Path) -> Result<(Vec<VideoLine>, usize)> {
    let path = dir.join(MANIFEST);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let complete = text.ends_with('\n') || text.is_empty();
    let raw: Vec<&str> = text.lines().collect();
    let mut lines = Vec::new();
    let mut dropped = 0usize;
    let mut seen = HashSet::new();
    for (n, l) in raw.iter().enumerate() {
        let last = n + 1 == raw.len();
        let parsed: Option<VideoLine> = serde_json::from_str(l).ok();
        let Some(line) = parsed else {
            if last && !complete {
                dropped += 1;
                continue;
            }
            bail!(
                "{} line {}: not a video manifest line; refusing to continue on a corrupt manifest",
                path.display(),
                n + 1
            );
        };
        if last && !complete {
            dropped += 1;
            continue;
        }
        if !seen.insert(line.id.clone()) || !frames_ok(dir, &line) {
            dropped += 1;
            continue;
        }
        lines.push(line);
    }
    if dropped > 0 {
        let tmp = dir.join(format!("{MANIFEST}.tmp"));
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            for l in &lines {
                w.write_all(serde_json::to_string(l)?.as_bytes())?;
                w.write_all(b"\n")?;
            }
            w.into_inner()?.sync_all()?;
        }
        fs::rename(&tmp, &path)?;
    }
    Ok((lines, dropped))
}

/// Complete lines of `failures.jsonl` (missing file = none).
pub fn read_video_failures(dir: &Path) -> Result<Vec<VideoFailure>> {
    let text = match fs::read_to_string(dir.join(FAILURES)) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).context("reading videos/failures.jsonl"),
    };
    let complete = text.rfind('\n').map_or("", |i| &text[..=i]);
    complete
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("parsing videos/failures.jsonl"))
        .collect()
}

fn append_failure(dir: &Path, f: &VideoFailure) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(FAILURES))
        .context("opening videos/failures.jsonl")?;
    let mut s = serde_json::to_string(f)?;
    s.push('\n');
    file.write_all(s.as_bytes())?;
    file.sync_data()?;
    Ok(())
}

fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("fsync {}", dir.display()))
}

/// Write the frames of one video into `dir/<stem>/` atomically (temporary directory + rename).
/// Returns the frame lines. `frames` is `(jpeg, width, height, t)`, poster first.
pub fn write_video_dir(
    dir: &Path,
    stem: &str,
    frames: &[(&[u8], u32, u32, Option<f64>)],
) -> Result<Vec<FrameLine>> {
    let tmp = dir.join(format!("{stem}.tmp"));
    let dst = dir.join(stem);
    if tmp.exists() {
        fs::remove_dir_all(&tmp)?;
    }
    fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut out = Vec::with_capacity(frames.len());
    for (i, (jpeg, w, h, t)) in frames.iter().enumerate() {
        let name = format!("f{i:02}.jpg");
        let mut f = File::create(tmp.join(&name))?;
        f.write_all(jpeg)?;
        f.sync_all()?;
        out.push(FrameLine {
            file: format!("{stem}/{name}"),
            t: *t,
            width: *w,
            height: *h,
        });
    }
    sync_dir(&tmp)?;
    if dst.exists() {
        // A directory without a manifest line, from a run killed between rename and append.
        fs::remove_dir_all(&dst)?;
    }
    fs::rename(&tmp, &dst).with_context(|| format!("renaming to {}", dst.display()))?;
    sync_dir(dir)?;
    Ok(out)
}

enum Msg {
    Ok {
        line: VideoLine,
        bytes: usize,
        from_icloud: bool,
        rendition_bytes: Option<u64>,
        request: Duration,
        decode: Duration,
        elapsed: Duration,
    },
    Err {
        id: String,
        error: String,
    },
}

/// Fetch the frames of every video not yet in `dir`'s manifest, `cfg.concurrency` at a time.
///
/// Must run on the main thread (it runs the main run loop while workers wait for Photos).
pub fn fetch_all_videos(
    dir: &Path,
    assets: &[AssetInfo],
    cfg: &VideoFetchAllConfig,
    mut on_event: impl FnMut(VideoFetchEvent<'_>),
) -> Result<VideoFetchReport> {
    if !is_main_thread() {
        bail!(
            "fetch_all_videos must run on the main thread: Photos may deliver results on the main \
             queue, and fetch_all_videos runs the main run loop to receive them"
        );
    }
    let started = Instant::now();
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "tmp") {
            if p.is_dir() {
                fs::remove_dir_all(&p)?;
            } else {
                fs::remove_file(&p)?;
            }
        }
    }
    let (existing, repaired) = load_video_manifest(dir)?;
    let done: HashSet<&str> = existing.iter().map(|l| l.id.as_str()).collect();
    let failed_before: HashSet<String> = if cfg.retry_failed {
        HashSet::new()
    } else {
        read_video_failures(dir)?
            .into_iter()
            .map(|f| f.id)
            .collect()
    };
    let todo: Vec<&AssetInfo> = assets
        .iter()
        .filter(|a| !done.contains(a.id.as_str()) && !failed_before.contains(&a.id))
        .collect();
    let mut report = VideoFetchReport {
        skipped: assets
            .iter()
            .filter(|a| done.contains(a.id.as_str()))
            .count(),
        skipped_failed: assets
            .iter()
            .filter(|a| !done.contains(a.id.as_str()) && failed_before.contains(&a.id))
            .count(),
        repaired,
        ..VideoFetchReport::default()
    };

    let mut manifest = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(MANIFEST))
        .context("opening videos/manifest.jsonl")?;
    let next = AtomicUsize::new(0);
    let low_disk = std::sync::atomic::AtomicU64::new(0);
    let (tx, rx) = mpsc::channel::<Msg>();
    let workers = cfg.concurrency.clamp(1, 16).min(todo.len().max(1));

    std::thread::scope(|s| -> Result<()> {
        for _ in 0..workers {
            let tx = tx.clone();
            let (todo, next, low_disk) = (&todo, &next, &low_disk);
            s.spawn(move || {
                loop {
                    if low_disk.load(Ordering::Relaxed) > 0
                        || crate::stop::stop_requested().is_some()
                    {
                        break;
                    }
                    if cfg.min_free_bytes > 0 {
                        match free_bytes(dir) {
                            Ok(f) if f < cfg.min_free_bytes => {
                                low_disk.store(f.max(1), Ordering::Relaxed);
                                break;
                            }
                            Ok(_) => {}
                            Err(_) => {
                                low_disk.store(1, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(a) = todo.get(i) else { break };
                    let t0 = Instant::now();
                    let msg = autoreleasepool(|_| match fetch_video_frames(a, &cfg.video) {
                        Ok(v) => {
                            let frames: Vec<_> = v
                                .frames
                                .iter()
                                .map(|f| (f.jpeg.as_slice(), f.width, f.height, f.t))
                                .collect();
                            match write_video_dir(dir, &a.file_stem(), &frames) {
                                Ok(fl) => Msg::Ok {
                                    line: VideoLine {
                                        id: a.id.clone(),
                                        created: a.created.clone(),
                                        duration: (a.duration * 1000.0).round() / 1000.0,
                                        width: fl[0].width,
                                        height: fl[0].height,
                                        favorite: a.favorite,
                                        subtypes: a.subtypes.clone(),
                                        frames: fl,
                                    },
                                    bytes: v.frames.iter().map(|f| f.jpeg.len()).sum(),
                                    from_icloud: v.from_icloud,
                                    rendition_bytes: v.rendition_bytes,
                                    request: v.request_time,
                                    decode: v.decode_time,
                                    elapsed: t0.elapsed(),
                                },
                                Err(e) => Msg::Err {
                                    id: a.id.clone(),
                                    error: format!("{e:#}"),
                                },
                            }
                        }
                        Err(e) => Msg::Err {
                            id: a.id.clone(),
                            error: format!("{e:#}"),
                        },
                    });
                    if tx.send(msg).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        loop {
            let msg = match rx.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => {
                    pump_main_run_loop(Duration::from_millis(20));
                    continue;
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            match msg {
                Msg::Ok {
                    line,
                    bytes,
                    from_icloud,
                    rendition_bytes,
                    request,
                    decode,
                    elapsed,
                } => {
                    let mut buf = serde_json::to_vec(&line)?;
                    buf.push(b'\n');
                    manifest
                        .write_all(&buf)
                        .context("appending videos/manifest.jsonl")?;
                    manifest.flush()?;
                    report.fetched += 1;
                    report.frames += line.frames.len();
                    report.bytes += bytes as u64;
                    report.rendition_bytes += rendition_bytes.unwrap_or(0);
                    report.from_icloud += usize::from(from_icloud);
                    on_event(VideoFetchEvent::Done {
                        line: &line,
                        bytes,
                        from_icloud,
                        rendition_bytes,
                        request,
                        decode,
                        elapsed,
                    });
                }
                Msg::Err { id, error } => {
                    append_failure(
                        dir,
                        &VideoFailure {
                            id: id.clone(),
                            error: error.clone(),
                            at: iso8601_utc(
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map_or(0.0, |d| d.as_secs_f64()),
                            ),
                        },
                    )?;
                    on_event(VideoFetchEvent::Failed {
                        id: &id,
                        error: &error,
                    });
                    report.failed.push((id, error));
                }
            }
        }
        Ok(())
    })?;
    manifest.sync_all()?;
    let ld = low_disk.load(Ordering::Relaxed);
    report.stopped_low_disk = (ld > 0).then_some(ld);
    report.interrupted = crate::stop::stop_requested();
    report.elapsed = started.elapsed();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tiny_jpeg(w: u16, h: u16) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("videos-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn video(d: &Path, stem: &str, n: usize) -> VideoLine {
        let j = tiny_jpeg(8, 6);
        let frames: Vec<_> = (0..n)
            .map(|i| {
                (
                    j.as_slice(),
                    8,
                    6,
                    if i == 0 { None } else { Some(i as f64) },
                )
            })
            .collect();
        let fl = write_video_dir(d, stem, &frames).unwrap();
        VideoLine {
            id: format!("{stem}/L0/001"),
            created: Some("2024-01-01T00:00:00Z".into()),
            duration: n as f64,
            width: 8,
            height: 6,
            favorite: false,
            subtypes: vec![],
            frames: fl,
        }
    }

    #[test]
    fn line_field_order_matches_contract() {
        let l = VideoLine {
            id: "V/L0/001".into(),
            created: None,
            duration: 2.5,
            width: 4,
            height: 3,
            favorite: true,
            subtypes: vec!["timelapse".into()],
            frames: vec![FrameLine {
                file: "V/f00.jpg".into(),
                t: None,
                width: 4,
                height: 3,
            }],
        };
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"id":"V/L0/001","created":null,"duration":2.5,"width":4,"height":3,"favorite":true,"subtypes":["timelapse"],"frames":[{"file":"V/f00.jpg","t":null,"width":4,"height":3}]}"#
        );
    }

    #[test]
    fn write_dir_is_atomic_and_replaces_orphans() {
        let d = tmpdir("write");
        let a = video(&d, "A", 3);
        assert_eq!(a.frames[2].file, "A/f02.jpg");
        assert_eq!(a.frames[0].t, None);
        assert!(!d.join("A.tmp").exists());
        // A second write (orphan directory without a manifest line) replaces it.
        let b = video(&d, "A", 2);
        assert_eq!(b.frames.len(), 2);
        assert!(!d.join("A/f02.jpg").exists());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn load_repairs_torn_missing_frame_and_duplicate() {
        let d = tmpdir("repair");
        let a = video(&d, "A", 3);
        let b = video(&d, "B", 3);
        let c = video(&d, "C", 2);
        fs::remove_file(d.join("B/f01.jpg")).unwrap(); // a frame is missing
        let mut text = String::new();
        for l in [&a, &a, &b, &c] {
            text += &serde_json::to_string(l).unwrap();
            text.push('\n');
        }
        text += r#"{"id":"D/L0/001","dur"#; // torn
        fs::write(d.join(MANIFEST), &text).unwrap();
        let (lines, dropped) = load_video_manifest(&d).unwrap();
        assert_eq!(lines, vec![a.clone(), c.clone()]);
        assert_eq!(dropped, 3);
        assert_eq!(load_video_manifest(&d).unwrap().1, 0);
        fs::write(
            d.join(MANIFEST),
            format!("junk\n{}\n", serde_json::to_string(&a).unwrap()),
        )
        .unwrap();
        assert!(load_video_manifest(&d).is_err());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn failures_round_trip() {
        let d = tmpdir("fail");
        assert!(read_video_failures(&d).unwrap().is_empty());
        let f = VideoFailure {
            id: "X/L0/001".into(),
            error: "boom".into(),
            at: "2026-01-01T00:00:00Z".into(),
        };
        append_failure(&d, &f).unwrap();
        // A torn line after it is ignored.
        let mut t = fs::read_to_string(d.join(FAILURES)).unwrap();
        t += r#"{"id":"Y"#;
        fs::write(d.join(FAILURES), t).unwrap();
        assert_eq!(read_video_failures(&d).unwrap(), vec![f]);
        fs::remove_dir_all(&d).unwrap();
    }
}
