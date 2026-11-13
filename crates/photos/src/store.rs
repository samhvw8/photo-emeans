//! Resumable bulk fetch into `$EMEANS_DATA/photos`.
//!
//! - Each JPEG goes to `<uuid>.jpg.tmp`, is fsynced, then renamed to `<uuid>.jpg`.
//! - Only after the rename does the single writer thread append the manifest line, with one
//!   `write_all` of the whole line plus `\n`.
//! - On start, the manifest is checked: a torn last line (crash mid-write) is dropped, entries
//!   whose file is missing or not a valid JPEG of the recorded size are dropped (and refetched),
//!   duplicate ids keep their first line. If anything was dropped the manifest is rewritten
//!   atomically. A bad line in the middle is corruption, and the run stops.
//! - Leftover `*.jpg.tmp` files are deleted. A `.jpg` without a manifest line is refetched.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use objc2::rc::autoreleasepool;
use serde::{Deserialize, Serialize};

use crate::assets::AssetInfo;
use crate::fetch::{FetchConfig, fetch_resized, is_main_thread, pump_main_run_loop};

pub const MANIFEST: &str = "manifest.jsonl";

/// One line of `photos/manifest.jsonl` (see AGENTS.md, data contract).
///
/// `width` and `height` are the pixel size of the JPEG file, not of the original.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestLine {
    pub id: String,
    pub file: String,
    pub created: Option<String>,
    pub width: u32,
    pub height: u32,
    pub favorite: bool,
    pub burst_id: Option<String>,
    pub subtypes: Vec<String>,
}

/// Options for [`fetch_all`].
#[derive(Debug, Clone)]
pub struct FetchAllConfig {
    /// Requests in flight at once.
    pub concurrency: usize,
    pub fetch: FetchConfig,
}

impl Default for FetchAllConfig {
    fn default() -> Self {
        Self {
            concurrency: 6,
            fetch: FetchConfig::default(),
        }
    }
}

/// Progress events, delivered on the calling thread.
#[derive(Debug)]
pub enum FetchEvent<'a> {
    Done {
        line: &'a ManifestLine,
        bytes: usize,
        from_icloud: bool,
        attempts: u32,
        elapsed: Duration,
    },
    Failed {
        id: &'a str,
        error: &'a str,
    },
}

/// Summary of a [`fetch_all`] run.
#[derive(Debug, Default)]
pub struct FetchReport {
    /// Assets already in the manifest with a valid file.
    pub skipped: usize,
    pub fetched: usize,
    pub from_icloud: usize,
    pub bytes: u64,
    /// `(id, error)` for assets that failed after all attempts. They have no manifest line.
    pub failed: Vec<(String, String)>,
    /// Manifest lines dropped at start (torn, invalid file, duplicate).
    pub repaired: usize,
    /// A stop was asked (SIGINT/SIGTERM, see [`crate::stop`]): the run started no new photo.
    pub interrupted: Option<i32>,
    pub elapsed: Duration,
}

/// Width and height from a JPEG's SOF marker, if the data is a complete JPEG (SOI ... EOI).
pub fn jpeg_size(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 4 || b[..2] != [0xFF, 0xD8] || b[b.len() - 2..] != [0xFF, 0xD9] {
        return None;
    }
    let mut i = 2;
    while i + 4 <= b.len() {
        if b[i] != 0xFF {
            return None;
        }
        let m = b[i + 1];
        if m == 0xFF {
            i += 1;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([b[i + 2], b[i + 3]]));
        let is_sof = matches!(m, 0xC0..=0xCF) && !matches!(m, 0xC4 | 0xC8 | 0xCC);
        if is_sof {
            if i + 9 > b.len() {
                return None;
            }
            let h = u16::from_be_bytes([b[i + 5], b[i + 6]]);
            let w = u16::from_be_bytes([b[i + 7], b[i + 8]]);
            return Some((u32::from(w), u32::from(h)));
        }
        i += 2 + len;
    }
    None
}

fn file_matches(path: &Path, line: &ManifestLine) -> bool {
    fs::read(path)
        .ok()
        .and_then(|b| jpeg_size(&b))
        .is_some_and(|wh| wh == (line.width, line.height))
}

/// Read and repair the manifest in `dir`. Returns the valid lines and how many were dropped.
pub fn load_manifest(dir: &Path) -> Result<(Vec<ManifestLine>, usize)> {
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
        let parsed: Option<ManifestLine> = serde_json::from_str(l).ok();
        let Some(line) = parsed else {
            if last && !complete {
                dropped += 1; // torn write at a crash
                continue;
            }
            bail!(
                "{} line {}: not a manifest line; refusing to continue on a corrupt manifest",
                path.display(),
                n + 1
            );
        };
        if last && !complete {
            // Parsed, but the newline is missing: the write may have been cut. Refetch it.
            dropped += 1;
            continue;
        }
        if !seen.insert(line.id.clone()) || !file_matches(&dir.join(&line.file), &line) {
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

fn write_atomic(dir: &Path, name: &str, data: &[u8]) -> Result<PathBuf> {
    let tmp = dir.join(format!("{name}.tmp"));
    let dst = dir.join(name);
    {
        let mut f = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &dst).with_context(|| format!("renaming to {}", dst.display()))?;
    Ok(dst)
}

enum Msg {
    Ok {
        line: ManifestLine,
        bytes: usize,
        from_icloud: bool,
        attempts: u32,
        elapsed: Duration,
    },
    Err {
        id: String,
        error: String,
    },
}

/// Fetch every asset not yet in `dir`'s manifest, `cfg.concurrency` at a time.
///
/// Must run on the main thread (it runs the main run loop while workers wait; see
/// [`crate::fetch::fetch_resized`]). Manifest lines are appended in completion order. Assets that fail after all attempts are
/// reported in [`FetchReport::failed`] and get no line, so the next run retries them.
pub fn fetch_all(
    dir: &Path,
    assets: &[AssetInfo],
    cfg: &FetchAllConfig,
    mut on_event: impl FnMut(FetchEvent<'_>),
) -> Result<FetchReport> {
    if !is_main_thread() {
        bail!(
            "fetch_all must run on the main thread: Photos delivers image results on the main \
             queue, and fetch_all runs the main run loop to receive them"
        );
    }
    let started = Instant::now();
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "tmp") {
            fs::remove_file(&p)?;
        }
    }
    let (existing, repaired) = load_manifest(dir)?;
    let done: HashSet<&str> = existing.iter().map(|l| l.id.as_str()).collect();
    let todo: Vec<&AssetInfo> = assets
        .iter()
        .filter(|a| !done.contains(a.id.as_str()))
        .collect();
    let mut report = FetchReport {
        skipped: assets.len() - todo.len(),
        repaired,
        ..FetchReport::default()
    };

    let mut manifest = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(MANIFEST))
        .context("opening manifest")?;
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<Msg>();
    let workers = cfg.concurrency.clamp(1, 32).min(todo.len().max(1));

    std::thread::scope(|s| -> Result<()> {
        for _ in 0..workers {
            let tx = tx.clone();
            let (todo, next) = (&todo, &next);
            s.spawn(move || {
                loop {
                    if crate::stop::stop_requested().is_some() {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(a) = todo.get(i) else { break };
                    let t0 = Instant::now();
                    let msg = autoreleasepool(|_| match fetch_resized(a, &cfg.fetch) {
                        Ok(f) => {
                            let file = format!("{}.jpg", a.file_stem());
                            match write_atomic(dir, &file, &f.jpeg) {
                                Ok(_) => Msg::Ok {
                                    line: ManifestLine {
                                        id: a.id.clone(),
                                        file,
                                        created: a.created.clone(),
                                        width: f.width,
                                        height: f.height,
                                        favorite: a.favorite,
                                        burst_id: a.burst_id.clone(),
                                        subtypes: a.subtypes.clone(),
                                    },
                                    bytes: f.jpeg.len(),
                                    from_icloud: f.from_icloud,
                                    attempts: f.attempts,
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
                    // Photos delivers the workers' results on the main queue; run it.
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
                    attempts,
                    elapsed,
                } => {
                    let mut buf = serde_json::to_vec(&line)?;
                    buf.push(b'\n');
                    manifest
                        .write_all(&buf)
                        .context("appending manifest line")?;
                    manifest.flush()?;
                    report.fetched += 1;
                    report.bytes += bytes as u64;
                    report.from_icloud += usize::from(from_icloud);
                    on_event(FetchEvent::Done {
                        line: &line,
                        bytes,
                        from_icloud,
                        attempts,
                        elapsed,
                    });
                }
                Msg::Err { id, error } => {
                    on_event(FetchEvent::Failed {
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
    report.interrupted = crate::stop::stop_requested();
    report.elapsed = started.elapsed();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg(w: u16, h: u16) -> Vec<u8> {
        // SOI, SOF0 with the size, EOI: enough for jpeg_size.
        let mut v = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    fn line(id: &str, w: u32, h: u32) -> ManifestLine {
        ManifestLine {
            id: format!("{id}/L0/001"),
            file: format!("{id}.jpg"),
            created: Some("2024-01-01T00:00:00Z".into()),
            width: w,
            height: h,
            favorite: false,
            burst_id: None,
            subtypes: vec![],
        }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photos-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn jpeg_size_parses_sof() {
        assert_eq!(jpeg_size(&tiny_jpeg(1024, 768)), Some((1024, 768)));
        let mut cut = tiny_jpeg(10, 10);
        cut.pop();
        assert_eq!(jpeg_size(&cut), None);
        assert_eq!(jpeg_size(b"nope"), None);
    }

    #[test]
    fn manifest_field_order_matches_contract() {
        let s = serde_json::to_string(&line("A", 3, 4)).unwrap();
        assert_eq!(
            s,
            r#"{"id":"A/L0/001","file":"A.jpg","created":"2024-01-01T00:00:00Z","width":3,"height":4,"favorite":false,"burst_id":null,"subtypes":[]}"#
        );
    }

    #[test]
    fn load_repairs_torn_missing_and_duplicate() {
        let d = tmpdir("repair");
        fs::write(d.join("A.jpg"), tiny_jpeg(4, 3)).unwrap();
        fs::write(d.join("B.jpg"), tiny_jpeg(4, 3)).unwrap(); // wrong size in manifest below
        let mut text = String::new();
        for l in [
            line("A", 4, 3),
            line("A", 4, 3),
            line("B", 8, 6),
            line("C", 4, 3),
        ] {
            text += &serde_json::to_string(&l).unwrap();
            text.push('\n');
        }
        text += r#"{"id":"D/L0/001","fi"#; // torn
        fs::write(d.join(MANIFEST), &text).unwrap();
        let (lines, dropped) = load_manifest(&d).unwrap();
        assert_eq!(lines, vec![line("A", 4, 3)]);
        assert_eq!(dropped, 4);
        let again = fs::read_to_string(d.join(MANIFEST)).unwrap();
        assert_eq!(
            again,
            serde_json::to_string(&line("A", 4, 3)).unwrap() + "\n"
        );
        assert_eq!(load_manifest(&d).unwrap().1, 0);
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn load_refuses_corrupt_middle_line() {
        let d = tmpdir("corrupt");
        fs::write(d.join("A.jpg"), tiny_jpeg(4, 3)).unwrap();
        let text = format!(
            "garbage\n{}\n",
            serde_json::to_string(&line("A", 4, 3)).unwrap()
        );
        fs::write(d.join(MANIFEST), text).unwrap();
        assert!(load_manifest(&d).is_err());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn write_atomic_leaves_no_tmp() {
        let d = tmpdir("atomic");
        let p = write_atomic(&d, "x.jpg", b"abc").unwrap();
        assert_eq!(fs::read(p).unwrap(), b"abc");
        assert!(!d.join("x.jpg.tmp").exists());
        fs::remove_dir_all(&d).unwrap();
    }
}
