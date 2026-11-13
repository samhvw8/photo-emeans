//! Read-only views of `photos/manifest.jsonl` and `videos/manifest.jsonl`, and the joined
//! per-asset view ([`media_by_id`]) that `cluster`, `gallery` and `albums` use.
//!
//! `photos::load_manifest` repairs and rewrites the manifest, and it reads every JPEG to check it.
//! The embed step must not write to `photos/` while a fetch may still append to it, so this reader
//! only reads: it takes complete lines, ignores a torn last line, keeps the first of duplicate
//! ids, and stops on a corrupt line in the middle.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
pub use photos::{ManifestLine, VideoLine};
use serde::de::DeserializeOwned;

use crate::store::Kind;

trait HasId {
    fn id(&self) -> &str;
}
impl HasId for ManifestLine {
    fn id(&self) -> &str {
        &self.id
    }
}
impl HasId for VideoLine {
    fn id(&self) -> &str {
        &self.id
    }
}

/// Complete photo manifest lines in file order (fetch completion order). Missing file = empty.
pub fn read(photos_dir: &Path) -> Result<Vec<ManifestLine>> {
    read_lines(photos_dir)
}

/// Complete video manifest lines in file order. Missing file = empty.
pub fn read_videos(videos_dir: &Path) -> Result<Vec<VideoLine>> {
    read_lines(videos_dir)
}

fn read_lines<T: DeserializeOwned + HasId>(dir: &Path) -> Result<Vec<T>> {
    let path = dir.join(photos::store::MANIFEST);
    let text = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    // Only bytes up to the last newline are complete; a fetch may be writing the rest.
    let end = text.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let text = std::str::from_utf8(&text[..end])
        .with_context(|| format!("{} is not UTF-8", path.display()))?;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (n, l) in text.lines().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let line: T = match serde_json::from_str(l) {
            Ok(x) => x,
            Err(e) => bail!(
                "{} line {}: not a manifest line ({e}); refusing to continue on a corrupt manifest",
                path.display(),
                n + 1
            ),
        };
        if seen.insert(line.id().to_string()) {
            out.push(line);
        }
    }
    Ok(out)
}

/// One photo or video, as `cluster`, `gallery` and `albums` see it.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub kind: Kind,
    /// Image shown for it, relative to `$EMEANS_DATA`: `photos/<uuid>.jpg` or the poster
    /// `videos/<uuid>/f00.jpg`.
    pub image: String,
    pub created: Option<String>,
    pub favorite: bool,
    pub burst_id: Option<String>,
    /// Size of the stored copy (photo JPEG, or the video's poster JPEG), not of the original.
    pub width: u32,
    pub height: u32,
    /// PhotoKit `mediaSubtypes` names (`screenshot`, `live`, ...).
    pub subtypes: Vec<String>,
    /// Seconds (videos; 0 for photos).
    pub duration: f64,
    /// Frame images relative to `$EMEANS_DATA`, poster first (videos; empty for photos).
    pub frames: Vec<String>,
}

/// Every fetched photo and video by asset id. An id in both manifests is an error.
pub fn media_by_id(data: &Path) -> Result<HashMap<String, Item>> {
    let mut out: HashMap<String, Item> = read(&data.join("photos"))?
        .into_iter()
        .map(|l| {
            (
                l.id.clone(),
                Item {
                    kind: Kind::Photo,
                    image: format!("photos/{}", l.file),
                    created: l.created,
                    favorite: l.favorite,
                    burst_id: l.burst_id,
                    width: l.width,
                    height: l.height,
                    subtypes: l.subtypes,
                    duration: 0.0,
                    frames: Vec::new(),
                },
            )
        })
        .collect();
    for v in read_videos(&data.join("videos"))? {
        ensure!(
            !v.frames.is_empty(),
            "video {} has no frames in videos/manifest.jsonl",
            v.id
        );
        let item = Item {
            kind: Kind::Video,
            image: format!("videos/{}", v.poster()),
            created: v.created.clone(),
            favorite: v.favorite,
            burst_id: None,
            width: v.width,
            height: v.height,
            subtypes: v.subtypes.clone(),
            duration: v.duration,
            frames: v
                .frames
                .iter()
                .map(|f| format!("videos/{}", f.file))
                .collect(),
        };
        ensure!(
            out.insert(v.id.clone(), item).is_none(),
            "asset {} is in both photos/ and videos/ manifests",
            v.id
        );
    }
    Ok(out)
}

/// True when the fetch has ended: the log's last non-empty line, after any `=== ... DONE` lines
/// a wrapper script adds, is the final summary that `fetch`, `fetch-videos` (and the `photos`
/// probe) print (`fetched N ...`), or the `Error: ...` line of a run that stopped.
pub fn fetch_finished(log: &Path) -> bool {
    let Ok(text) = fs::read_to_string(log) else {
        return false;
    };
    text.lines()
        .rev()
        .find(|l| !l.trim().is_empty() && !l.starts_with("=== "))
        .is_some_and(|l| l.starts_with("fetched ") || l.starts_with("Error: "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(id: &str) -> String {
        format!(
            r#"{{"id":"{id}","file":"{id}.jpg","created":"2020-01-01T00:00:00Z","width":4,"height":3,"favorite":false,"burst_id":null,"subtypes":[]}}"#
        )
    }

    #[test]
    fn torn_tail_and_duplicates() {
        let dir = std::env::temp_dir().join(format!("pe-manifest-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let m = dir.join("manifest.jsonl");
        fs::write(
            &m,
            format!("{}\n{}\n{}\n{{\"id\":\"x", line("a"), line("b"), line("a")),
        )
        .unwrap();
        let v = read(&dir).unwrap();
        assert_eq!(
            v.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        fs::write(&m, format!("{}\nnot json\n{}\n", line("a"), line("b"))).unwrap();
        assert!(read(&dir).is_err());
        let log = dir.join("fetch.log");
        fs::write(&log, "  1 abc\n").unwrap();
        assert!(!fetch_finished(&log));
        fs::write(&log, "  1 abc\nfetched 1 (from iCloud 0), skipped 0\n\n").unwrap();
        assert!(fetch_finished(&log));
        fs::write(
            &log,
            "fetched 1 videos (8 frames)\n=== 12:00:00 DONE exit=0\n",
        )
        .unwrap();
        assert!(fetch_finished(&log));
        fs::write(
            &log,
            "  1 abc\nError: stopped early\n=== 12:00:00 DONE exit=1\n",
        )
        .unwrap();
        assert!(fetch_finished(&log));
        fs::write(&log, "  1 abc\n=== 12:00:00 DONE\n").unwrap();
        assert!(!fetch_finished(&log));
        fs::remove_dir_all(&dir).unwrap();
    }
}
