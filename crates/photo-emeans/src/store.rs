//! The embedding store: `embed/embeddings.f32`, `embed/rows.jsonl`, `embed/meta.json`.
//!
//! Crash safety rests on one write order. Each chunk appends its f32 rows and fsyncs, then appends
//! the matching `rows.jsonl` lines and fsyncs, then rewrites `meta.json` atomically. A crash can
//! therefore leave (a) a torn last line in `rows.jsonl`, (b) more f32 rows than lines, or (c) a
//! partial f32 row. [`EmbedStore::open`] repairs all three by truncating to the shorter file at a
//! whole-row boundary and says so loudly. Anything else (a bad line in the middle, a row number
//! out of order, a duplicate id, a non-unit vector, a different model) stops the run: those are
//! not crash artifacts, and guessing would corrupt the data.
//!
//! Readers ([`load`]) take the complete `rows.jsonl` lines as the truth and need at least that many
//! f32 rows, so they can read a consistent prefix while an embed run is still appending.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::util::{now_iso, sync_dir, write_atomic};

/// Embedding width (EmbeddingGemma 2).
pub const D: usize = encoder::families::embeddinggemma2::EMBED_DIM;
const ROW_BYTES: u64 = (D * 4) as u64;
pub const EMBEDDINGS: &str = "embeddings.f32";
pub const ROWS: &str = "rows.jsonl";
pub const META: &str = "meta.json";
pub const FAILURES: &str = "failures.jsonl";
const LOCK: &str = ".lock";
/// How far a stored row's L2 norm may be from 1.
const NORM_TOL: f32 = 1e-3;

/// `embed/meta.json`. `dtype` is the storage type (always `"f32"`); `compute_dtype` is the dtype
/// the model ran in. A run refuses to append to a store made by another model, revision or
/// compute dtype, because mixed vectors would cluster as if they were comparable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub model: String,
    pub revision: String,
    pub d: usize,
    pub n: usize,
    pub dtype: String,
    #[serde(default)]
    pub compute_dtype: Option<String>,
    /// Prompt prepended to images (`null`: none, as the model card says).
    #[serde(default)]
    pub prompt: Option<String>,
    pub created: String,
    #[serde(default)]
    pub updated: Option<String>,
}

impl Meta {
    pub fn new(model: &str, revision: &str, compute_dtype: &str) -> Self {
        Self {
            model: model.into(),
            revision: revision.into(),
            d: D,
            n: 0,
            dtype: "f32".into(),
            compute_dtype: Some(compute_dtype.into()),
            prompt: None,
            created: now_iso(),
            updated: None,
        }
    }

    fn check_same_source(&self, want: &Meta) -> Result<()> {
        ensure!(
            self.model == want.model
                && self.revision == want.revision
                && self.d == want.d
                && self.dtype == want.dtype
                && self.compute_dtype == want.compute_dtype
                && self.prompt == want.prompt,
            "embed/meta.json was made by {} @ {} ({:?}, prompt {:?}), this run is {} @ {} ({:?}, prompt {:?}); \
             refusing to mix vectors. Use the same --dtype, or move embed/ aside to start over",
            self.model,
            self.revision,
            self.compute_dtype,
            self.prompt,
            want.model,
            want.revision,
            want.compute_dtype,
            want.prompt
        );
        Ok(())
    }
}

/// What a row's vector stands for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// A photo (`photos/manifest.jsonl`). Lines without `kind` (written before videos) are photos.
    #[default]
    Photo,
    /// A video (`videos/manifest.jsonl`): the L2-normalized mean of its frame vectors.
    Video,
    /// One video frame, in the frame store `embed/frames/` (id `<asset id>#fNN`).
    Frame,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Photo => "photo",
            Kind::Video => "video",
            Kind::Frame => "frame",
        }
    }
}

/// One line of `rows.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowLine {
    pub row: usize,
    pub id: String,
    #[serde(default)]
    pub kind: Kind,
}

/// What [`EmbedStore::open`] had to repair.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Repair {
    /// A last `rows.jsonl` line without its newline was dropped.
    pub torn_line: bool,
    /// Complete `rows.jsonl` lines dropped because their f32 row was missing.
    pub lines_dropped: usize,
    /// Whole f32 rows dropped because their `rows.jsonl` line was missing.
    pub rows_dropped: usize,
    /// Bytes of a partial f32 row dropped.
    pub partial_bytes: u64,
}

impl Repair {
    pub fn any(&self) -> bool {
        self.torn_line || self.lines_dropped > 0 || self.rows_dropped > 0 || self.partial_bytes > 0
    }
}

/// Parsed `rows.jsonl`: complete lines, byte offset where each line starts, and torn-tail flag.
struct RowsFile {
    lines: Vec<RowLine>,
    starts: Vec<u64>,
    complete_len: u64,
    torn: bool,
}

fn parse_rows(path: &Path) -> Result<RowsFile> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let end = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let torn = end < bytes.len();
    let text = std::str::from_utf8(&bytes[..end])
        .with_context(|| format!("{} is not UTF-8", path.display()))?;
    let mut lines = Vec::new();
    let mut starts = Vec::new();
    let mut ids = HashSet::new();
    let mut off = 0u64;
    for (i, l) in text.split_inclusive('\n').enumerate() {
        let line: RowLine = serde_json::from_str(l.trim_end()).with_context(|| {
            format!(
                "{} line {}: not a row line; refusing to continue on a corrupt rows file",
                path.display(),
                i + 1
            )
        })?;
        ensure!(
            line.row == i,
            "{} line {} says row {}; rows must be numbered 0, 1, 2, ...",
            path.display(),
            i + 1,
            line.row
        );
        ensure!(
            ids.insert(line.id.clone()),
            "{} line {}: id {} appears twice",
            path.display(),
            i + 1,
            line.id
        );
        starts.push(off);
        off += l.len() as u64;
        lines.push(line);
    }
    Ok(RowsFile {
        lines,
        starts,
        complete_len: end as u64,
        torn,
    })
}

fn read_rows_f32(path: &Path, n: usize) -> Result<Vec<f32>> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut buf = vec![0u8; n * D * 4];
    f.read_exact(&mut buf)
        .with_context(|| format!("reading {n} rows of {}", path.display()))?;
    Ok(buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn check_unit_rows(x: &[f32], ids: &[RowLine]) -> Result<()> {
    for (i, r) in x.as_chunks::<D>().0.iter().enumerate() {
        let nrm = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        ensure!(
            r.iter().all(|v| v.is_finite()) && (nrm - 1.0).abs() <= NORM_TOL,
            "embeddings.f32 row {i} (id {}) has norm {nrm}; the file is corrupt, not a crash artifact",
            ids[i].id
        );
    }
    Ok(())
}

/// The writer side of the embedding store. Holds an exclusive lock on `embed/.lock`.
pub struct EmbedStore {
    dir: PathBuf,
    f32_file: File,
    rows_file: File,
    meta: Meta,
    ids: Vec<String>,
    id_set: HashSet<String>,
    kinds: Vec<Kind>,
    poisoned: bool,
    _lock: File,
}

impl EmbedStore {
    /// Open (or create) the store in `dir`, reconcile it, and lock it against a second writer.
    /// `want` names the model, revision and compute dtype of this run; its `n` is ignored.
    pub fn open(dir: &Path, want: Meta, log: &mut dyn FnMut(&str)) -> Result<(Self, Repair)> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK))
            .context("opening embed/.lock")?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                bail!(
                    "another embed run holds {}; only one may write",
                    dir.join(LOCK).display()
                )
            }
            Err(fs::TryLockError::Error(e)) => return Err(e).context("locking embed/.lock"),
        }

        let rows_path = dir.join(ROWS);
        let f32_path = dir.join(EMBEDDINGS);
        let meta_path = dir.join(META);
        let rows = parse_rows(&rows_path)?;
        let f32_len = match fs::metadata(&f32_path) {
            Ok(m) => m.len(),
            Err(e) if e.kind() == ErrorKind::NotFound => 0,
            Err(e) => return Err(e).with_context(|| format!("stat {}", f32_path.display())),
        };
        let full = (f32_len / ROW_BYTES) as usize;
        let n = rows.lines.len().min(full);
        let repair = Repair {
            torn_line: rows.torn,
            lines_dropped: rows.lines.len() - n,
            rows_dropped: full - n,
            partial_bytes: f32_len % ROW_BYTES,
        };

        let old_meta: Option<Meta> = match fs::read(&meta_path) {
            Ok(b) => Some(serde_json::from_slice(&b).context("parsing embed/meta.json")?),
            Err(e) if e.kind() == ErrorKind::NotFound => None,
            Err(e) => return Err(e).context("reading embed/meta.json"),
        };
        let meta = match old_meta {
            Some(m) => {
                m.check_same_source(&want)?;
                m
            }
            None => {
                ensure!(
                    rows.lines.is_empty() && f32_len == 0,
                    "{} has data but no meta.json, so the model that made it is unknown; \
                     refusing to append. Move embed/ aside to start over",
                    dir.display()
                );
                want
            }
        };

        if repair.any() {
            log(&format!(
                "!!! embed store repair in {}: rows.jsonl had {} complete lines{}, embeddings.f32 had {} whole rows + {} stray bytes; \
                 truncating both to {n} rows (dropping {} lines, {} f32 rows). This is expected after a crash or kill.",
                dir.display(),
                rows.lines.len(),
                if rows.torn { " + a torn last line" } else { "" },
                full,
                repair.partial_bytes,
                repair.lines_dropped,
                repair.rows_dropped,
            ));
            let rows_cut = if n < rows.lines.len() {
                rows.starts[n]
            } else {
                rows.complete_len
            };
            let f = OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&rows_path)
                .context("opening rows.jsonl to repair")?;
            f.set_len(rows_cut)?;
            f.sync_all()?;
            let f = OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&f32_path)
                .context("opening embeddings.f32 to repair")?;
            f.set_len(n as u64 * ROW_BYTES)?;
            f.sync_all()?;
        }

        let lines = &rows.lines[..n];
        if n > 0 {
            check_unit_rows(&read_rows_f32(&f32_path, n)?, lines)?;
        }
        let open_append = |p: &Path| -> Result<File> {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .with_context(|| format!("opening {}", p.display()))
        };
        let f32_file = open_append(&f32_path)?;
        let rows_file = open_append(&rows_path)?;
        sync_dir(dir)?;
        let ids: Vec<String> = lines.iter().map(|l| l.id.clone()).collect();
        let mut store = Self {
            dir: dir.to_path_buf(),
            f32_file,
            rows_file,
            id_set: ids.iter().cloned().collect(),
            kinds: lines.iter().map(|l| l.kind).collect(),
            ids,
            meta,
            poisoned: false,
            _lock: lock,
        };
        store.write_meta()?;
        Ok((store, repair))
    }

    pub fn n(&self) -> usize {
        self.ids.len()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.id_set.contains(id)
    }

    /// Rows of each kind.
    pub fn count(&self, kind: Kind) -> usize {
        self.kinds.iter().filter(|&&k| k == kind).count()
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn write_meta(&mut self) -> Result<()> {
        self.meta.n = self.ids.len();
        self.meta.updated = Some(now_iso());
        let mut s = serde_json::to_string_pretty(&self.meta)?;
        s.push('\n');
        write_atomic(&self.dir.join(META), s.as_bytes())
    }

    /// Append rows of one `kind`: f32 data + fsync, then the `rows.jsonl` lines + fsync, then
    /// `meta.json`. Every vector must be finite and unit length. After any error the store refuses
    /// more writes.
    pub fn append(&mut self, rows: &[(String, [f32; D])], kind: Kind) -> Result<()> {
        ensure!(
            !self.poisoned,
            "embed store refused: an earlier write failed"
        );
        if rows.is_empty() {
            return Ok(());
        }
        let mut batch_ids = HashSet::new();
        for (id, v) in rows {
            ensure!(
                !self.id_set.contains(id) && batch_ids.insert(id.as_str()),
                "id {id} is already embedded"
            );
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            ensure!(
                v.iter().all(|x| x.is_finite()) && (nrm - 1.0).abs() <= NORM_TOL,
                "vector for {id} has norm {nrm}; refusing to store it"
            );
        }
        self.poisoned = true;
        let mut buf = Vec::with_capacity(rows.len() * D * 4);
        for (_, v) in rows {
            for x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        self.f32_file
            .write_all(&buf)
            .context("appending embeddings.f32")?;
        self.f32_file.sync_data().context("fsync embeddings.f32")?;
        let mut text = String::new();
        for (i, (id, _)) in rows.iter().enumerate() {
            text.push_str(&serde_json::to_string(&RowLine {
                row: self.ids.len() + i,
                id: id.clone(),
                kind,
            })?);
            text.push('\n');
        }
        self.rows_file
            .write_all(text.as_bytes())
            .context("appending rows.jsonl")?;
        self.rows_file.sync_data().context("fsync rows.jsonl")?;
        for (id, _) in rows {
            self.ids.push(id.clone());
            self.id_set.insert(id.clone());
            self.kinds.push(kind);
        }
        self.write_meta()?;
        self.poisoned = false;
        Ok(())
    }
}

/// A consistent snapshot of the store for readers.
pub struct Embeddings {
    pub meta: Meta,
    pub ids: Vec<String>,
    /// Kind of each row (`photo` for lines written before `kind` existed).
    pub kinds: Vec<Kind>,
    /// `n x D`, row-major.
    pub x: Vec<f32>,
}

impl Embeddings {
    pub fn n(&self) -> usize {
        self.ids.len()
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.x[i * D..(i + 1) * D]
    }
}

/// Read the store without repairing it. The complete `rows.jsonl` lines decide `n`; the f32 file
/// must hold at least `n` rows (it is written first, so it may hold more while an embed runs).
pub fn load(dir: &Path) -> Result<Embeddings> {
    let meta: Meta = serde_json::from_slice(
        &fs::read(dir.join(META)).with_context(|| format!("reading {}/{META}", dir.display()))?,
    )
    .context("parsing embed/meta.json")?;
    ensure!(
        meta.d == D && meta.dtype == "f32",
        "unexpected meta {meta:?}"
    );
    let rows = parse_rows(&dir.join(ROWS))?;
    let n = rows.lines.len();
    let f32_len = fs::metadata(dir.join(EMBEDDINGS)).map_or(0, |m| m.len());
    ensure!(
        f32_len >= n as u64 * ROW_BYTES,
        "embed store is misaligned: rows.jsonl has {n} lines but embeddings.f32 holds only {} rows; \
         run `photo-emeans embed` once to reconcile it",
        f32_len / ROW_BYTES
    );
    let x = read_rows_f32(&dir.join(EMBEDDINGS), n)?;
    check_unit_rows(&x, &rows.lines)?;
    Ok(Embeddings {
        meta,
        kinds: rows.lines.iter().map(|l| l.kind).collect(),
        ids: rows.lines.into_iter().map(|l| l.id).collect(),
        x,
    })
}

/// One line of `embed/failures.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub id: String,
    pub file: String,
    pub error: String,
    pub at: String,
}

/// Ids listed in `failures.jsonl` (complete lines only).
pub fn read_failures(dir: &Path) -> Result<Vec<Failure>> {
    let text = match fs::read_to_string(dir.join(FAILURES)) {
        Ok(t) => t,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).context("reading failures.jsonl"),
    };
    let complete = text.rfind('\n').map_or("", |i| &text[..=i]);
    complete
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("parsing failures.jsonl"))
        .collect()
}

/// Append one failure line and fsync it.
pub fn append_failure(dir: &Path, f: &Failure) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(FAILURES))
        .context("opening failures.jsonl")?;
    let mut s = serde_json::to_string(f)?;
    s.push('\n');
    file.write_all(s.as_bytes())?;
    file.sync_data()?;
    Ok(())
}
