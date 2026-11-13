//! The `cluster` and `add` steps: read `embed/`, optionally group near-duplicates, cluster with
//! plain spherical e-means (`--k`) or the two-level automatic hierarchy (`--auto`), and write
//! `clusters/<run>/` (see AGENTS.md, data contract).
//!
//! A run directory is built as `clusters/.<run>.tmp` and renamed into place, so a crash never
//! leaves a half-written run. `add` builds the new state the same way and swaps it in.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use emeans::{
    AutoConfig, Backend, Config, DedupConfig, DedupInput, DedupResult, Emeans, Hierarchy,
    auto_cluster, dedup, make_assigner,
};
use serde::{Deserialize, Serialize};

use crate::manifest::{self, Item};
use crate::store::{self, D, Embeddings, Kind, Meta};
use crate::util::{now_iso, parse_iso8601, sync_dir, write_atomic};

pub const REPORT: &str = "report.json";
pub const LABELS: &str = "labels.i32";
pub const CENTROIDS: &str = "centroids.f32";
pub const LABELS_L2: &str = "labels_l2.i32";
pub const CENTROIDS_L2: &str = "centroids_l2.f32";
pub const SIM: &str = "sim.f32";
pub const OUTLIER: &str = "outlier.u8";
pub const DEDUP_REP: &str = "dedup.i32";
pub const DEDUP_STATE: &str = "dedup.json";
pub const MODEL_DIR: &str = "model";
pub const MODEL_FILE: &str = "model.emeans";

#[derive(Debug, Clone)]
pub enum Mode {
    /// Plain spherical e-means with this k.
    K(usize),
    /// Two-level hierarchy with automatic k.
    Auto(Box<AutoConfig>),
}

#[derive(Debug, Clone)]
pub struct ClusterOpts {
    pub run: String,
    pub mode: Mode,
    /// Assignment backend for e-means (and the dedup kNN).
    pub backend: Backend,
    /// Group near-duplicates first and cluster one representative per group.
    pub dedup: Option<DedupConfig>,
    /// `--k` only.
    pub nepochs: f64,
    pub seed: u32,
    /// Replace an existing run directory.
    pub force: bool,
}

// ------------------------------------------------------------------ report

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub index: usize,
    pub size: usize,
    pub mean_sim: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupInfo {
    pub index: usize,
    /// Stable level-1 id (survives `add`).
    pub id: u32,
    pub size: usize,
    pub mean_sim: f32,
    pub decision: serde_json::Value,
    /// Indices into `centroids_l2.f32` / `labels_l2.i32`.
    pub leaves: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafInfo {
    pub index: usize,
    /// Level-1 index (row of `centroids.f32`).
    pub group: usize,
    pub l1_id: u32,
    pub l2_id: u32,
    pub size: usize,
    pub mean_sim: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoReport {
    pub config: serde_json::Value,
    pub chosen_k: serde_json::Value,
    pub groups: Vec<GroupInfo>,
    pub leaves: Vec<LeafInfo>,
    pub diagnostics: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupReport {
    pub config: DedupConfig,
    pub groups: usize,
    /// Photos in groups of two or more.
    pub photos_in_stacks: usize,
    pub largest_group: usize,
    pub linked_pairs: usize,
    pub split_components: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timings {
    pub load_s: f64,
    pub dedup_s: f64,
    pub cluster_s: f64,
    pub write_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    /// Mean `2 - 2 cos` of every photo to its own (leaf) centroid.
    pub objective: f64,
    pub mean_sim: f64,
    pub min_size: usize,
    pub median_size: usize,
    pub max_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub at: String,
    pub action: String,
    pub n_before: usize,
    pub n_after: usize,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub run: String,
    pub created: String,
    /// `"k"` or `"auto"`.
    pub mode: String,
    /// Photos and videos covered: the first `n` rows of `embed/`.
    pub n: usize,
    /// How many of the `n` rows are videos.
    #[serde(default)]
    pub n_videos: usize,
    pub d: usize,
    /// FNV-1a 64 of the first `n` embed ids joined by newlines; readers check it.
    pub ids_fnv: String,
    pub embed: Meta,
    pub backend: String,
    /// Rows the clustering saw (dedup representatives, or `n`).
    pub clustered_rows: usize,
    /// Rows of `centroids.f32`.
    pub k: usize,
    /// Rows of `centroids_l2.f32` (auto only).
    pub k_leaves: Option<usize>,
    pub stats: Stats,
    pub timings: Timings,
    pub recipe: Option<serde_json::Value>,
    pub iteration_stats: Option<serde_json::Value>,
    pub dedup: Option<DedupReport>,
    /// Rows of `centroids.f32` (level-1 groups for auto).
    pub clusters: Vec<ClusterInfo>,
    pub auto: Option<AutoReport>,
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
}

pub fn ids_fnv(ids: &[String]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            h = (h ^ u64::from(b'\n')).wrapping_mul(0x100_0000_01b3);
        }
        for b in id.bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

// ------------------------------------------------------------------ helpers

pub fn backend_name(b: Backend) -> String {
    match b {
        Backend::Cpu(k) => format!("cpu-{k:?}").to_lowercase(),
        Backend::MetalFused => "metal-fused".into(),
        other => format!("{other:?}").to_lowercase(),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn read_i32(path: &Path) -> Result<Vec<i32>> {
    let b = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        b.len() % 4 == 0,
        "{} is not a whole number of i32",
        path.display()
    );
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect())
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        b.len() % 4 == 0,
        "{} is not a whole number of f32",
        path.display()
    );
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Every fetched photo and video by asset id (`photos/` and `videos/` manifests).
pub fn manifest_by_id(data: &Path) -> Result<HashMap<String, Item>> {
    manifest::media_by_id(data)
}

/// Per-row side information for dedup: creation time, interned burst id, favorite priority.
/// Videos have no burst id.
pub struct SideInfo {
    pub created: Vec<f64>,
    pub burst: Vec<Option<u32>>,
    pub priority: Vec<f32>,
}

impl SideInfo {
    pub fn new(ids: &[String], man: &HashMap<String, Item>) -> Self {
        let mut intern: HashMap<&str, u32> = HashMap::new();
        let mut s = SideInfo {
            created: Vec::with_capacity(ids.len()),
            burst: Vec::with_capacity(ids.len()),
            priority: Vec::with_capacity(ids.len()),
        };
        for id in ids {
            let l = man.get(id);
            s.created.push(
                l.and_then(|l| l.created.as_deref())
                    .and_then(parse_iso8601)
                    .unwrap_or(f64::NAN),
            );
            s.burst.push(l.and_then(|l| l.burst_id.as_deref()).map(|b| {
                let next = intern.len() as u32;
                *intern.entry(b).or_insert(next)
            }));
            s.priority.push(if l.is_some_and(|l| l.favorite) {
                1.0
            } else {
                0.0
            });
        }
        s
    }

    pub fn input(&self) -> DedupInput<'_> {
        DedupInput {
            created_unix: Some(&self.created),
            burst: Some(&self.burst),
            priority: Some(&self.priority),
        }
    }
}

fn dedup_report(dd: &DedupResult, cfg: &DedupConfig) -> DedupReport {
    DedupReport {
        config: cfg.clone(),
        groups: dd.n_groups(),
        photos_in_stacks: dd
            .sizes
            .iter()
            .filter(|&&s| s > 1)
            .map(|&s| s as usize)
            .sum(),
        largest_group: dd.sizes.iter().copied().max().unwrap_or(0) as usize,
        linked_pairs: dd.linked_pairs,
        split_components: dd.split_components,
    }
}

fn size_stats(sizes: &[usize], sims: &[f32]) -> Stats {
    let mut s: Vec<usize> = sizes.iter().copied().filter(|&x| x > 0).collect();
    s.sort_unstable();
    let n = sims.len().max(1) as f64;
    Stats {
        objective: sims.iter().map(|&x| 2.0 - 2.0 * f64::from(x)).sum::<f64>() / n,
        mean_sim: sims.iter().map(|&x| f64::from(x)).sum::<f64>() / n,
        min_size: s.first().copied().unwrap_or(0),
        median_size: s.get(s.len() / 2).copied().unwrap_or(0),
        max_size: s.last().copied().unwrap_or(0),
    }
}

/// Size and mean similarity of each label value `0..k`.
fn per_label(labels: &[i32], sims: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut acc = vec![(0usize, 0f64); k];
    for (&l, &s) in labels.iter().zip(sims) {
        if l >= 0 {
            acc[l as usize].0 += 1;
            acc[l as usize].1 += f64::from(s);
        }
    }
    acc.into_iter()
        .map(|(c, s)| (c, if c > 0 { (s / c as f64) as f32 } else { 0.0 }))
        .collect()
}

/// Everything a hierarchy run writes, in index form.
pub struct HierOut {
    pub labels1: Vec<i32>,
    pub labels2: Vec<i32>,
    pub sim: Vec<f32>,
    pub centroids1: Vec<f32>,
    pub centroids2: Vec<f32>,
    pub groups: Vec<GroupInfo>,
    pub leaves: Vec<LeafInfo>,
}

/// Turn per-photo stable leaf ids into index files. `sim` is each photo's cosine to its leaf
/// centroid (computed here for every photo, also for photos behind a dedup representative).
pub fn hierarchy_outputs(h: &Hierarchy, leaf_ids: &[(u32, u32)], x: &[f32]) -> Result<HierOut> {
    ensure!(x.len() == leaf_ids.len() * D, "one leaf per photo");
    let mut leaves = Vec::new();
    let mut centroids2 = Vec::new();
    let mut index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut slot_of_l1: HashMap<u32, usize> = HashMap::new();
    for (g, grp) in h.groups.iter().enumerate() {
        let l1_id = h.ids1[g];
        slot_of_l1.insert(l1_id, g);
        let c = h.leaf_centroids(g);
        let k2 = c.len() / D;
        for s in 0..k2 {
            // An empty group has no model and no level-2 ids; it shows as one empty leaf.
            let l2_id = grp.ids.get(s).copied().unwrap_or(u32::MAX);
            index.insert((l1_id, l2_id), leaves.len());
            leaves.push(LeafInfo {
                index: leaves.len(),
                group: g,
                l1_id,
                l2_id,
                size: 0,
                mean_sim: 0.0,
            });
        }
        centroids2.extend_from_slice(&c);
    }
    let mut labels1 = Vec::with_capacity(leaf_ids.len());
    let mut labels2 = Vec::with_capacity(leaf_ids.len());
    let mut sim = Vec::with_capacity(leaf_ids.len());
    for (i, &(a, b)) in leaf_ids.iter().enumerate() {
        let leaf = *index.get(&(a, b)).with_context(|| {
            format!("photo row {i} has leaf ({a}, {b}), which the model does not have")
        })?;
        let g = slot_of_l1[&a];
        labels1.push(g as i32);
        labels2.push(leaf as i32);
        sim.push(dot(
            &x[i * D..(i + 1) * D],
            &centroids2[leaf * D..(leaf + 1) * D],
        ));
    }
    for (l, (size, ms)) in per_label(&labels2, &sim, leaves.len())
        .into_iter()
        .enumerate()
    {
        leaves[l].size = size;
        leaves[l].mean_sim = ms;
    }
    let by_group = per_label(&labels1, &sim, h.groups.len());
    let groups = h
        .groups
        .iter()
        .enumerate()
        .map(|(g, grp)| GroupInfo {
            index: g,
            id: h.ids1[g],
            size: by_group[g].0,
            mean_sim: by_group[g].1,
            decision: serde_json::to_value(&grp.decision).unwrap_or_default(),
            leaves: leaves
                .iter()
                .filter(|l| l.group == g)
                .map(|l| l.index)
                .collect(),
        })
        .collect();
    Ok(HierOut {
        labels1,
        labels2,
        sim,
        centroids1: h.level1_centroids().to_vec(),
        centroids2,
        groups,
        leaves,
    })
}

/// Write the files of a run into a fresh temporary directory and swap it into `run_dir`.
/// `extra` writes more files (models) into the temporary directory.
/// `remove_dir_all` with a few retries. On macOS it can fail with "Directory not empty" while
/// another process (Spotlight indexing a gallery's thumbnails) adds or holds entries.
fn remove_dir_retry(path: &Path) -> std::io::Result<()> {
    let mut last = None;
    for attempt in 0..4 {
        match fs::remove_dir_all(path) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => last = Some(e),
        }
        std::thread::sleep(std::time::Duration::from_millis(250 << attempt));
    }
    Err(last.expect("an error after failed attempts"))
}

/// Returns a warning when the new run is in place but the previous version could not be deleted
/// (it stays as `.<run>.old` and the next commit deletes it first).
fn commit_run_dir(
    run_dir: &Path,
    replace: bool,
    files: &[(&str, Vec<u8>)],
    extra: impl FnOnce(&Path) -> Result<()>,
) -> Result<Option<String>> {
    let parent = run_dir.parent().context("run dir has no parent")?;
    fs::create_dir_all(parent)?;
    let name = run_dir
        .file_name()
        .context("run dir has no name")?
        .to_string_lossy();
    let tmp = parent.join(format!(".{name}.tmp"));
    let old = parent.join(format!(".{name}.old"));
    remove_dir_retry(&tmp).with_context(|| format!("removing {}", tmp.display()))?;
    remove_dir_retry(&old).with_context(|| {
        format!(
            "removing the previous version {} left by an earlier run",
            old.display()
        )
    })?;
    fs::create_dir_all(&tmp)?;
    for (f, data) in files {
        let p = tmp.join(f);
        fs::write(&p, data).with_context(|| format!("writing {}", p.display()))?;
        fs::File::open(&p)?.sync_all()?;
    }
    extra(&tmp)?;
    sync_dir(&tmp)?;
    if run_dir.exists() {
        ensure!(
            replace,
            "{} exists; pass --force to replace it",
            run_dir.display()
        );
        fs::rename(run_dir, &old)?;
    }
    fs::rename(&tmp, run_dir)?;
    sync_dir(parent)?;
    // The new run is committed. Failing to delete the old version must not report the run as
    // failed: say so, and leave it for the next commit to delete.
    Ok(remove_dir_retry(&old).err().map(|e| {
        format!(
            "warning: the new run is in place, but the previous version {} could not be deleted \
             ({e}); delete it by hand or let the next run delete it",
            old.display()
        )
    }))
}

// ------------------------------------------------------------------ cluster

/// Writes the trained model into the run's temporary directory.
type SaveModel = Box<dyn FnOnce(&Path) -> Result<()>>;

/// Cluster the current embeddings into `clusters/<run>/`. Returns the run directory.
pub fn cluster(data: &Path, opts: &ClusterOpts, log: &mut dyn FnMut(&str)) -> Result<PathBuf> {
    ensure!(
        !opts.run.is_empty() && !opts.run.contains(['/', '\\']) && !opts.run.starts_with('.'),
        "bad run name {:?}",
        opts.run
    );
    let run_dir = data.join("clusters").join(&opts.run);
    ensure!(
        opts.force || !run_dir.exists(),
        "{} exists; pass --force to replace it",
        run_dir.display()
    );
    let t0 = Instant::now();
    let emb = store::load(&data.join("embed"))?;
    let n = emb.n();
    ensure!(n >= 2, "only {n} embedded rows; run embed first");
    let load_s = t0.elapsed().as_secs_f64();
    log(&format!("loaded {n} embeddings in {load_s:.1}s"));

    let t = Instant::now();
    let (dd, xr) = match &opts.dedup {
        Some(cfg) => {
            let man = manifest_by_id(data)?;
            let side = SideInfo::new(&emb.ids, &man);
            let dd = dedup(&emb.x, D, &side.input(), cfg, opts.backend)?;
            let xr = dd.gather(&emb.x, D);
            log(&format!(
                "dedup: {n} rows -> {} groups ({} rows in stacks, largest {}), {:.1}s",
                dd.n_groups(),
                dedup_report(&dd, cfg).photos_in_stacks,
                dedup_report(&dd, cfg).largest_group,
                t.elapsed().as_secs_f64()
            ));
            (Some(dd), xr)
        }
        None => (None, emb.x.clone()),
    };
    let dedup_s = if dd.is_some() {
        t.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let nr = xr.len() / D;
    let expand = |rep_vals: &[(u32, u32)]| -> Vec<(u32, u32)> {
        match &dd {
            Some(dd) => dd.expand(rep_vals),
            None => rep_vals.to_vec(),
        }
    };

    let t = Instant::now();
    let base = |mode: &str| Report {
        run: opts.run.clone(),
        created: now_iso(),
        mode: mode.into(),
        n,
        n_videos: emb.kinds.iter().filter(|&&k| k == Kind::Video).count(),
        d: D,
        ids_fnv: ids_fnv(&emb.ids),
        embed: emb.meta.clone(),
        backend: backend_name(opts.backend),
        clustered_rows: nr,
        k: 0,
        k_leaves: None,
        stats: size_stats(&[], &[]),
        timings: Timings {
            load_s,
            dedup_s,
            cluster_s: 0.0,
            write_s: 0.0,
        },
        recipe: None,
        iteration_stats: None,
        dedup: dd
            .as_ref()
            .zip(opts.dedup.as_ref())
            .map(|(dd, c)| dedup_report(dd, c)),
        clusters: Vec::new(),
        auto: None,
        history: Vec::new(),
    };
    let mut files: Vec<(&str, Vec<u8>)> = Vec::new();
    let (mut report, model): (Report, SaveModel) = match &opts.mode {
        Mode::K(k) => {
            ensure!(*k >= 1 && *k <= nr, "k = {k} but only {nr} rows to cluster");
            let mut c = Config::new(D, *k);
            c.spherical = true;
            c.nepochs = opts.nepochs;
            c.seed = opts.seed;
            c.capture_assignments = false;
            let mut m = Emeans::with_assigner(c, make_assigner(opts.backend)?)?;
            m.train(&xr)?;
            let a = m.assign(&xr)?;
            let cluster_s = t.elapsed().as_secs_f64();
            log(&format!("e-means k={k} on {nr} rows: {cluster_s:.1}s"));
            let cents = m
                .centroids()
                .context("trained model has no centroids")?
                .to_vec();
            let per_photo: Vec<u32> = match &dd {
                Some(dd) => dd.expand(&a.labels),
                None => a.labels.clone(),
            };
            let labels: Vec<i32> = per_photo.iter().map(|&l| l as i32).collect();
            let sim: Vec<f32> = (0..n)
                .map(|i| {
                    let l = per_photo[i] as usize;
                    dot(emb.row(i), &cents[l * D..(l + 1) * D])
                })
                .collect();
            let pl = per_label(&labels, &sim, *k);
            let mut r = base("k");
            r.k = *k;
            r.timings.cluster_s = cluster_s;
            r.stats = size_stats(&pl.iter().map(|p| p.0).collect::<Vec<_>>(), &sim);
            r.recipe = m.recipe().map(serde_json::to_value).transpose()?;
            r.iteration_stats = Some(serde_json::to_value(m.iteration_stats())?);
            r.clusters = pl
                .iter()
                .enumerate()
                .map(|(i, &(size, mean_sim))| ClusterInfo {
                    index: i,
                    size,
                    mean_sim,
                })
                .collect();
            files.push((LABELS, bytes_i32(&labels)));
            files.push((CENTROIDS, bytes_f32(&cents)));
            files.push((SIM, bytes_f32(&sim)));
            (r, Box::new(move |dir: &Path| m.save(dir.join(MODEL_FILE))))
        }
        Mode::Auto(cfg) => {
            let mut cfg = (**cfg).clone();
            cfg.backend = opts.backend;
            let (mut h, labels) = auto_cluster(&xr, D, &cfg)?;
            h.dedup = opts.dedup.clone();
            let cluster_s = t.elapsed().as_secs_f64();
            let ck = h.chosen_k();
            log(&format!(
                "auto: k1 = {}, leaves = {}, on {nr} rows: {cluster_s:.1}s",
                ck.k1,
                ck.k2.iter().map(|&k| k.max(1)).sum::<usize>()
            ));
            let rep_leaf: Vec<(u32, u32)> = (0..labels.len()).map(|i| labels.leaf(i)).collect();
            let rep_out: Vec<u8> = labels.outlier.iter().map(|&o| u8::from(o)).collect();
            let leaf_ids = expand(&rep_leaf);
            let outlier = match &dd {
                Some(dd) => dd.expand(&rep_out),
                None => rep_out,
            };
            let mut r = base("auto");
            let out = hierarchy_outputs(&h, &leaf_ids, &emb.x)?;
            fill_auto_report(&mut r, &h, &out)?;
            r.timings.cluster_s = cluster_s;
            files.push((LABELS, bytes_i32(&out.labels1)));
            files.push((CENTROIDS, bytes_f32(&out.centroids1)));
            files.push((LABELS_L2, bytes_i32(&out.labels2)));
            files.push((CENTROIDS_L2, bytes_f32(&out.centroids2)));
            files.push((SIM, bytes_f32(&out.sim)));
            files.push((OUTLIER, outlier));
            (r, Box::new(move |dir: &Path| h.save(dir.join(MODEL_DIR))))
        }
    };
    if let Some(dd) = &dd {
        let rep_row: Vec<i32> = dd
            .group_of
            .iter()
            .map(|&g| dd.representatives[g as usize] as i32)
            .collect();
        files.push((DEDUP_REP, bytes_i32(&rep_row)));
        files.push((DEDUP_STATE, serde_json::to_vec(dd)?));
    }
    let t = Instant::now();
    report.timings.write_s = 0.0;
    files.push((REPORT, serde_json::to_vec_pretty(&report)?));
    if let Some(w) = commit_run_dir(&run_dir, opts.force, &files, model)? {
        log(&w);
    }
    // Record the write time too (the report in the directory is rewritten atomically).
    report.timings.write_s = t.elapsed().as_secs_f64();
    write_atomic(&run_dir.join(REPORT), &serde_json::to_vec_pretty(&report)?)?;
    log(&format!(
        "wrote {} ({} clusters{}, objective {:.4}, mean sim {:.4})",
        run_dir.display(),
        report.k,
        report
            .k_leaves
            .map(|l| format!(", {l} leaves"))
            .unwrap_or_default(),
        report.stats.objective,
        report.stats.mean_sim
    ));
    Ok(run_dir)
}

fn fill_auto_report(r: &mut Report, h: &Hierarchy, out: &HierOut) -> Result<()> {
    r.k = h.groups.len();
    r.k_leaves = Some(out.leaves.len());
    r.stats = size_stats(
        &out.leaves.iter().map(|l| l.size).collect::<Vec<_>>(),
        &out.sim,
    );
    r.clusters = out
        .groups
        .iter()
        .map(|g| ClusterInfo {
            index: g.index,
            size: g.size,
            mean_sim: g.mean_sim,
        })
        .collect();
    r.auto = Some(AutoReport {
        config: serde_json::to_value(&h.config)?,
        chosen_k: serde_json::to_value(h.chosen_k())?,
        groups: out.groups.clone(),
        leaves: out.leaves.clone(),
        diagnostics: serde_json::to_value(&h.diagnostics)?,
    });
    Ok(())
}

// ------------------------------------------------------------------ reading a run

pub fn read_report(run_dir: &Path) -> Result<Report> {
    serde_json::from_slice(
        &fs::read(run_dir.join(REPORT))
            .with_context(|| format!("reading {}/{REPORT}", run_dir.display()))?,
    )
    .context("parsing report.json")
}

/// Check that the first `report.n` embed rows are the ones the run was made from.
pub fn check_ids(report: &Report, emb: &Embeddings) -> Result<()> {
    ensure!(
        emb.n() >= report.n,
        "the run covers {} photos but embed/ has only {}",
        report.n,
        emb.n()
    );
    ensure!(
        ids_fnv(&emb.ids[..report.n]) == report.ids_fnv,
        "embed/rows.jsonl no longer starts with the rows this run was made from; recluster"
    );
    Ok(())
}

// ------------------------------------------------------------------ add

#[derive(Debug, Clone, Default)]
pub struct AddSummary {
    pub n_before: usize,
    pub n_after: usize,
    pub action: String,
    pub new_outliers: usize,
}

/// Assign photos embedded after the run was made, update the hierarchy by its policy, and
/// rewrite `clusters/<run>/` (the gallery is dropped: rerun `gallery`).
pub fn add(
    data: &Path,
    run: &str,
    backend: Backend,
    log: &mut dyn FnMut(&str),
) -> Result<AddSummary> {
    let run_dir = data.join("clusters").join(run);
    let mut report = read_report(&run_dir)?;
    ensure!(
        report.mode == "auto",
        "add needs a run made with --auto (it updates the saved hierarchy); {run} is {:?}",
        report.mode
    );
    let emb = store::load(&data.join("embed"))?;
    check_ids(&report, &emb)?;
    let n_old = report.n;
    let n = emb.n();
    if n == n_old {
        log(&format!("{run}: no new embedded photos (n = {n})"));
        return Ok(AddSummary {
            n_before: n_old,
            n_after: n,
            action: "none".into(),
            new_outliers: 0,
        });
    }
    let mut h = Hierarchy::load(run_dir.join(MODEL_DIR), backend)?;
    let t = Instant::now();
    let (leaf_ids, outlier, action, new_outliers, detail, dd) = match &report.dedup {
        Some(_) => {
            let mut dd: DedupResult = serde_json::from_slice(&fs::read(run_dir.join(DEDUP_STATE))?)
                .context("parsing dedup.json")?;
            let man = manifest_by_id(data)?;
            let side = SideInfo::new(&emb.ids, &man);
            let r = h.add_photos(&emb.x, n_old, &mut dd, &side.input(), backend)?;
            let outlier: Vec<u8> = r.outlier.iter().map(|&o| u8::from(o)).collect();
            let detail = serde_json::json!({
                "joined_existing_stacks": r.joined_existing,
                "new_stacks": r.new_groups,
                "new_outliers": r.reps.new_outliers,
                "pending": r.reps.pending,
            });
            (
                r.leaf,
                outlier,
                format!("{:?}", r.reps.action),
                r.reps.new_outliers,
                detail,
                Some(dd),
            )
        }
        None => {
            let r = h.add(&emb.x, n_old)?;
            let leaf: Vec<(u32, u32)> = (0..r.labels.len()).map(|i| r.labels.leaf(i)).collect();
            let outlier = r.labels.outlier.iter().map(|&o| u8::from(o)).collect();
            let detail =
                serde_json::json!({ "new_outliers": r.new_outliers, "pending": r.pending });
            (
                leaf,
                outlier,
                format!("{:?}", r.action),
                r.new_outliers,
                detail,
                None,
            )
        }
    };
    ensure!(
        leaf_ids.len() == n,
        "add returned {} labels for {n} photos",
        leaf_ids.len()
    );
    log(&format!(
        "{run}: added {} photos ({action}, {new_outliers} new outliers) in {:.1}s",
        n - n_old,
        t.elapsed().as_secs_f64()
    ));
    let out = hierarchy_outputs(&h, &leaf_ids, &emb.x)?;
    let history = std::mem::take(&mut report.history);
    report.n = n;
    report.n_videos = emb.kinds.iter().filter(|&&k| k == Kind::Video).count();
    report.ids_fnv = ids_fnv(&emb.ids);
    report.embed = emb.meta.clone();
    report.clustered_rows = h.n_seen();
    fill_auto_report(&mut report, &h, &out)?;
    report.history = history;
    report.history.push(HistoryEntry {
        at: now_iso(),
        action: action.clone(),
        n_before: n_old,
        n_after: n,
        detail,
    });
    let mut files: Vec<(&str, Vec<u8>)> = vec![
        (LABELS, bytes_i32(&out.labels1)),
        (CENTROIDS, bytes_f32(&out.centroids1)),
        (LABELS_L2, bytes_i32(&out.labels2)),
        (CENTROIDS_L2, bytes_f32(&out.centroids2)),
        (SIM, bytes_f32(&out.sim)),
        (OUTLIER, outlier),
    ];
    if let Some(dd) = &dd {
        let rep_row: Vec<i32> = dd
            .group_of
            .iter()
            .map(|&g| dd.representatives[g as usize] as i32)
            .collect();
        files.push((DEDUP_REP, bytes_i32(&rep_row)));
        files.push((DEDUP_STATE, serde_json::to_vec(dd)?));
    }
    files.push((REPORT, serde_json::to_vec_pretty(&report)?));
    if let Some(w) = commit_run_dir(&run_dir, true, &files, |dir| h.save(dir.join(MODEL_DIR)))? {
        log(&w);
    }
    Ok(AddSummary {
        n_before: n_old,
        n_after: n,
        action,
        new_outliers,
    })
}

/// Parse a backend name (`metal-fused`, `cpu`, ...).
pub fn parse_backend(s: &str) -> Result<Backend> {
    s.parse()
}

/// Default run name: `k64`, `auto`, `auto-dedup`, ...
pub fn default_run_name(mode: &Mode, dedup: bool) -> String {
    let base = match mode {
        Mode::K(k) => format!("k{k}"),
        Mode::Auto(_) => "auto".into(),
    };
    if dedup { format!("{base}-dedup") } else { base }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn force_replace_survives_an_undeletable_old_version() {
        let d = std::env::temp_dir().join(format!("pe-commit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let run = d.join("clusters/r");
        assert_eq!(
            commit_run_dir(&run, false, &[("a", b"1".to_vec())], |_| Ok(())).unwrap(),
            None
        );
        // A subdirectory whose entries cannot be removed (no write permission), like a directory
        // that another process keeps filling.
        let locked = run.join("gallery");
        fs::create_dir_all(locked.join("thumbs")).unwrap();
        fs::write(locked.join("thumbs/x.jpg"), b"x").unwrap();
        fs::set_permissions(locked.join("thumbs"), fs::Permissions::from_mode(0o555)).unwrap();
        let w = commit_run_dir(&run, true, &[("a", b"2".to_vec())], |_| Ok(())).unwrap();
        assert!(w.unwrap().contains("could not be deleted"));
        assert_eq!(
            fs::read(run.join("a")).unwrap(),
            b"2",
            "the new run is in place"
        );
        let old = d.join("clusters/.r.old");
        assert!(old.exists());
        // Once the old version can be deleted, the next commit deletes it first.
        fs::set_permissions(
            old.join("gallery/thumbs"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        assert_eq!(
            commit_run_dir(&run, true, &[("a", b"3".to_vec())], |_| Ok(())).unwrap(),
            None
        );
        assert!(!old.exists());
        assert_eq!(fs::read(run.join("a")).unwrap(), b"3");
        fs::remove_dir_all(&d).unwrap();
    }
}
