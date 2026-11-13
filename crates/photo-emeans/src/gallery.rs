//! The `gallery` step: a static, local HTML page for one run.
//!
//! `clusters/<run>/gallery/index.html` holds all data inline (no network, no CDN, no fonts) and
//! loads thumbnails from `gallery/thumbs/` and full images from `../../../photos/` and
//! `../../../videos/` by relative path. Thumbnails (long side 256 px) are made once into
//! `$EMEANS_DATA/thumbs/256/` and hard-linked into each gallery, so a second gallery costs no time
//! or space. A video's tile is its poster frame (thumbnail `v-<uuid>.jpg`) with a play badge and
//! its duration; the viewer shows the poster and a strip of the sampled frames.
//!
//! The page lets the user mark clusters for albums, edit each album title, and download
//! `approved.json` (`[{cluster, title, asset_ids}]`) for the `albums` step. Marks are kept in the
//! browser's localStorage for this run only as a convenience; the downloaded file is the record.
//!
//! With `dedup.json`, a "Clean up" tab lists the near-duplicate groups of 2 or more photos with a
//! suggested keeper ([`crate::cleanup`]) and downloads `cleanup.json`
//! (`[{asset_id, file, group, reason}]`) for `albums --cleanup`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::Serialize;

use crate::cleanup::{self, CleanupData};
use crate::cluster::{self, CENTROIDS, DEDUP_REP, DEDUP_STATE, LABELS, LABELS_L2, OUTLIER, SIM};
use crate::store;
use crate::util::write_atomic;

pub const THUMB_SIDE: u32 = 256;
const TEMPLATE: &str = include_str!("gallery.html");
const DATA_MARK: &str = "/*__DATA__*/";

#[derive(Debug, Clone)]
pub struct GalleryOpts {
    /// Thumbnails shown per cluster before "Show all".
    pub per_cluster: usize,
}

impl Default for GalleryOpts {
    fn default() -> Self {
        Self { per_cluster: 40 }
    }
}

#[derive(Serialize)]
struct Card {
    /// Label shown and written to approved.json: `"7"` (flat or level-1) or `"3.2"` (leaf).
    key: String,
    /// Level-1 group index for leaves; None for flat runs and for group cards.
    group: Option<usize>,
    size: usize,
    mean_sim: f32,
    start: String,
    end: String,
    /// Photo rows, most typical first (by cosine to the centroid).
    rows: Vec<u32>,
}

type Row = (
    String,
    String,
    String,
    u8,
    u8,
    String,
    Option<(f64, Vec<String>)>,
);

/// Thumbnail file name for an item: the photo's own file name, or `v-<uuid>.jpg` for a video.
pub fn thumb_name(item: &crate::manifest::Item) -> String {
    match item.kind {
        crate::store::Kind::Video => {
            let stem = item.image.split('/').nth(1).unwrap_or("video");
            format!("v-{stem}.jpg")
        }
        _ => item
            .image
            .rsplit('/')
            .next()
            .unwrap_or(&item.image)
            .to_string(),
    }
}

#[derive(Serialize)]
struct Page<'a> {
    run: &'a str,
    mode: &'a str,
    ids_fnv: &'a str,
    created: &'a str,
    n: usize,
    per_cluster: usize,
    /// Per row: [thumbnail file, asset id, YYYY-MM-DD or "", favorite 0/1, outlier 0/1, image
    /// path relative to $EMEANS_DATA, null or [duration seconds, [frame paths]] for a video].
    photos: Vec<Row>,
    /// Rows that are videos.
    n_videos: usize,
    /// Per photo: its dedup representative row (None without dedup).
    rep: Option<Vec<i32>>,
    stacks: usize,
    /// Flat clusters (`--k`) or leaves (`--auto`).
    cards: Vec<Card>,
    /// Level-1 groups (`--auto`), each with the indices of its leaves in `cards`.
    groups: Option<Vec<(Card, Vec<usize>)>>,
    /// Near-duplicate groups for the "Clean up" tab (runs with `dedup.json`).
    cleanup: Option<CleanupData>,
}

/// Make (or reuse) the 256 px thumbnail of `src` at `dst`.
pub fn make_thumb(src: &Path, dst: &Path) -> Result<()> {
    if fs::read(dst)
        .ok()
        .and_then(|b| photos::store::jpeg_size(&b))
        .is_some()
    {
        return Ok(());
    }
    let img = encoder::open_image(src)?;
    let t = img.thumbnail(THUMB_SIDE, THUMB_SIDE).to_rgb8();
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 80)
        .encode_image(&t)
        .with_context(|| format!("encoding thumbnail of {}", src.display()))?;
    let tmp = dst.with_extension("jpg.tmp");
    fs::write(&tmp, &buf)?;
    fs::rename(&tmp, dst)?;
    Ok(())
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }
    if fs::hard_link(src, dst).is_err() {
        fs::copy(src, dst).with_context(|| format!("copying {}", src.display()))?;
    }
    Ok(())
}

fn date_range(rows: &[u32], dates: &[String]) -> (String, String) {
    let mut v: Vec<&str> = rows
        .iter()
        .map(|&r| dates[r as usize].as_str())
        .filter(|d| !d.is_empty())
        .collect();
    v.sort_unstable();
    (
        v.first().map(|s| s.to_string()).unwrap_or_default(),
        v.last().map(|s| s.to_string()).unwrap_or_default(),
    )
}

/// Rows per label value `0..k`, sorted by similarity, best first.
fn rows_by_label(labels: &[i32], sim: &[f32], k: usize) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new(); k];
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 && (l as usize) < k {
            out[l as usize].push(i as u32);
        }
    }
    for v in &mut out {
        v.sort_by(|&a, &b| sim[b as usize].total_cmp(&sim[a as usize]).then(a.cmp(&b)));
    }
    out
}

/// Build `clusters/<run>/gallery/`. Returns the path of `index.html`.
pub fn build(
    data: &Path,
    run: &str,
    opts: &GalleryOpts,
    log: &mut dyn FnMut(&str),
) -> Result<PathBuf> {
    let run_dir = data.join("clusters").join(run);
    let report = cluster::read_report(&run_dir)?;
    let emb = store::load(&data.join("embed"))?;
    cluster::check_ids(&report, &emb)?;
    let n = report.n;
    let ids = &emb.ids[..n];
    let read_n = |name: &str| -> Result<Vec<i32>> {
        let v = cluster::read_i32(&run_dir.join(name))?;
        ensure!(v.len() == n, "{name} has {} rows, the run has {n}", v.len());
        Ok(v)
    };
    let labels = read_n(LABELS)?;
    let sim = cluster::read_f32(&run_dir.join(SIM))?;
    ensure!(
        sim.len() == n,
        "sim.f32 has {} rows, the run has {n}",
        sim.len()
    );
    ensure!(
        cluster::read_f32(&run_dir.join(CENTROIDS))?.len() == report.k * store::D,
        "centroids.f32 does not hold k = {} rows",
        report.k
    );
    let rep = if run_dir.join(DEDUP_REP).exists() {
        Some(read_n(DEDUP_REP)?)
    } else {
        None
    };
    let outlier = fs::read(run_dir.join(OUTLIER)).unwrap_or_else(|_| vec![0; n]);
    ensure!(
        outlier.len() == n,
        "outlier.u8 has {} rows, the run has {n}",
        outlier.len()
    );

    let man = cluster::manifest_by_id(data)?;
    let mut items = Vec::with_capacity(n);
    // (source image relative to data, thumbnail name)
    let mut files: Vec<(String, String)> = Vec::with_capacity(n);
    let mut dates = Vec::with_capacity(n);
    let mut photos: Vec<Row> = Vec::with_capacity(n);
    let mut n_videos = 0;
    for (i, id) in ids.iter().enumerate() {
        let l = man.get(id).with_context(|| {
            format!("embedded asset {id} is not in photos/ or videos/manifest.jsonl")
        })?;
        let date = l
            .created
            .as_deref()
            .and_then(|c| c.get(..10))
            .unwrap_or("")
            .to_string();
        items.push(l);
        let thumb = thumb_name(l);
        let video = (l.kind == crate::store::Kind::Video).then(|| (l.duration, l.frames.clone()));
        n_videos += usize::from(video.is_some());
        files.push((l.image.clone(), thumb.clone()));
        dates.push(date.clone());
        photos.push((
            thumb,
            id.clone(),
            date,
            u8::from(l.favorite),
            outlier[i],
            l.image.clone(),
            video,
        ));
    }

    // Thumbnails: shared cache, then hard links into this gallery.
    let gallery = run_dir.join("gallery");
    let thumbs = gallery.join("thumbs");
    let cache = data.join("thumbs").join(THUMB_SIDE.to_string());
    fs::create_dir_all(&thumbs)?;
    fs::create_dir_all(&cache)?;
    let t = Instant::now();
    let done = AtomicUsize::new(0);
    let failed: Vec<String> = files
        .par_iter()
        .filter_map(|(src, f)| {
            let r = make_thumb(&data.join(src), &cache.join(f))
                .and_then(|()| link_or_copy(&cache.join(f), &thumbs.join(f)));
            let k = done.fetch_add(1, Ordering::Relaxed) + 1;
            if k.is_multiple_of(2000) {
                eprintln!("  thumbnails {k}/{n}");
            }
            r.err().map(|e| format!("{f}: {e:#}"))
        })
        .collect();
    ensure!(
        failed.is_empty(),
        "{} thumbnails failed (first: {})",
        failed.len(),
        failed[0]
    );
    log(&format!(
        "{n} thumbnails ready in {:.1}s",
        t.elapsed().as_secs_f64()
    ));

    let card = |key: String, group: Option<usize>, rows: Vec<u32>| -> Card {
        let (start, end) = date_range(&rows, &dates);
        let mean_sim = if rows.is_empty() {
            0.0
        } else {
            rows.iter().map(|&r| sim[r as usize]).sum::<f32>() / rows.len() as f32
        };
        Card {
            key,
            group,
            size: rows.len(),
            mean_sim,
            start,
            end,
            rows,
        }
    };
    let (cards, groups) = match &report.auto {
        None => {
            let by = rows_by_label(&labels, &sim, report.k);
            let cards = by
                .into_iter()
                .enumerate()
                .filter(|(_, r)| !r.is_empty())
                .map(|(i, r)| card(i.to_string(), None, r))
                .collect();
            (cards, None)
        }
        Some(auto) => {
            let l2 = read_n(LABELS_L2)?;
            let leaf_rows = rows_by_label(&l2, &sim, auto.leaves.len());
            let group_rows = rows_by_label(&labels, &sim, auto.groups.len());
            let mut cards = Vec::new();
            let mut groups = Vec::new();
            for g in &auto.groups {
                if group_rows[g.index].is_empty() {
                    continue;
                }
                let mut members = Vec::new();
                let mut j = 0;
                for &leaf in &g.leaves {
                    if leaf_rows[leaf].is_empty() {
                        continue;
                    }
                    j += 1;
                    members.push(cards.len());
                    cards.push(card(
                        format!("{}.{j}", g.index),
                        Some(g.index),
                        leaf_rows[leaf].clone(),
                    ));
                }
                groups.push((
                    card(g.index.to_string(), None, group_rows[g.index].clone()),
                    members,
                ));
            }
            (cards, Some(groups))
        }
    };
    let stacks = rep
        .as_ref()
        .map(|r| {
            let mut sizes: HashMap<i32, usize> = HashMap::new();
            for &x in r {
                *sizes.entry(x).or_default() += 1;
            }
            sizes.values().filter(|&&s| s > 1).count()
        })
        .unwrap_or(0);
    let cleanup = if run_dir.join(DEDUP_STATE).exists() {
        let dd: emeans::DedupResult = serde_json::from_slice(&fs::read(run_dir.join(DEDUP_STATE))?)
            .context("parsing dedup.json")?;
        Some(cleanup::groups(&dd, &items)?)
    } else {
        None
    };
    let page = Page {
        run,
        mode: &report.mode,
        ids_fnv: &report.ids_fnv,
        created: &report.created,
        n,
        per_cluster: opts.per_cluster.max(1),
        photos,
        n_videos,
        rep,
        stacks,
        cards,
        groups,
        cleanup,
    };
    // `</` inside a <script> element would end it early.
    let json = serde_json::to_string(&page)?.replace("</", "<\\/");
    ensure!(
        TEMPLATE.contains(DATA_MARK),
        "gallery template has no data mark"
    );
    let html = TEMPLATE.replacen(DATA_MARK, &json, 1);
    let index = gallery.join("index.html");
    write_atomic(&index, html.as_bytes())?;
    log(&format!(
        "gallery: {} clusters{}{}, {} KB page",
        page.cards.len(),
        page.groups
            .as_ref()
            .map(|g| format!(" in {} groups", g.len()))
            .unwrap_or_default(),
        page.cleanup
            .as_ref()
            .map(|c| {
                let keep: usize = c.groups.iter().map(|g| g.keep.len()).sum();
                let rows: usize = c.groups.iter().map(|g| g.rows.len()).sum();
                format!(
                    "; clean-up: {} near-duplicate groups, {} photos, {} pre-marked for removal",
                    c.groups.len(),
                    rows,
                    rows - keep
                )
            })
            .unwrap_or_default(),
        html.len() / 1024
    ));
    Ok(index)
}
