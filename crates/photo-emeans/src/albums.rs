//! The `albums` step: read the `approved.json` that the gallery downloads, check it, print the
//! exact albums that would be created, and create them only with `--yes`. With `--cleanup`, read
//! `cleanup.json` instead and plan one album of the photos marked for removal; nothing is deleted.
//!
//! Albums sync to iCloud and appear on every device, so the default is a dry run.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use photos::AlbumRequest;
use serde::Deserialize;

use crate::cluster::manifest_by_id;
use crate::mainthread;

/// One entry of `approved.json`.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct Approved {
    /// Cluster label from the gallery (`"3"`, `"3.2"`); informational.
    pub cluster: serde_json::Value,
    pub title: String,
    pub asset_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PlannedAlbum {
    pub cluster: String,
    pub request: AlbumRequest,
    /// How many of its assets are videos.
    pub videos: usize,
    /// First and last creation date (`YYYY-MM-DD`) of its photos, when known.
    pub dates: Option<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub folder: String,
    pub albums: Vec<PlannedAlbum>,
    /// Distinct assets (photos and videos).
    pub photos: usize,
    /// How many of them are videos.
    pub videos: usize,
    /// Photos that appear in more than one album.
    pub shared: usize,
}

/// Parse `approved.json` text.
pub fn parse(text: &str) -> Result<Vec<Approved>> {
    serde_json::from_str(text)
        .context("approved.json must be a JSON array of {cluster, title, asset_ids}")
}

/// Check the approved list against the folder rules and the manifests (`known` asset ids, with
/// optional creation dates). Every id must be a fetched photo or video. PhotoKit adds photos and
/// videos to an album the same way (by `PHAsset` local identifier).
pub fn plan(
    folder: &str,
    approved: &[Approved],
    known: &HashMap<String, Option<String>>,
) -> Result<Plan> {
    plan_with_videos(folder, approved, known, &Default::default())
}

/// [`plan`], counting the ids in `videos` as videos.
pub fn plan_with_videos(
    folder: &str,
    approved: &[Approved],
    known: &HashMap<String, Option<String>>,
    videos: &std::collections::HashSet<String>,
) -> Result<Plan> {
    let albums: Vec<PlannedAlbum> = approved
        .iter()
        .map(|a| {
            let mut dates: Vec<&str> = a
                .asset_ids
                .iter()
                .filter_map(|id| known.get(id).and_then(|c| c.as_deref()))
                .filter_map(|c| c.get(..10))
                .collect();
            dates.sort_unstable();
            PlannedAlbum {
                cluster: match &a.cluster {
                    serde_json::Value::String(s) => s.clone(),
                    v => v.to_string(),
                },
                request: AlbumRequest {
                    title: a.title.trim().to_string(),
                    asset_ids: a.asset_ids.clone(),
                },
                videos: a.asset_ids.iter().filter(|id| videos.contains(*id)).count(),
                dates: dates
                    .first()
                    .zip(dates.last())
                    .map(|(a, b)| (a.to_string(), b.to_string())),
            }
        })
        .collect();
    let reqs: Vec<AlbumRequest> = albums.iter().map(|a| a.request.clone()).collect();
    // The same checks create_albums runs first (titles, duplicates, empty lists).
    photos::albums::check_request(folder, &reqs, true)?;
    let mut unknown = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for r in &reqs {
        for id in &r.asset_ids {
            if !known.contains_key(id) {
                unknown.push(id.as_str());
            }
            *seen.entry(id).or_default() += 1;
        }
    }
    ensure!(
        unknown.is_empty(),
        "{} asset ids in approved.json are not in photos/ or videos/manifest.jsonl (first: {}); \
         was the file made from another library or edited?",
        unknown.len(),
        unknown[0]
    );
    Ok(Plan {
        folder: folder.to_string(),
        videos: seen.keys().filter(|id| videos.contains(**id)).count(),
        photos: seen.len(),
        shared: seen.values().filter(|&&c| c > 1).count(),
        albums,
    })
}

/// The plan as the lines `albums` prints.
pub fn format_plan(p: &Plan) -> Vec<String> {
    let mut out = vec![format!(
        "Plan: one Photos folder {:?} with {} albums, {}{}.",
        p.folder,
        p.albums.len(),
        if p.videos > 0 {
            format!(
                "{} distinct assets ({} photos, {} videos)",
                p.photos,
                p.photos - p.videos,
                p.videos
            )
        } else {
            format!("{} distinct photos", p.photos)
        },
        if p.shared > 0 {
            format!(" ({} photos are in more than one album)", p.shared)
        } else {
            String::new()
        }
    )];
    let w = p
        .albums
        .iter()
        .map(|a| a.request.title.chars().count())
        .max()
        .unwrap_or(0);
    for (i, a) in p.albums.iter().enumerate() {
        out.push(format!(
            "  {:>3}. {:<w$}  {:>5} photos{}  cluster {:<6} {}",
            i + 1,
            a.request.title,
            a.request.asset_ids.len(),
            if a.videos > 0 {
                format!(" (incl. {} videos)", a.videos)
            } else {
                String::new()
            },
            a.cluster,
            a.dates
                .as_ref()
                .map(|(s, e)| if s == e {
                    s.clone()
                } else {
                    format!("{s} .. {e}")
                })
                .unwrap_or_default(),
        ));
    }
    out
}

/// Run the command. Without `yes` this only prints the plan. Must run on the main thread.
pub fn run(
    data: &Path,
    approved: &Path,
    folder: &str,
    yes: bool,
    timeout: Duration,
    out: &mut dyn FnMut(&str),
) -> Result<()> {
    let text =
        fs::read_to_string(approved).with_context(|| format!("reading {}", approved.display()))?;
    let list = parse(&text)?;
    let media = manifest_by_id(data)?;
    let videos: std::collections::HashSet<String> = media
        .iter()
        .filter(|(_, m)| m.kind == crate::store::Kind::Video)
        .map(|(id, _)| id.clone())
        .collect();
    let known: HashMap<String, Option<String>> =
        media.into_iter().map(|(id, l)| (id, l.created)).collect();
    let p = plan_with_videos(folder, &list, &known, &videos)?;
    for l in format_plan(&p) {
        out(&l);
    }
    if !yes {
        out("Dry run: nothing was created. Albums sync to iCloud on all devices.");
        out("To create exactly these albums, run the same command with --yes.");
        return Ok(());
    }
    let reqs: Vec<AlbumRequest> = p.albums.iter().map(|a| a.request.clone()).collect();
    let id = create(folder, reqs, timeout)?;
    out(&format!(
        "Created folder {:?} ({id}) with {} albums.",
        p.folder,
        p.albums.len()
    ));
    Ok(())
}

/// Default title of the `--cleanup` album.
pub const CLEANUP_TITLE: &str = "emeans \u{b7} to delete (review)";

/// The lines `albums --cleanup` prints for its one album.
pub fn format_cleanup(folder: &str, title: &str, c: &crate::cleanup::Checked) -> Vec<String> {
    let n = c.asset_ids.len();
    vec![
        format!(
            "Plan: one Photos folder {folder:?} with 1 album {title:?}: {}, from {} near-duplicate \
             groups{}.",
            if c.videos > 0 {
                format!("{n} assets ({} photos, {} videos)", n - c.videos, c.videos)
            } else {
                format!("{n} photos")
            },
            c.groups,
            c.dates
                .as_ref()
                .map(|(s, e)| if s == e {
                    format!(", {s}")
                } else {
                    format!(", {s} .. {e}")
                })
                .unwrap_or_default()
        ),
        "Nothing is deleted. The album only collects the marked photos; review it in Photos and \
         delete there."
            .to_string(),
    ]
}

/// `albums --cleanup`: one album with every asset of `cleanup.json`. Without `yes` this only
/// prints the plan. Must run on the main thread.
pub fn run_cleanup(
    data: &Path,
    cleanup: &Path,
    folder: &str,
    title: &str,
    yes: bool,
    timeout: Duration,
    out: &mut dyn FnMut(&str),
) -> Result<()> {
    let text =
        fs::read_to_string(cleanup).with_context(|| format!("reading {}", cleanup.display()))?;
    let list = crate::cleanup::parse(&text)?;
    let checked = crate::cleanup::check(&list, &manifest_by_id(data)?)?;
    let reqs = vec![AlbumRequest {
        title: title.trim().to_string(),
        asset_ids: checked.asset_ids.clone(),
    }];
    // The same checks create_albums runs first (folder, title, duplicate or empty ids).
    photos::albums::check_request(folder, &reqs, true)?;
    for l in format_cleanup(folder, title.trim(), &checked) {
        out(&l);
    }
    if !yes {
        out("Dry run: nothing was created. Albums sync to iCloud on all devices.");
        out("To create exactly this album, run the same command with --yes.");
        return Ok(());
    }
    let id = create(folder, reqs, timeout)?;
    out(&format!(
        "Created folder {folder:?} ({id}) with the album {:?}.",
        title.trim()
    ));
    Ok(())
}

/// Default title of the `--duplicates` album.
pub const DUPLICATES_TITLE: &str = "emeans \u{b7} duplicates to review";

/// A title a person can use: the members' capture months and the count, `2019-06 \u{2192} 2019-08
/// \u{b7} 46`, `2019-06 \u{b7} 12` for one month, `undated \u{b7} 3` without dates.
pub fn cluster_title(dates: &[&str], count: usize) -> String {
    let mut months: Vec<&str> = dates.iter().filter_map(|d| d.get(..7)).collect();
    months.sort_unstable();
    match (months.first(), months.last()) {
        (Some(a), Some(b)) if a == b => format!("{a} \u{b7} {count}"),
        (Some(a), Some(b)) => format!("{a} \u{2192} {b} \u{b7} {count}"),
        _ => format!("undated \u{b7} {count}"),
    }
}

/// Make titles unique in order: the second `T` becomes `T #2`, the third `T #3`, skipping any
/// suffixed title that is already taken.
pub fn unique_titles(titles: Vec<String>) -> Vec<String> {
    let mut taken: std::collections::HashSet<String> = titles.iter().cloned().collect();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    titles
        .into_iter()
        .map(|t| {
            if seen.insert(t.clone()) {
                return t;
            }
            let mut i = 2;
            loop {
                let c = format!("{t} #{i}");
                if !taken.contains(&c) {
                    taken.insert(c.clone());
                    seen.insert(c.clone());
                    return c;
                }
                i += 1;
            }
        })
        .collect()
}

/// Album lists built from a run without the gallery: `--all-clusters` and `--duplicates`.
#[derive(Debug, Clone, Default)]
pub struct RunLists {
    /// One album per cluster (`--k`) or leaf (`--auto`), every member (stack members included).
    pub all_clusters: bool,
    /// One album of the non-keepers of every near-duplicate group of 2 or more.
    pub duplicates: bool,
    pub duplicates_title: String,
}

/// Build the approved list for `run` (the `approved.json` format). Cluster albums come first,
/// oldest first; the duplicates album last. Fails if the duplicates album would hold a keeper.
pub fn run_lists(data: &Path, run: &str, opts: &RunLists) -> Result<Vec<Approved>> {
    use crate::cluster::{self, DEDUP_STATE, LABELS, LABELS_L2};
    ensure!(
        opts.all_clusters || opts.duplicates,
        "give --all-clusters, --duplicates or both"
    );
    let run_dir = data.join("clusters").join(run);
    let report = cluster::read_report(&run_dir)?;
    let emb = crate::store::load(&data.join("embed"))?;
    cluster::check_ids(&report, &emb)?;
    let n = report.n;
    let ids = &emb.ids[..n];
    let media = manifest_by_id(data)?;
    let items: Vec<&crate::manifest::Item> = ids
        .iter()
        .map(|id| {
            media.get(id).with_context(|| {
                format!("embedded asset {id} is not in photos/ or videos/manifest.jsonl")
            })
        })
        .collect::<Result<_>>()?;
    let date = |r: usize| items[r].created.as_deref().unwrap_or("");
    let mut out = Vec::new();
    if opts.all_clusters {
        let read = |name: &str| -> Result<Vec<i32>> {
            let v = cluster::read_i32(&run_dir.join(name))?;
            ensure!(v.len() == n, "{name} has {} rows, the run has {n}", v.len());
            Ok(v)
        };
        // Same keys as the gallery cards: "7" (flat) or "<group>.<j>" (leaf j of a group).
        let mut clusters: Vec<(String, Vec<usize>)> = Vec::new();
        match &report.auto {
            None => {
                let labels = read(LABELS)?;
                let mut by = vec![Vec::new(); report.k];
                for (r, &l) in labels.iter().enumerate() {
                    if l >= 0 && (l as usize) < report.k {
                        by[l as usize].push(r);
                    }
                }
                for (i, rows) in by.into_iter().enumerate() {
                    if !rows.is_empty() {
                        clusters.push((i.to_string(), rows));
                    }
                }
            }
            Some(auto) => {
                let l2 = read(LABELS_L2)?;
                let mut by = vec![Vec::new(); auto.leaves.len()];
                for (r, &l) in l2.iter().enumerate() {
                    if l >= 0 && (l as usize) < by.len() {
                        by[l as usize].push(r);
                    }
                }
                for g in &auto.groups {
                    let mut j = 0;
                    for &leaf in &g.leaves {
                        if by[leaf].is_empty() {
                            continue;
                        }
                        j += 1;
                        clusters.push((format!("{}.{j}", g.index), std::mem::take(&mut by[leaf])));
                    }
                }
            }
        }
        let start = |rows: &[usize]| {
            rows.iter()
                .map(|&r| date(r))
                .filter(|d| !d.is_empty())
                .min()
                .unwrap_or("9999")
                .to_string()
        };
        clusters.sort_by_cached_key(|(k, rows)| {
            (start(rows), std::cmp::Reverse(rows.len()), k.clone())
        });
        let titles = unique_titles(
            clusters
                .iter()
                .map(|(_, rows)| {
                    let d: Vec<&str> = rows.iter().map(|&r| date(r)).collect();
                    cluster_title(&d, rows.len())
                })
                .collect(),
        );
        for ((key, rows), title) in clusters.into_iter().zip(titles) {
            out.push(Approved {
                cluster: serde_json::Value::String(key),
                title,
                asset_ids: rows.iter().map(|&r| ids[r].clone()).collect(),
            });
        }
    }
    if opts.duplicates {
        let path = run_dir.join(DEDUP_STATE);
        ensure!(
            path.exists(),
            "run {run} has no dedup.json; cluster it with --dedup to get near-duplicate groups"
        );
        let dd: emeans::DedupResult =
            serde_json::from_slice(&fs::read(&path)?).context("parsing dedup.json")?;
        let groups = crate::cleanup::groups(&dd, &items)?;
        let mut rows = Vec::new();
        for g in &groups.groups {
            let keeper = g.rows[0];
            ensure!(g.keep.contains(&keeper), "group {}: keeper not kept", g.g);
            rows.extend(g.rows.iter().copied().filter(|r| !g.keep.contains(r)));
        }
        let keepers: std::collections::HashSet<u32> =
            groups.groups.iter().map(|g| g.rows[0]).collect();
        ensure!(
            rows.iter().all(|r| !keepers.contains(r)),
            "the duplicates album would hold a keeper"
        );
        ensure!(!rows.is_empty(), "run {run} has no near-duplicates to list");
        out.push(Approved {
            cluster: serde_json::Value::String("duplicates".into()),
            title: opts.duplicates_title.trim().to_string(),
            asset_ids: rows.iter().map(|&r| ids[r as usize].clone()).collect(),
        });
    }
    Ok(out)
}

/// `albums --run`: write the lists to `out` (the `approved.json` format) and print the dry-run
/// plan. Never creates albums: `albums --approved <out> --yes` does, after review.
pub fn run_from_run(
    data: &Path,
    run: &str,
    opts: &RunLists,
    folder: &str,
    out_path: &Path,
    out: &mut dyn FnMut(&str),
) -> Result<()> {
    let list = run_lists(data, run, opts)?;
    let media = manifest_by_id(data)?;
    let videos: std::collections::HashSet<String> = media
        .iter()
        .filter(|(_, m)| m.kind == crate::store::Kind::Video)
        .map(|(id, _)| id.clone())
        .collect();
    let known: HashMap<String, Option<String>> =
        media.into_iter().map(|(id, l)| (id, l.created)).collect();
    let p = plan_with_videos(folder, &list, &known, &videos)?;
    let json = serde_json::to_vec_pretty(
        &list
            .iter()
            .map(|a| serde_json::json!({"cluster": a.cluster, "title": a.title, "asset_ids": a.asset_ids}))
            .collect::<Vec<_>>(),
    )?;
    crate::util::write_atomic(out_path, &json)?;
    for l in format_plan(&p) {
        out(&l);
    }
    out(&format!("Wrote {}.", out_path.display()));
    out("Dry run: nothing was created. Albums sync to iCloud on all devices.");
    out(&format!(
        "To create exactly these albums, run: albums --approved {} --folder {folder:?} --yes",
        out_path.display()
    ));
    Ok(())
}

/// A checked rename: the folder and each album with its local identifier, and notes to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenamePlan {
    pub folder: photos::Rename,
    /// Same order as the file's albums.
    pub albums: Vec<photos::Rename>,
    /// Assets in each album now (same order).
    pub assets: Vec<usize>,
    /// Non-fatal notes: an asset count that changed since the album was created, extra albums.
    pub notes: Vec<String>,
}

/// Match `album-names.json` against the folder as Photos lists it. Forward: old titles become new
/// titles; `reverse`: new titles go back to the old ones. Every album is found by the folder and
/// its current title (unique in the folder), and must match a local identifier the file already
/// records. Fails if an album is missing, a title is ambiguous, or a new title would clash with an
/// album that is not renamed.
pub fn plan_rename(
    f: &crate::names::NamesFile,
    listing: &photos::FolderListing,
    reverse: bool,
) -> Result<RenamePlan> {
    let pick = |old: &str, new: &str| -> (String, String) {
        if reverse {
            (new.to_string(), old.to_string())
        } else {
            (old.to_string(), new.to_string())
        }
    };
    let (ffrom, fto) = pick(&f.folder.old_title, &f.folder.new_title);
    ensure!(
        listing.title == ffrom,
        "the folder found is titled {:?}, expected {ffrom:?}",
        listing.title
    );
    if let Some(id) = &f.folder.local_id {
        ensure!(
            *id == listing.id,
            "folder {ffrom:?} is {}, but the file recorded {id}",
            listing.id
        );
    }
    let mut by_title: HashMap<&str, Vec<&photos::CollectionInfo>> = HashMap::new();
    for c in &listing.children {
        by_title.entry(c.title.as_str()).or_default().push(c);
    }
    let mut albums = Vec::with_capacity(f.albums.len());
    let mut assets = Vec::with_capacity(f.albums.len());
    let mut notes = Vec::new();
    let mut missing = Vec::new();
    let mut applied = 0;
    for a in &f.albums {
        let (from, to) = pick(&a.old_title, &a.new_title);
        match by_title.get(from.as_str()).map(Vec::as_slice) {
            Some([c]) => {
                let n = c.assets.with_context(|| {
                    format!("{from:?} in folder {ffrom:?} is a folder, not an album")
                })?;
                if let Some(id) = &a.local_id {
                    ensure!(
                        *id == c.id,
                        "album {from:?} is {}, but the file recorded {id}",
                        c.id
                    );
                }
                if n != a.count {
                    notes.push(format!(
                        "album {from:?} holds {n} assets now, {} when created",
                        a.count
                    ));
                }
                albums.push(photos::Rename {
                    id: c.id.clone(),
                    from,
                    to,
                });
                assets.push(n);
            }
            Some(many) => bail!(
                "{} albums in folder {ffrom:?} are titled {from:?}; rename one by hand first",
                many.len()
            ),
            None => {
                if by_title.contains_key(to.as_str()) {
                    applied += 1;
                }
                missing.push(from);
            }
        }
    }
    if !missing.is_empty() {
        bail!(
            "{} of {} albums are not in folder {ffrom:?} under their expected title (first: {:?}){}",
            missing.len(),
            f.albums.len(),
            missing[0],
            if applied > 0 {
                format!(
                    "; {applied} of them already carry the target title. Was this rename already \
                     applied{}?",
                    if reverse { " in reverse" } else { "" }
                )
            } else {
                String::new()
            }
        );
    }
    let renamed: std::collections::HashSet<&str> = albums.iter().map(|r| r.id.as_str()).collect();
    let targets: std::collections::HashSet<&str> = albums.iter().map(|r| r.to.as_str()).collect();
    for c in &listing.children {
        if !renamed.contains(c.id.as_str()) {
            ensure!(
                !targets.contains(c.title.as_str()),
                "album {:?} is not renamed but one of the new titles is the same",
                c.title
            );
            if !f.unchanged.contains(&c.title) {
                notes.push(format!(
                    "folder {ffrom:?} also holds {:?}; it stays as it is",
                    c.title
                ));
            }
        }
    }
    let folder = photos::Rename {
        id: listing.id.clone(),
        from: ffrom,
        to: fto,
    };
    photos::albums::check_renames(&folder, &albums, true)?;
    Ok(RenamePlan {
        folder,
        albums,
        assets,
        notes,
    })
}

/// The lines `albums --rename-from` prints.
pub fn format_rename(p: &RenamePlan) -> Vec<String> {
    let mut out = vec![format!(
        "Plan: rename folder {:?} to {:?} ({}) and {} of its albums (titles only):",
        p.folder.from,
        p.folder.to,
        p.folder.id,
        p.albums.len()
    )];
    let w = p
        .albums
        .iter()
        .map(|r| r.from.chars().count())
        .max()
        .unwrap_or(0);
    for (i, (r, n)) in p.albums.iter().zip(&p.assets).enumerate() {
        out.push(format!(
            "  {:>3}. {:<w$}  ->  {}   ({n} assets)",
            i + 1,
            r.from,
            r.to
        ));
    }
    out.extend(p.notes.iter().map(|n| format!("Note: {n}")));
    out
}

/// `albums --rename-from FILE [--reverse]`: find the folder in Photos (read-only), check the
/// plan, record the local identifiers in FILE, print the plan, and rename only with `yes`. Must
/// run on the main thread.
pub fn run_rename(
    names_path: &Path,
    reverse: bool,
    yes: bool,
    timeout: Duration,
    out: &mut dyn FnMut(&str),
) -> Result<()> {
    use crate::names::NamesFile;
    ensure!(
        mainthread::is_main_thread(),
        "albums --rename-from must run on the main thread (it keeps the main queue running for Photos)"
    );
    let mut f = NamesFile::read(names_path)?;
    photos::ensure_authorized(Duration::from_secs(300))?;
    let from = if reverse {
        &f.folder.new_title
    } else {
        &f.folder.old_title
    };
    let listing = photos::find_folder(from)?;
    let p = plan_rename(&f, &listing, reverse)?;
    // Record the identifiers once, so the undo can check it finds the same albums.
    if f.folder.local_id.is_none() || f.albums.iter().any(|a| a.local_id.is_none()) {
        f.folder.local_id = Some(p.folder.id.clone());
        for (a, r) in f.albums.iter_mut().zip(&p.albums) {
            a.local_id = Some(r.id.clone());
        }
        f.write(names_path)?;
        out(&format!(
            "Recorded the local identifiers in {}.",
            names_path.display()
        ));
    }
    for l in format_rename(&p) {
        out(&l);
    }
    let undo = if reverse { "" } else { " --reverse" };
    if !yes {
        out("Dry run: nothing was renamed. Titles sync to iCloud on all devices.");
        out(&format!(
            "To rename exactly these, run: albums --rename-from {}{} --yes",
            names_path.display(),
            if reverse { " --reverse" } else { "" }
        ));
        return Ok(());
    }
    let (folder, albums) = (p.folder.clone(), p.albums.clone());
    // performChangesAndWait must not run on the main thread (see `create`).
    mainthread::run_pumping(
        move || photos::rename_collections(&folder, &albums, true),
        timeout,
        Duration::from_millis(20),
    )
    .with_context(|| {
        format!(
            "Photos did not confirm the rename. The change is one transaction: it either completes \
             or does nothing. Run albums --rename-from {} (dry run) to see which titles Photos has",
            names_path.display()
        )
    })??;
    let after = photos::find_folder(&p.folder.to)?;
    out(&format!(
        "Renamed folder {:?} to {:?} and {} albums; the folder now holds {} collections.",
        p.folder.from,
        p.folder.to,
        p.albums.len(),
        after.children.len()
    ));
    out(&format!(
        "Undo: albums --rename-from {}{undo} --yes",
        names_path.display()
    ));
    Ok(())
}

/// Create the albums through `photos::create_albums`. Must run on the main thread.
fn create(folder: &str, reqs: Vec<AlbumRequest>, timeout: Duration) -> Result<String> {
    ensure!(
        mainthread::is_main_thread(),
        "albums --yes must run on the main thread (it keeps the main queue running for Photos)"
    );
    photos::ensure_authorized(Duration::from_secs(300))?;
    let folder_name = folder.to_string();
    // create_albums uses performChangesAndWait, which Apple says must not run on the main
    // thread and may block while Photos prompts the user. Run it on a worker; keep the main run
    // loop going meanwhile (see `mainthread`).
    let id = mainthread::run_pumping(
        move || photos::create_albums(&folder_name, &reqs, true),
        timeout,
        Duration::from_millis(20),
    )
    .with_context(|| {
        format!(
            "Photos did not confirm the albums. The change is one transaction: it either completes \
             or does nothing. Open Photos and look for the folder {folder:?} before you run --yes \
             again, or a second folder will be created"
        )
    })??;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known() -> HashMap<String, Option<String>> {
        [
            ("a/L0/001", Some("2019-06-01T10:00:00Z")),
            ("b/L0/001", Some("2019-08-30T10:00:00Z")),
            ("c/L0/001", None),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.map(String::from)))
        .collect()
    }

    #[test]
    fn plan_and_format() {
        let list = parse(
            r#"[{"cluster":"3.2","title":" Beach ","asset_ids":["a/L0/001","b/L0/001"]},
                {"cluster":7,"title":"Cats","asset_ids":["b/L0/001","c/L0/001"]}]"#,
        )
        .unwrap();
        let p = plan("emeans", &list, &known()).unwrap();
        assert_eq!((p.albums.len(), p.photos, p.shared, p.videos), (2, 3, 1, 0));
        let vids = ["c/L0/001".to_string()].into_iter().collect();
        let pv = plan_with_videos("emeans", &list, &known(), &vids).unwrap();
        assert_eq!(pv.videos, 1);
        let lines = format_plan(&pv);
        assert!(
            lines[0].contains("3 distinct assets (2 photos, 1 videos)"),
            "{}",
            lines[0]
        );
        assert!(
            lines[2].contains("2 photos (incl. 1 videos)"),
            "{}",
            lines[2]
        );
        assert!(!lines[1].contains("incl."));
        assert_eq!(p.albums[0].request.title, "Beach");
        assert_eq!(p.albums[1].cluster, "7");
        let lines = format_plan(&p);
        assert!(
            lines[0].contains("2 albums, 3 distinct photos (1 photos are in more than one album)")
        );
        assert!(lines[1].contains("Beach") && lines[1].contains("2019-06-01 .. 2019-08-30"));
        assert!(lines[2].contains("2019-08-30") && !lines[2].contains(".."));
    }

    #[test]
    fn titles() {
        assert_eq!(
            cluster_title(&["2019-08-30T10:00:00Z", "2019-06-01T00:00:00Z", ""], 46),
            "2019-06 \u{2192} 2019-08 \u{b7} 46"
        );
        assert_eq!(
            cluster_title(&["2019-06-01", "2019-06-30"], 2),
            "2019-06 \u{b7} 2"
        );
        assert_eq!(cluster_title(&["", ""], 3), "undated \u{b7} 3");
        let t = |v: &[&str]| unique_titles(v.iter().map(|s| s.to_string()).collect());
        assert_eq!(t(&["A", "B", "A", "A"]), ["A", "B", "A #2", "A #3"]);
        // A suffixed title that already exists is skipped.
        assert_eq!(t(&["A", "A #2", "A"]), ["A", "A #2", "A #3"]);
    }

    fn names_file() -> crate::names::NamesFile {
        use crate::names::{AlbumName, FolderNames, NamesFile, ScoreOpts};
        let album = |c: &str, old: &str, new: &str, count: usize| AlbumName {
            cluster: c.into(),
            local_id: None,
            old_title: old.into(),
            new_title: new.into(),
            count,
            mixed: false,
            margin: 0.5,
            top: vec![],
        };
        NamesFile {
            version: 1,
            created: "2026-11-13T00:00:00Z".into(),
            approved: "approved.json".into(),
            model: "m".into(),
            prompt: "p".into(),
            labels_source: "builtin".into(),
            labels_fnv: "0".into(),
            labels: 2,
            method: ScoreOpts::default(),
            folder: FolderNames {
                local_id: None,
                old_title: "emeans".into(),
                new_title: "Photo groups".into(),
            },
            unchanged: vec!["dups".into()],
            albums: vec![
                album("1.1", "2019-01 \u{b7} 3", "Beach \u{b7} 2019 \u{b7} 3", 3),
                album("2.1", "2020-01 \u{b7} 2", "Cat \u{b7} 2020 \u{b7} 2", 2),
            ],
        }
    }

    fn listing(folder: &str, kids: &[(&str, &str, Option<usize>)]) -> photos::FolderListing {
        photos::FolderListing {
            id: "F".into(),
            title: folder.into(),
            children: kids
                .iter()
                .map(|(id, t, n)| photos::CollectionInfo {
                    id: id.to_string(),
                    title: t.to_string(),
                    assets: *n,
                })
                .collect(),
        }
    }

    #[test]
    fn rename_plan_forward_and_reverse() {
        let f = names_file();
        let before = listing(
            "emeans",
            &[
                ("a", "2019-01 \u{b7} 3", Some(3)),
                ("b", "2020-01 \u{b7} 2", Some(1)),
                ("d", "dups", Some(9)),
            ],
        );
        let p = plan_rename(&f, &before, false).unwrap();
        assert_eq!(
            (p.folder.from.as_str(), p.folder.to.as_str()),
            ("emeans", "Photo groups")
        );
        assert_eq!(p.albums[0].id, "a");
        assert_eq!(p.albums[1].to, "Cat \u{b7} 2020 \u{b7} 2");
        assert_eq!(p.assets, [3, 1]);
        // A changed count is a note, the duplicates album is not.
        assert_eq!(p.notes.len(), 1, "{:?}", p.notes);
        assert!(p.notes[0].contains("holds 1 assets now, 2 when created"));
        let lines = format_rename(&p);
        assert!(lines[1].contains("->  Beach"), "{}", lines[1]);

        // Undo after the change: new titles back to old ones, in folder "Photo groups".
        let mut f2 = f.clone();
        f2.folder.local_id = Some("F".into());
        f2.albums[0].local_id = Some("a".into());
        f2.albums[1].local_id = Some("b".into());
        let after = listing(
            "Photo groups",
            &[
                ("a", "Beach \u{b7} 2019 \u{b7} 3", Some(3)),
                ("b", "Cat \u{b7} 2020 \u{b7} 2", Some(2)),
                ("d", "dups", Some(9)),
            ],
        );
        let r = plan_rename(&f2, &after, true).unwrap();
        assert_eq!(r.albums[0].to, "2019-01 \u{b7} 3");
        assert_eq!(r.folder.to, "emeans");
        assert!(r.notes.is_empty());
        // Applying forward again says it looks applied.
        let e = plan_rename(
            &f2,
            &listing(
                "emeans",
                &after
                    .children
                    .iter()
                    .map(|c| (c.id.as_str(), c.title.as_str(), c.assets))
                    .collect::<Vec<_>>(),
            ),
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("already"), "{e}");
        // A recorded id that does not match fails.
        let mut f3 = f2.clone();
        f3.albums[0].local_id = Some("zzz".into());
        assert!(plan_rename(&f3, &after, true).is_err());
    }

    #[test]
    fn rename_plan_refusals() {
        let f = names_file();
        let ok = [
            ("a", "2019-01 \u{b7} 3", Some(3)),
            ("b", "2020-01 \u{b7} 2", Some(2)),
        ];
        // Wrong folder title.
        assert!(plan_rename(&f, &listing("other", &ok), false).is_err());
        // A missing album.
        assert!(plan_rename(&f, &listing("emeans", &ok[..1]), false).is_err());
        // Two albums with the same old title.
        let mut two = ok.to_vec();
        two.push(("c", "2019-01 \u{b7} 3", Some(3)));
        assert!(plan_rename(&f, &listing("emeans", &two), false).is_err());
        // A sub-folder with the title.
        assert!(
            plan_rename(
                &f,
                &listing("emeans", &[("a", "2019-01 \u{b7} 3", None), ok[1]]),
                false
            )
            .is_err()
        );
        // Another album already has a new title.
        let mut clash = ok.to_vec();
        clash.push(("x", "Cat \u{b7} 2020 \u{b7} 2", Some(1)));
        let e = plan_rename(&f, &listing("emeans", &clash), false).unwrap_err();
        assert!(e.to_string().contains("not renamed"), "{e}");
        // An unknown extra album is only a note.
        let mut extra = ok.to_vec();
        extra.push(("y", "Holiday", Some(4)));
        let p = plan_rename(&f, &listing("emeans", &extra), false).unwrap();
        assert!(p.notes.iter().any(|n| n.contains("Holiday")));
    }

    #[test]
    fn rejects_bad_input() {
        let k = known();
        for bad in [
            r#"[{"cluster":1,"title":"","asset_ids":["a/L0/001"]}]"#,
            r#"[{"cluster":1,"title":"X","asset_ids":[]}]"#,
            r#"[{"cluster":1,"title":"X","asset_ids":["a/L0/001"]},{"cluster":2,"title":"X","asset_ids":["b/L0/001"]}]"#,
            r#"[{"cluster":1,"title":"X","asset_ids":["a/L0/001","a/L0/001"]}]"#,
            r#"[{"cluster":1,"title":"X","asset_ids":["zzz"]}]"#,
            r#"[]"#,
        ] {
            assert!(plan("emeans", &parse(bad).unwrap(), &k).is_err(), "{bad}");
        }
        assert!(
            plan(
                "  ",
                &parse(r#"[{"cluster":1,"title":"X","asset_ids":["a/L0/001"]}]"#).unwrap(),
                &k
            )
            .is_err()
        );
        assert!(parse(r#"{"title":"X"}"#).is_err());
    }
}
