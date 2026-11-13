//! cluster -> gallery -> add -> albums (dry run) on synthetic vectors and tiny JPEGs. CPU only.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use emeans::{AutoConfig, Backend, CpuKernel, DedupConfig};
use photo_emeans::cluster::{self, ClusterOpts, Mode};
use photo_emeans::gallery::{self, GalleryOpts};
use photo_emeans::store::{self, D, EmbedStore, Meta};
use photo_emeans::{albums, util};

const CENTERS: usize = 6;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    }
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

/// Row `i` of the synthetic library: true cluster `i % CENTERS`; every 10th row is a
/// near-copy of the row before it (a burst-like near-duplicate).
fn vectors(n: usize, seed: u64) -> (Vec<[f32; D]>, Vec<usize>) {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let centers: Vec<Vec<f32>> = (0..CENTERS)
        .map(|_| {
            let mut c: Vec<f32> = (0..D).map(|_| r.next()).collect();
            normalize(&mut c);
            c
        })
        .collect();
    let mut r = Rng(seed);
    let mut out: Vec<[f32; D]> = Vec::new();
    let mut truth = Vec::new();
    for i in 0..n {
        let t = i % CENTERS;
        let mut v = [0f32; D];
        if i % 10 == 9 && i > 0 {
            let prev = out[i - 1];
            for j in 0..D {
                v[j] = prev[j] + 0.002 * r.next();
            }
            truth.push(truth[i - 1]);
        } else {
            for j in 0..D {
                v[j] = centers[t][j] + 0.045 * r.next();
            }
            truth.push(t);
        }
        normalize(&mut v);
        out.push(v);
    }
    (out, truth)
}

fn id(i: usize) -> String {
    format!("P{i:04}-AAAA/L0/001")
}

fn file(i: usize) -> String {
    format!("P{i:04}-AAAA.jpg")
}

/// Photos `from..from+n`: JPEG, manifest line, and embedding row.
fn add_rows(data: &Path, from: usize, n: usize, vecs: &[[f32; D]]) {
    fs::create_dir_all(data.join("photos")).unwrap();
    let mut m = OpenOptions::new()
        .create(true)
        .append(true)
        .open(data.join("photos/manifest.jsonl"))
        .unwrap();
    for i in from..from + n {
        let img = image::RgbImage::from_fn(320, 240, |x, y| {
            image::Rgb([(x + i as u32) as u8, y as u8, (i * 40) as u8])
        });
        img.save(data.join("photos").join(file(i))).unwrap();
        let created = util::now_unix() - 86_400.0 * (400 - i) as f64;
        writeln!(
            m,
            r#"{{"id":"{}","file":"{}","created":"{}","width":320,"height":240,"favorite":{},"burst_id":null,"subtypes":[]}}"#,
            id(i),
            file(i),
            photos::assets::iso8601_utc(created),
            i % 17 == 0
        )
        .unwrap();
    }
    let (mut st, _) = EmbedStore::open(
        &data.join("embed"),
        Meta::new("synthetic", "1", "f32"),
        &mut |_| {},
    )
    .unwrap();
    let rows: Vec<(String, [f32; D])> = (from..from + n).map(|i| (id(i), vecs[i])).collect();
    st.append(&rows, store::Kind::Photo).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pe-pipe-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

const CPU: Backend = Backend::Cpu(CpuKernel::Accelerate);

fn auto_cfg() -> AutoConfig {
    AutoConfig {
        k1_range: (2, 10),
        restarts: 2,
        min_cluster_size: 10,
        max_cluster_size: 200,
        ..AutoConfig::default()
    }
}

fn page_data(index: &Path) -> serde_json::Value {
    let html = fs::read_to_string(index).unwrap();
    let start = html
        .find(r#"<script id="data" type="application/json">"#)
        .unwrap();
    let rest = &html[start..];
    let body = &rest[rest.find('>').unwrap() + 1..rest.find("</script>").unwrap()];
    assert!(
        !html.contains("<link") && !html.contains("src=\"http"),
        "no network resources"
    );
    serde_json::from_str(&body.replace("<\\/", "</")).unwrap()
}

#[test]
fn k_mode_files_and_gallery() {
    let d = tmp("k");
    let n = 240;
    let (vecs, truth) = vectors(n, 7);
    add_rows(&d, 0, n, &vecs);
    let opts = ClusterOpts {
        run: "k6".into(),
        mode: Mode::K(CENTERS),
        backend: CPU,
        dedup: None,
        nepochs: 5.0,
        seed: 1,
        force: false,
    };
    let run = cluster::cluster(&d, &opts, &mut |_| {}).unwrap();
    let labels = cluster::read_i32(&run.join("labels.i32")).unwrap();
    let cents = cluster::read_f32(&run.join("centroids.f32")).unwrap();
    let sim = cluster::read_f32(&run.join("sim.f32")).unwrap();
    assert_eq!(labels.len(), n);
    assert_eq!(cents.len(), CENTERS * D);
    assert_eq!(sim.len(), n);
    assert!(run.join("model.emeans").is_file());
    let emb = store::load(&d.join("embed")).unwrap();
    for i in 0..n {
        let l = labels[i] as usize;
        assert!(l < CENTERS);
        let s: f32 = emb
            .row(i)
            .iter()
            .zip(&cents[l * D..(l + 1) * D])
            .map(|(a, b)| a * b)
            .sum();
        assert!((s - sim[i]).abs() < 1e-5);
    }
    // Each label is the nearest centroid (labels.i32 and centroids.f32 agree).
    for (i, &si) in sim.iter().enumerate() {
        let best = (0..CENTERS)
            .map(|c| {
                emb.row(i)
                    .iter()
                    .zip(&cents[c * D..(c + 1) * D])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .fold(f32::MIN, f32::max);
        assert!(si >= best - 1e-5, "row {i} is not at its nearest centroid");
    }
    let _ = &truth;
    let rep = cluster::read_report(&run).unwrap();
    assert_eq!(
        (rep.mode.as_str(), rep.n, rep.k, rep.clusters.len()),
        ("k", n, CENTERS, CENTERS)
    );
    assert_eq!(rep.clusters.iter().map(|c| c.size).sum::<usize>(), n);
    assert!(rep.recipe.is_some());
    // Existing run is not replaced without force.
    assert!(cluster::cluster(&d, &opts, &mut |_| {}).is_err());
    cluster::cluster(
        &d,
        &ClusterOpts {
            force: true,
            ..opts
        },
        &mut |_| {},
    )
    .unwrap();

    let index = gallery::build(&d, "k6", &GalleryOpts::default(), &mut |_| {}).unwrap();
    let v = page_data(&index);
    assert_eq!(v["photos"].as_array().unwrap().len(), n);
    assert!(v["groups"].is_null());
    let cards = v["cards"].as_array().unwrap();
    assert_eq!(cards.len(), CENTERS);
    let mut all: Vec<u64> = cards
        .iter()
        .flat_map(|c| {
            c["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_u64().unwrap())
        })
        .collect();
    all.sort_unstable();
    assert_eq!(all, (0..n as u64).collect::<Vec<_>>());
    // Thumbnails: one per photo, long side 256.
    for i in 0..n {
        let b = fs::read(index.parent().unwrap().join("thumbs").join(file(i))).unwrap();
        let (w, h) = photos::store::jpeg_size(&b).unwrap();
        assert_eq!(w.max(h), 256);
    }
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn auto_dedup_gallery_add_and_albums_dry_run() {
    let d = tmp("auto");
    let n = 360;
    let extra = 40;
    let (vecs, _) = vectors(n + extra, 11);
    add_rows(&d, 0, n, &vecs);
    let opts = ClusterOpts {
        run: "auto-dedup".into(),
        mode: Mode::Auto(Box::new(auto_cfg())),
        backend: CPU,
        dedup: Some(DedupConfig::default()),
        nepochs: 5.0,
        seed: 1,
        force: false,
    };
    let run = cluster::cluster(&d, &opts, &mut |_| {}).unwrap();
    let rep = cluster::read_report(&run).unwrap();
    let auto = rep.auto.as_ref().unwrap();
    let dd = rep.dedup.as_ref().unwrap();
    // Every 10th row is a near-copy of the one before: 36 stacks of 2.
    assert_eq!(dd.groups, n - n / 10);
    assert_eq!(dd.photos_in_stacks, 2 * (n / 10));
    assert_eq!(rep.clustered_rows, dd.groups);
    let l1 = cluster::read_i32(&run.join("labels.i32")).unwrap();
    let l2 = cluster::read_i32(&run.join("labels_l2.i32")).unwrap();
    let reps = cluster::read_i32(&run.join("dedup.i32")).unwrap();
    assert_eq!((l1.len(), l2.len(), reps.len()), (n, n, n));
    assert_eq!(
        cluster::read_f32(&run.join("centroids.f32")).unwrap().len(),
        rep.k * D
    );
    assert_eq!(
        cluster::read_f32(&run.join("centroids_l2.f32"))
            .unwrap()
            .len(),
        rep.k_leaves.unwrap() * D
    );
    for i in 0..n {
        let leaf = &auto.leaves[l2[i] as usize];
        assert_eq!(leaf.group as i32, l1[i], "leaf and level-1 label agree");
        let r = reps[i] as usize;
        assert_eq!(reps[r] as usize, r, "a representative represents itself");
        assert_eq!(l2[r], l2[i], "a stack stays in one leaf");
    }
    assert_eq!(auto.groups.iter().map(|g| g.size).sum::<usize>(), n);
    assert_eq!(auto.leaves.iter().map(|l| l.size).sum::<usize>(), n);
    assert!(run.join("model").is_dir() && run.join("dedup.json").is_file());

    let index = gallery::build(&d, "auto-dedup", &GalleryOpts::default(), &mut |_| {}).unwrap();
    let v = page_data(&index);
    assert_eq!(v["stacks"].as_u64().unwrap() as usize, n / 10);
    let groups = v["groups"].as_array().unwrap();
    assert!(!groups.is_empty());
    let cards = v["cards"].as_array().unwrap();
    let leaf_total: u64 = cards.iter().map(|c| c["size"].as_u64().unwrap()).sum();
    assert_eq!(leaf_total as usize, n);

    // add: 40 more photos embedded later.
    add_rows(&d, n, extra, &vecs);
    let s = cluster::add(&d, "auto-dedup", CPU, &mut |_| {}).unwrap();
    assert_eq!((s.n_before, s.n_after), (n, n + extra));
    let rep2 = cluster::read_report(&run).unwrap();
    assert_eq!(rep2.n, n + extra);
    assert_eq!(rep2.history.len(), 1);
    assert_eq!(
        cluster::read_i32(&run.join("labels_l2.i32")).unwrap().len(),
        n + extra
    );
    assert_eq!(
        cluster::read_i32(&run.join("dedup.i32")).unwrap().len(),
        n + extra
    );
    assert!(!run.join("gallery").exists(), "add drops the stale gallery");
    let again = cluster::add(&d, "auto-dedup", CPU, &mut |_| {}).unwrap();
    assert_eq!(again.action, "none");
    let index = gallery::build(&d, "auto-dedup", &GalleryOpts::default(), &mut |_| {}).unwrap();
    assert_eq!(
        page_data(&index)["photos"].as_array().unwrap().len(),
        n + extra
    );

    // albums: an approved.json as the page would write it, dry run only.
    let v = page_data(&index);
    let photos = v["photos"].as_array().unwrap();
    let approved: Vec<serde_json::Value> = v["cards"]
        .as_array()
        .unwrap()
        .iter()
        .take(2)
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "cluster": c["key"],
                "title": format!("Album {i}"),
                "asset_ids": c["rows"].as_array().unwrap().iter()
                    .map(|r| photos[r.as_u64().unwrap() as usize][1].clone()).collect::<Vec<_>>(),
            })
        })
        .collect();
    let path = d.join("approved.json");
    fs::write(&path, serde_json::to_vec(&approved).unwrap()).unwrap();
    let mut lines = Vec::new();
    albums::run(
        &d,
        &path,
        "emeans",
        false,
        std::time::Duration::from_secs(1),
        &mut |l| lines.push(l.to_string()),
    )
    .unwrap();
    assert!(lines[0].starts_with("Plan: one Photos folder \"emeans\" with 2 albums"));
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("Dry run: nothing was created"))
    );
    assert!(!lines.iter().any(|l| l.starts_with("Created")));
    // ids that are not in the manifest are refused.
    fs::write(
        &path,
        r#"[{"cluster":"x","title":"T","asset_ids":["nope"]}]"#,
    )
    .unwrap();
    assert!(
        albums::run(
            &d,
            &path,
            "emeans",
            false,
            std::time::Duration::from_secs(1),
            &mut |_| {}
        )
        .is_err()
    );
    fs::remove_dir_all(&d).unwrap();
}

/// Rewrite photo `i`'s manifest line with `f` (the JPEG stays as it is).
fn edit_manifest(data: &Path, f: impl Fn(usize, &mut serde_json::Value)) {
    let p = data.join("photos/manifest.jsonl");
    let text = fs::read_to_string(&p).unwrap();
    let mut out = String::new();
    for (i, l) in text.lines().enumerate() {
        let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
        f(i, &mut v);
        out.push_str(&serde_json::to_string(&v).unwrap());
        out.push('\n');
    }
    fs::write(&p, out).unwrap();
}

#[test]
fn cleanup_gallery_and_albums_dry_run() {
    let d = tmp("clean");
    let n = 120;
    let (vecs, _) = vectors(n, 5);
    add_rows(&d, 0, n, &vecs);
    let opts = ClusterOpts {
        run: "k-dedup".into(),
        mode: Mode::K(CENTERS),
        backend: CPU,
        dedup: Some(DedupConfig::default()),
        nepochs: 5.0,
        seed: 1,
        force: false,
    };
    let run = cluster::cluster(&d, &opts, &mut |_| {}).unwrap();
    // Row 39 has a larger stored copy than row 38; rows 48 and 49 are screenshots. Rows 68 and
    // 119 are favorites (i % 17 == 0), each in a stack of two.
    edit_manifest(&d, |i, v| {
        if i == 39 {
            v["width"] = 640.into();
            v["height"] = 480.into();
        }
        if i == 48 || i == 49 {
            v["subtypes"] = serde_json::json!(["screenshot"]);
        }
        if i == 58 {
            v["subtypes"] = serde_json::json!(["screenshot"]);
        }
    });
    let dd: emeans::DedupResult =
        serde_json::from_slice(&fs::read(run.join("dedup.json")).unwrap()).unwrap();
    let index = gallery::build(&d, "k-dedup", &GalleryOpts::default(), &mut |_| {}).unwrap();
    assert!(fs::read_to_string(&index).unwrap().contains("Clean up"));
    let v = page_data(&index);
    let c = &v["cleanup"];
    assert_eq!(
        c["anchors_stored"],
        serde_json::Value::Bool(dd.anchors.len() == dd.representatives.len())
    );
    let groups = c["groups"].as_array().unwrap();
    // Every 10th row is a near-copy of the one before: 12 stacks of 2.
    assert_eq!(groups.len(), n / 10);
    let u = |x: &serde_json::Value| x.as_u64().unwrap() as u32;
    let mut marked = Vec::new();
    for gr in groups {
        let g = u(&gr["g"]) as usize;
        let rows: Vec<u32> = gr["rows"].as_array().unwrap().iter().map(u).collect();
        let keep: Vec<u32> = gr["keep"].as_array().unwrap().iter().map(u).collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(u(&gr["anchor"]), dd.anchor(g));
        for &r in &rows {
            assert_eq!(dd.group_of[r as usize] as usize, g);
        }
        let hi = rows.iter().max().copied().unwrap();
        assert_eq!(hi % 10, 9, "a stack is a row and its near-copy");
        let expect = match hi {
            39 => 39,   // larger stored copy
            69 => 68,   // favorite
            119 => 119, // favorite
            _ => dd.anchor(g),
        };
        assert_eq!(keep, [expect], "group {g} rows {rows:?}");
        assert_eq!(rows[0], expect, "the keeper is ranked first");
        assert_eq!(gr["shots"].as_bool().unwrap(), hi == 49, "rows {rows:?}");
        marked.extend(rows.iter().filter(|r| !keep.contains(r)).map(|&r| (r, g)));
    }
    assert_eq!(marked.len(), n / 10);

    // cleanup.json as the page writes it; dry run only.
    let photos = v["photos"].as_array().unwrap();
    let list: Vec<serde_json::Value> = marked
        .iter()
        .map(|&(r, g)| {
            serde_json::json!({
                "asset_id": photos[r as usize][1],
                "file": photos[r as usize][5],
                "group": g,
                "reason": "near-duplicate",
            })
        })
        .collect();
    let path = d.join("cleanup.json");
    fs::write(&path, serde_json::to_vec(&list).unwrap()).unwrap();
    let mut lines = Vec::new();
    albums::run_cleanup(
        &d,
        &path,
        "emeans",
        albums::CLEANUP_TITLE,
        false,
        std::time::Duration::from_secs(1),
        &mut |l| lines.push(l.to_string()),
    )
    .unwrap();
    assert!(
        lines[0].starts_with(
            "Plan: one Photos folder \"emeans\" with 1 album \"emeans · to delete (review)\": \
             12 photos, from 12 near-duplicate groups"
        ),
        "{}",
        lines[0]
    );
    assert!(lines[1].starts_with("Nothing is deleted."));
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("Dry run: nothing was created"))
    );
    assert!(!lines.iter().any(|l| l.starts_with("Created")));
    // A file entry that does not match the manifest is refused.
    let mut bad = list.clone();
    bad[0]["file"] = "photos/other.jpg".into();
    fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
    let run_bad = |p: &Path| {
        albums::run_cleanup(
            &d,
            p,
            "emeans",
            albums::CLEANUP_TITLE,
            false,
            std::time::Duration::from_secs(1),
            &mut |_| {},
        )
    };
    assert!(run_bad(&path).is_err());
    fs::write(&path, "[]").unwrap();
    assert!(run_bad(&path).is_err());

    // albums --run: every cluster as an album, plus the duplicates album, without the gallery.
    let lists = albums::run_lists(
        &d,
        "k-dedup",
        &albums::RunLists {
            all_clusters: true,
            duplicates: true,
            duplicates_title: albums::DUPLICATES_TITLE.into(),
        },
    )
    .unwrap();
    let (dups, clusters) = lists.split_last().unwrap();
    assert_eq!(clusters.len(), CENTERS);
    let mut all: Vec<&String> = clusters.iter().flat_map(|a| &a.asset_ids).collect();
    all.sort();
    all.dedup();
    assert_eq!(
        all.len(),
        n,
        "every row, stack members included, in exactly one album"
    );
    assert_eq!(clusters.iter().map(|a| a.asset_ids.len()).sum::<usize>(), n);
    for a in clusters {
        // "<months> · <count>", with " #N" when the same title came earlier.
        let base = a.title.split(" #").next().unwrap();
        assert!(
            base.ends_with(&format!(" \u{b7} {}", a.asset_ids.len())),
            "{}",
            a.title
        );
    }
    let titles: std::collections::HashSet<&str> = lists.iter().map(|a| a.title.as_str()).collect();
    assert_eq!(titles.len(), lists.len(), "titles are unique");
    assert_eq!(dups.title, albums::DUPLICATES_TITLE);
    let mut want: Vec<String> = marked
        .iter()
        .map(|&(r, _)| photos[r as usize][1].as_str().unwrap().to_string())
        .collect();
    let mut got = dups.asset_ids.clone();
    want.sort();
    got.sort();
    assert_eq!(got, want, "duplicates = the non-keepers, never a keeper");
    let out = d.join("approved-from-run.json");
    let mut lines = Vec::new();
    albums::run_from_run(
        &d,
        "k-dedup",
        &albums::RunLists {
            all_clusters: true,
            duplicates: true,
            duplicates_title: albums::DUPLICATES_TITLE.into(),
        },
        "emeans",
        &out,
        &mut |l| lines.push(l.to_string()),
    )
    .unwrap();
    assert!(
        lines[0].starts_with("Plan: one Photos folder \"emeans\" with 7 albums"),
        "{}",
        lines[0]
    );
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("Dry run: nothing was created"))
    );
    // The written file is what `albums --approved` reads (dry run here).
    albums::run(
        &d,
        &out,
        "emeans",
        false,
        std::time::Duration::from_secs(1),
        &mut |_| {},
    )
    .unwrap();
    fs::remove_dir_all(&d).unwrap();
}
