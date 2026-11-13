//! Embed resume and alignment, with a deterministic fake model (no GPU) and synthetic JPEGs.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use encoder::Patches;
use photo_emeans::embed::{self, EmbedOpts, ImageEmbedder};
use photo_emeans::store::{self, D, EmbedStore, Meta};

/// Unit vector derived from the patch values: same image, same vector.
struct Fake {
    calls: usize,
    fail_on_call: Option<usize>,
}

impl ImageEmbedder for Fake {
    fn embed(&mut self, batch: &[Patches]) -> Result<Vec<[f32; D]>> {
        self.calls += 1;
        if Some(self.calls) == self.fail_on_call {
            bail!("simulated GPU fault");
        }
        Ok(batch
            .iter()
            .map(|p| {
                let mut h: u64 = 1469598103934665603;
                for v in p.data.iter().step_by(7) {
                    h = (h ^ v.to_bits() as u64).wrapping_mul(1099511628211);
                }
                let mut v = [0f32; D];
                for x in v.iter_mut() {
                    h ^= h << 13;
                    h ^= h >> 7;
                    h ^= h << 17;
                    *x = (h % 2001) as f32 / 1000.0 - 1.0;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                v
            })
            .collect())
    }
}

fn fake() -> Fake {
    Fake {
        calls: 0,
        fail_on_call: None,
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pe-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join("photos")).unwrap();
    d
}

/// Write `n` small JPEGs (ids p000..) and their manifest lines, starting at `from`.
fn add_photos(data: &Path, from: usize, n: usize) {
    let mut m = OpenOptions::new()
        .create(true)
        .append(true)
        .open(data.join("photos/manifest.jsonl"))
        .unwrap();
    for i in from..from + n {
        let id = format!("p{i:03}/L0/001");
        let file = format!("p{i:03}.jpg");
        let img = image::RgbImage::from_fn(64 + (i as u32 % 5) * 8, 48, |x, y| {
            image::Rgb([(x * 3 + i as u32 * 17) as u8, (y * 5) as u8, (i * 31) as u8])
        });
        img.save(data.join("photos").join(&file)).unwrap();
        writeln!(
            m,
            r#"{{"id":"{id}","file":"{file}","created":"2020-01-{:02}T10:00:00Z","width":{},"height":48,"favorite":false,"burst_id":null,"subtypes":[]}}"#,
            1 + i % 28,
            img.width()
        )
        .unwrap();
    }
}

fn meta() -> Meta {
    Meta::new("fake", "r1", "f32")
}

fn opts(chunk: usize) -> EmbedOpts {
    EmbedOpts {
        chunk,
        progress_every: 1000,
        ..EmbedOpts::default()
    }
}

fn run(data: &Path, emb: &mut Fake, o: &EmbedOpts) -> Result<embed::EmbedSummary> {
    let (mut st, _) = EmbedStore::open(&data.join("embed"), meta(), &mut |_| {})?;
    embed::run(&data.join("photos"), &mut st, emb, o, &mut |_| {})
}

fn by_id(data: &Path) -> HashMap<String, Vec<f32>> {
    let e = store::load(&data.join("embed")).unwrap();
    let rows = fs::read_to_string(data.join("embed/rows.jsonl")).unwrap();
    let bytes = fs::metadata(data.join("embed/embeddings.f32"))
        .unwrap()
        .len();
    assert_eq!(rows.lines().count(), e.n());
    assert_eq!(
        bytes,
        (e.n() * D * 4) as u64,
        "f32 file and rows.jsonl disagree"
    );
    let m: serde_json::Value =
        serde_json::from_slice(&fs::read(data.join("embed/meta.json")).unwrap()).unwrap();
    assert_eq!(m["n"].as_u64().unwrap() as usize, e.n());
    (0..e.n())
        .map(|i| (e.ids[i].clone(), e.row(i).to_vec()))
        .collect()
}

fn clean_reference(n: usize) -> HashMap<String, Vec<f32>> {
    let d = tmp(&format!("ref{n}"));
    add_photos(&d, 0, n);
    run(&d, &mut fake(), &opts(4)).unwrap();
    let r = by_id(&d);
    fs::remove_dir_all(&d).unwrap();
    r
}

#[test]
fn full_run_aligned_and_skips_corrupt_file() {
    let d = tmp("full");
    add_photos(&d, 0, 10);
    fs::write(d.join("photos/p003.jpg"), b"not a jpeg").unwrap();
    let s = run(&d, &mut fake(), &opts(3)).unwrap();
    assert_eq!((s.embedded, s.failed, s.total, s.remaining), (9, 1, 9, 0));
    let got = by_id(&d);
    assert!(!got.contains_key("p003/L0/001"));
    let f = store::read_failures(&d.join("embed")).unwrap();
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].id, "p003/L0/001");
    // Second run: nothing to do, the failure stays skipped.
    let s = run(&d, &mut fake(), &opts(3)).unwrap();
    assert_eq!((s.embedded, s.failed, s.total), (0, 0, 9));
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn rows_order_and_numbering() {
    let d = tmp("order");
    add_photos(&d, 0, 7);
    run(&d, &mut fake(), &opts(2)).unwrap();
    let rows = fs::read_to_string(d.join("embed/rows.jsonl")).unwrap();
    for (i, l) in rows.lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        assert_eq!(v["row"].as_u64().unwrap() as usize, i);
        assert_eq!(v["id"].as_str().unwrap(), format!("p{i:03}/L0/001"));
    }
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn crash_after_f32_write_is_repaired_and_resumes_to_clean_result() {
    let reference = clean_reference(12);
    let d = tmp("crash-f32");
    add_photos(&d, 0, 12);
    run(
        &d,
        &mut fake(),
        &EmbedOpts {
            limit: Some(5),
            ..opts(2)
        },
    )
    .unwrap();
    // Simulate a kill after the f32 append of the next chunk but before its rows lines:
    // two whole rows plus a partial third.
    let mut f = OpenOptions::new()
        .append(true)
        .open(d.join("embed/embeddings.f32"))
        .unwrap();
    f.write_all(&vec![0x3fu8; 2 * D * 4 + 100]).unwrap();
    drop(f);
    let mut logs = Vec::new();
    let (st, rep) =
        EmbedStore::open(&d.join("embed"), meta(), &mut |m| logs.push(m.to_string())).unwrap();
    assert_eq!(st.n(), 5);
    assert_eq!(
        (
            rep.rows_dropped,
            rep.partial_bytes,
            rep.lines_dropped,
            rep.torn_line
        ),
        (2, 100, 0, false)
    );
    assert!(logs[0].starts_with("!!!"), "repair must be logged loudly");
    drop(st);
    run(&d, &mut fake(), &opts(2)).unwrap();
    assert_eq!(by_id(&d), reference);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn torn_rows_line_is_repaired_and_resumes_to_clean_result() {
    let reference = clean_reference(9);
    let d = tmp("crash-rows");
    add_photos(&d, 0, 9);
    run(&d, &mut fake(), &opts(3)).unwrap();
    // Cut rows.jsonl in the middle of line 7: lines 0..6 stay, line 6 is torn.
    let rows = fs::read_to_string(d.join("embed/rows.jsonl")).unwrap();
    let starts: Vec<usize> = std::iter::once(0)
        .chain(rows.match_indices('\n').map(|(i, _)| i + 1))
        .collect();
    fs::write(d.join("embed/rows.jsonl"), &rows[..starts[6] + 5]).unwrap();
    let (st, rep) = EmbedStore::open(&d.join("embed"), meta(), &mut |_| {}).unwrap();
    assert_eq!(st.n(), 6);
    assert!(rep.torn_line);
    assert_eq!(rep.rows_dropped, 3);
    drop(st);
    let s = run(&d, &mut fake(), &opts(3)).unwrap();
    assert_eq!(s.embedded, 3);
    assert_eq!(by_id(&d), reference);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn model_error_is_fatal_but_keeps_finished_rows() {
    let reference = clean_reference(10);
    let d = tmp("gpu-fault");
    add_photos(&d, 0, 10);
    let err = run(
        &d,
        &mut Fake {
            calls: 0,
            fail_on_call: Some(6),
        },
        &opts(4),
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("simulated GPU fault"));
    assert_eq!(
        by_id(&d).len(),
        5,
        "the 5 vectors computed before the fault are committed"
    );
    run(&d, &mut fake(), &opts(4)).unwrap();
    assert_eq!(by_id(&d), reference);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn growing_manifest_and_follow() {
    let d = tmp("grow");
    add_photos(&d, 0, 4);
    assert_eq!(run(&d, &mut fake(), &opts(2)).unwrap().total, 4);
    // The fetch appends more photos and a torn last line.
    add_photos(&d, 4, 3);
    let mut m = OpenOptions::new()
        .append(true)
        .open(d.join("photos/manifest.jsonl"))
        .unwrap();
    m.write_all(b"{\"id\":\"p9").unwrap();
    drop(m);
    fs::write(
        d.join("fetch.log"),
        "fetched 7 (from iCloud 0), skipped 0\n",
    )
    .unwrap();
    let o = EmbedOpts {
        follow: true,
        fetch_log: Some(d.join("fetch.log")),
        poll: std::time::Duration::from_millis(10),
        ..opts(2)
    };
    let s = run(&d, &mut fake(), &o).unwrap();
    assert_eq!((s.embedded, s.total), (3, 7));
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn refuses_mixing_models_corruption_and_second_writer() {
    let d = tmp("refuse");
    add_photos(&d, 0, 3);
    run(&d, &mut fake(), &opts(2)).unwrap();
    let dir = d.join("embed");
    // Another compute dtype.
    let e = EmbedStore::open(&dir, Meta::new("fake", "r1", "bf16"), &mut |_| {})
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("refusing to mix"));
    // A second writer while the first holds the lock.
    let (first, _) = EmbedStore::open(&dir, meta(), &mut |_| {}).unwrap();
    let e = EmbedStore::open(&dir, meta(), &mut |_| {}).err().unwrap();
    assert!(format!("{e:#}").contains("another embed run"));
    drop(first);
    // A non-unit row is corruption, not a crash artifact.
    let mut b = fs::read(dir.join("embeddings.f32")).unwrap();
    b[D * 4..D * 4 + 4].copy_from_slice(&5.0f32.to_le_bytes());
    fs::write(dir.join("embeddings.f32"), &b).unwrap();
    assert!(EmbedStore::open(&dir, meta(), &mut |_| {}).is_err());
    assert!(store::load(&dir).is_err());
    // A bad line in the middle of rows.jsonl.
    run_fresh_then(&d, |dir| {
        let rows = fs::read_to_string(dir.join("rows.jsonl")).unwrap();
        fs::write(
            dir.join("rows.jsonl"),
            rows.replacen("\"row\":1", "\"row\":7", 1),
        )
        .unwrap();
    });
    assert!(EmbedStore::open(&dir, meta(), &mut |_| {}).is_err());
    // Data without meta.json.
    run_fresh_then(&d, |dir| fs::remove_file(dir.join("meta.json")).unwrap());
    assert!(EmbedStore::open(&dir, meta(), &mut |_| {}).is_err());
    fs::remove_dir_all(&d).unwrap();
}

fn run_fresh_then(d: &Path, f: impl FnOnce(&Path)) {
    let dir = d.join("embed");
    fs::remove_dir_all(&dir).unwrap();
    run(d, &mut fake(), &opts(2)).unwrap();
    f(&dir);
}

#[test]
fn reader_accepts_f32_ahead_of_rows() {
    let d = tmp("reader");
    add_photos(&d, 0, 4);
    run(&d, &mut fake(), &opts(2)).unwrap();
    let before = store::load(&d.join("embed")).unwrap();
    let mut f = OpenOptions::new()
        .append(true)
        .open(d.join("embed/embeddings.f32"))
        .unwrap();
    f.write_all(&vec![0u8; D * 4]).unwrap(); // an embed run in the middle of a chunk
    drop(f);
    let after = store::load(&d.join("embed")).unwrap();
    assert_eq!(after.ids, before.ids);
    assert_eq!(after.x, before.x);
    // rows ahead of f32 is misalignment: readers refuse.
    let b = fs::read(d.join("embed/embeddings.f32")).unwrap();
    fs::write(d.join("embed/embeddings.f32"), &b[..2 * D * 4]).unwrap();
    assert!(store::load(&d.join("embed")).is_err());
    fs::remove_dir_all(&d).unwrap();
}
