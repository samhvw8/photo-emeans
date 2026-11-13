//! Video embedding: frames into `embed/frames/`, one mean row per video into `embed/`, resume
//! after a kill at either write, a broken frame, photos and videos in one store, and the GPU lock.
//! Deterministic fake model (no GPU), synthetic JPEG frames.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use encoder::Patches;
use photo_emeans::embed::{self, EmbedOpts, GpuLock, ImageEmbedder, frame_id, mean_unit};
use photo_emeans::manifest;
use photo_emeans::store::{self, D, EmbedStore, Kind, Meta};
use photos::video_store::write_video_dir;
use photos::{VideoLine, store::jpeg_size};

struct Fake {
    images: usize,
}

impl ImageEmbedder for Fake {
    fn embed(&mut self, batch: &[Patches]) -> Result<Vec<[f32; D]>> {
        self.images += batch.len();
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

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pe-video-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join("videos")).unwrap();
    d
}

fn jpeg(seed: u32, w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            (x * 3 + seed * 37) as u8,
            (y * 5 + seed * 11) as u8,
            (seed * 53) as u8,
        ])
    });
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90)
        .encode_image(&img)
        .unwrap();
    buf
}

/// Video `v` with `n` frames: frame dirs written like `fetch-videos` does, manifest line appended.
fn add_video(data: &Path, v: usize, n: usize) -> VideoLine {
    let dir = data.join("videos");
    let stem = format!("V{v:03}-AAAA");
    let imgs: Vec<Vec<u8>> = (0..n).map(|i| jpeg((v * 10 + i) as u32, 64, 48)).collect();
    let frames: Vec<(&[u8], u32, u32, Option<f64>)> = imgs
        .iter()
        .enumerate()
        .map(|(i, j)| {
            (
                j.as_slice(),
                64,
                48,
                if i == 0 { None } else { Some(i as f64) },
            )
        })
        .collect();
    let fl = write_video_dir(&dir, &stem, &frames).unwrap();
    assert_eq!(jpeg_size(&imgs[0]), Some((64, 48)));
    let line = VideoLine {
        id: format!("{stem}/L0/001"),
        created: Some(format!("2021-03-{:02}T10:00:00Z", 1 + v)),
        duration: n as f64,
        width: 64,
        height: 48,
        favorite: v == 0,
        subtypes: vec![],
        frames: fl,
    };
    let mut m = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("manifest.jsonl"))
        .unwrap();
    writeln!(m, "{}", serde_json::to_string(&line).unwrap()).unwrap();
    line
}

fn meta() -> Meta {
    Meta::new("fake", "r1", "f32")
}

fn opts(chunk: usize) -> EmbedOpts {
    EmbedOpts {
        chunk,
        batch: 1,
        progress_every: 1000,
        ..EmbedOpts::default()
    }
}

fn run(data: &Path, emb: &mut Fake, o: &EmbedOpts) -> Result<embed::VideoSummary> {
    let (mut main, _) = EmbedStore::open(&data.join("embed"), meta(), &mut |_| {})?;
    let (mut frames, _) = EmbedStore::open(&data.join("embed/frames"), meta(), &mut |_| {})?;
    embed::run_videos(
        &data.join("videos"),
        &mut main,
        &mut frames,
        emb,
        o,
        &mut GpuLock::none(),
        &mut |_| {},
    )
}

/// Every video row in the main store must be the normalized mean of its stored frames.
fn check_means(data: &Path, videos: &[VideoLine]) -> usize {
    let main = store::load(&data.join("embed")).unwrap();
    let frames = store::load(&data.join("embed/frames")).unwrap();
    assert!(frames.kinds.iter().all(|&k| k == Kind::Frame));
    let fr: HashMap<&str, usize> = frames
        .ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let mut checked = 0;
    for (i, id) in main.ids.iter().enumerate() {
        if main.kinds[i] != Kind::Video {
            continue;
        }
        let v = videos.iter().find(|v| &v.id == id).unwrap();
        let fv: Vec<[f32; D]> = (0..v.frames.len())
            .map(|k| frames.row(fr[frame_id(id, k).as_str()]).try_into().unwrap())
            .collect();
        let want = mean_unit(&fv.iter().collect::<Vec<_>>()).unwrap();
        let got = main.row(i);
        let maxdiff = want
            .iter()
            .zip(got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(maxdiff < 1e-6, "video {id}: row differs by {maxdiff}");
        checked += 1;
    }
    checked
}

#[test]
fn videos_become_mean_rows() {
    let d = tmp("mean");
    let vids: Vec<VideoLine> = (0..4).map(|v| add_video(&d, v, 2 + v)).collect();
    let mut f = Fake { images: 0 };
    let s = run(&d, &mut f, &opts(3)).unwrap();
    assert_eq!((s.frames, s.videos, s.failed, s.remaining), (14, 4, 0, 0));
    assert_eq!(f.images, 14);
    assert_eq!(check_means(&d, &vids), 4);
    let rows = fs::read_to_string(d.join("embed/rows.jsonl")).unwrap();
    assert!(
        rows.lines().all(|l| l.ends_with(r#""kind":"video"}"#)),
        "{rows}"
    );
    // Nothing left: a second run embeds nothing.
    let mut f = Fake { images: 0 };
    let s = run(&d, &mut f, &opts(3)).unwrap();
    assert_eq!((s.frames, s.videos, f.images), (0, 0, 0));
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn resume_after_limit_and_after_kill_between_stores() {
    let d = tmp("resume");
    let vids: Vec<VideoLine> = (0..3).map(|v| add_video(&d, v, 4)).collect();
    // A run stopped after 5 frames: video 0 is complete, video 1 has one frame.
    let s = run(
        &d,
        &mut Fake { images: 0 },
        &EmbedOpts {
            limit: Some(5),
            ..opts(2)
        },
    )
    .unwrap();
    assert_eq!((s.frames, s.videos, s.remaining), (5, 1, 2));
    // A kill between the frame commit and the video row: put video 2's frames in the frame
    // store by hand, as if the process died before appending its main row.
    {
        let (mut frames, _) =
            EmbedStore::open(&d.join("embed/frames"), meta(), &mut |_| {}).unwrap();
        let mut f = Fake { images: 0 };
        let rows: Vec<(String, [f32; D])> = vids[2]
            .frames
            .iter()
            .enumerate()
            .map(|(k, fr)| {
                let p = embed::prepare(&d.join("videos").join(&fr.file)).unwrap();
                (frame_id(&vids[2].id, k), f.embed(&[p]).unwrap()[0])
            })
            .collect();
        frames.append(&rows, Kind::Frame).unwrap();
    }
    let mut f = Fake { images: 0 };
    let s = run(&d, &mut f, &opts(2)).unwrap();
    // Video 2 needs no model call; video 1 needs its last 3 frames.
    assert_eq!((s.frames, s.videos, s.remaining), (3, 2, 0));
    assert_eq!(f.images, 3);
    assert_eq!(check_means(&d, &vids), 3);
    // Same rows as a clean run (resume does not change any vector).
    let clean = tmp("resume-clean");
    let cv: Vec<VideoLine> = (0..3).map(|v| add_video(&clean, v, 4)).collect();
    run(&clean, &mut Fake { images: 0 }, &opts(2)).unwrap();
    let a = store::load(&d.join("embed")).unwrap();
    let b = store::load(&clean.join("embed")).unwrap();
    for (i, id) in a.ids.iter().enumerate() {
        let j = b.ids.iter().position(|x| x == id).unwrap();
        assert_eq!(a.row(i), b.row(j), "{id}");
    }
    assert_eq!(cv.len(), 3);
    fs::remove_dir_all(&d).unwrap();
    fs::remove_dir_all(&clean).unwrap();
}

#[test]
fn broken_frame_skips_the_video() {
    let d = tmp("broken");
    let vids: Vec<VideoLine> = (0..3).map(|v| add_video(&d, v, 3)).collect();
    fs::write(
        d.join("videos").join(&vids[1].frames[1].file),
        b"not a jpeg",
    )
    .unwrap();
    let s = run(&d, &mut Fake { images: 0 }, &opts(2)).unwrap();
    assert_eq!((s.videos, s.failed, s.remaining), (2, 1, 0));
    let fails = store::read_failures(&d.join("embed")).unwrap();
    assert_eq!(fails.len(), 1);
    assert_eq!(fails[0].id, vids[1].id);
    assert_eq!(fails[0].file, format!("videos/{}", vids[1].frames[1].file));
    let main = store::load(&d.join("embed")).unwrap();
    assert!(!main.ids.contains(&vids[1].id));
    // Later runs skip it; --retry-failed tries again (and fails again).
    let s = run(&d, &mut Fake { images: 0 }, &opts(2)).unwrap();
    assert_eq!((s.frames, s.videos, s.failed), (0, 0, 0));
    let s = run(
        &d,
        &mut Fake { images: 0 },
        &EmbedOpts {
            retry_failed: true,
            ..opts(2)
        },
    )
    .unwrap();
    assert_eq!(s.failed, 1);
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn old_rows_without_kind_read_as_photos() {
    let d = tmp("kinds");
    fs::create_dir_all(d.join("embed")).unwrap();
    {
        let (mut st, _) = EmbedStore::open(&d.join("embed"), meta(), &mut |_| {}).unwrap();
        let mut v = [0f32; D];
        v[0] = 1.0;
        st.append(&[("p0/L0/001".into(), v)], Kind::Photo).unwrap();
    }
    // Rewrite the line as an old store wrote it (no kind).
    fs::write(
        d.join("embed/rows.jsonl"),
        "{\"row\":0,\"id\":\"p0/L0/001\"}\n",
    )
    .unwrap();
    add_video(&d, 0, 2);
    let s = run(&d, &mut Fake { images: 0 }, &opts(8)).unwrap();
    assert_eq!(s.videos, 1);
    let e = store::load(&d.join("embed")).unwrap();
    assert_eq!(e.kinds, vec![Kind::Photo, Kind::Video]);
    let rows = fs::read_to_string(d.join("embed/rows.jsonl")).unwrap();
    assert_eq!(
        rows.lines().next().unwrap(),
        "{\"row\":0,\"id\":\"p0/L0/001\"}"
    );
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn media_view_joins_photos_and_videos() {
    let d = tmp("media");
    fs::create_dir_all(d.join("photos")).unwrap();
    fs::write(
        d.join("photos/manifest.jsonl"),
        "{\"id\":\"P/L0/001\",\"file\":\"P.jpg\",\"created\":null,\"width\":4,\"height\":3,\"favorite\":false,\"burst_id\":\"b\",\"subtypes\":[]}\n",
    )
    .unwrap();
    let v = add_video(&d, 0, 3);
    let m = manifest::media_by_id(&d).unwrap();
    assert_eq!(m.len(), 2);
    assert_eq!(m["P/L0/001"].kind, Kind::Photo);
    assert_eq!(m["P/L0/001"].image, "photos/P.jpg");
    let mv = &m[&v.id];
    assert_eq!(mv.kind, Kind::Video);
    assert_eq!(mv.image, "videos/V000-AAAA/f00.jpg");
    assert_eq!(mv.frames.len(), 3);
    assert_eq!(mv.duration, 3.0);
    assert!(mv.favorite && mv.burst_id.is_none());
    // The same id in both manifests is refused.
    let mut text = fs::read_to_string(d.join("photos/manifest.jsonl")).unwrap();
    text += &format!(
        "{{\"id\":\"{}\",\"file\":\"X.jpg\",\"created\":null,\"width\":4,\"height\":3,\"favorite\":false,\"burst_id\":null,\"subtypes\":[]}}\n",
        v.id
    );
    fs::write(d.join("photos/manifest.jsonl"), text).unwrap();
    assert!(manifest::media_by_id(&d).is_err());
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn gpu_lock_is_released_between_chunks() {
    let d = tmp("lock");
    (0..3).for_each(|v| {
        add_video(&d, v, 4);
    });
    let lock_path = d.join("gpu.lock");
    let (mut main, _) = EmbedStore::open(&d.join("embed"), meta(), &mut |_| {}).unwrap();
    let (mut frames, _) = EmbedStore::open(&d.join("embed/frames"), meta(), &mut |_| {}).unwrap();
    let mut lock = GpuLock::new(&lock_path, 4).unwrap();
    lock.pause = std::time::Duration::from_millis(1);
    // A model that checks, on every call, that the lock is held by us and not by another handle.
    struct Probe<'a> {
        inner: Fake,
        path: &'a Path,
        calls: usize,
    }
    impl ImageEmbedder for Probe<'_> {
        fn embed(&mut self, batch: &[Patches]) -> Result<Vec<[f32; D]>> {
            let other = fs::File::open(self.path)?;
            assert!(
                other.try_lock().is_err(),
                "model ran without the GPU lock held"
            );
            self.calls += 1;
            self.inner.embed(batch)
        }
    }
    let mut p = Probe {
        inner: Fake { images: 0 },
        path: &lock_path,
        calls: 0,
    };
    let s = embed::run_videos(
        &d.join("videos"),
        &mut main,
        &mut frames,
        &mut p,
        &opts(2),
        &mut lock,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!((s.frames, s.videos, p.calls), (12, 3, 12));
    // After the run the lock is free for others.
    assert!(!lock.is_held());
    fs::File::open(&lock_path).unwrap().try_lock().unwrap();
    fs::remove_dir_all(&d).unwrap();
}
