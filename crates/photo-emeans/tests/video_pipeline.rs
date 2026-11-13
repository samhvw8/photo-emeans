//! Videos end to end on generated sample media (never the user's library):
//! ffmpeg makes three short clips (one with a 90° display rotation), AVFoundation decodes their
//! frames (`photos::frames_from_file`, the same generator and JPEG path as `fetch-videos`), the
//! frames go into `videos/` like `fetch-videos` writes them, a fake model embeds photos and
//! frames, then `cluster`, `gallery` and `albums` (dry run) run on photos and videos together. A
//! run made before the videos were added must still build its gallery.
//!
//! Needs `ffmpeg` on PATH; without it the test prints SKIPPED and passes.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use emeans::{Backend, CpuKernel, DedupConfig};
use encoder::Patches;
use photo_emeans::albums;
use photo_emeans::cluster::{self, ClusterOpts, Mode};
use photo_emeans::embed::{self, EmbedOpts, GpuLock, ImageEmbedder};
use photo_emeans::gallery::{self, GalleryOpts};
use photo_emeans::store::{self, D, EmbedStore, Kind, Meta};
use photos::VideoLine;
use photos::video::{frame_count, sample_times};
use photos::video_store::write_video_dir;

/// Vector from the mean colour of the image, so similar-looking images get similar vectors.
struct ColourModel;

impl ImageEmbedder for ColourModel {
    fn embed(&mut self, batch: &[Patches]) -> Result<Vec<[f32; D]>> {
        Ok(batch
            .iter()
            .map(|p| {
                let mut c = [0f64; 3];
                for (i, v) in p.data.iter().enumerate() {
                    c[i % 3] += f64::from(*v);
                }
                let mut v = [0f32; D];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = (c[j % 3] / p.data.len() as f64) as f32 + 0.01 * ((j * 7919) % 13) as f32;
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                v
            })
            .collect())
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pe-vpipe-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(d.join("photos")).unwrap();
    fs::create_dir_all(d.join("videos")).unwrap();
    fs::create_dir_all(d.join("media")).unwrap();
    d
}

fn ffmpeg(args: &[&str]) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .status()
        .is_ok_and(|s| s.success())
}

/// Photos 0..n: JPEG + manifest line, colours in two families.
fn add_photos(data: &Path, n: usize) {
    let mut m = OpenOptions::new()
        .create(true)
        .append(true)
        .open(data.join("photos/manifest.jsonl"))
        .unwrap();
    for i in 0..n {
        let red = i % 2 == 0;
        let img = image::RgbImage::from_fn(96, 64, |x, y| {
            let n = ((x * 7 + y * 3 + i as u32 * 5) % 20) as u8;
            if red {
                image::Rgb([200 + n, 30 + n, 30])
            } else {
                image::Rgb([30, 60 + n, 200 + n])
            }
        });
        let file = format!("P{i:03}-AAAA.jpg");
        img.save(data.join("photos").join(&file)).unwrap();
        writeln!(
            m,
            r#"{{"id":"P{i:03}-AAAA/L0/001","file":"{file}","created":"2021-05-{:02}T10:00:00Z","width":96,"height":64,"favorite":false,"burst_id":null,"subtypes":[]}}"#,
            1 + i % 28
        )
        .unwrap();
    }
}

struct Clip {
    stem: &'static str,
    duration: f64,
    rotated: bool,
}

/// Decode a clip like `fetch-videos` (poster = the frame at 0 s here) and write it into `videos/`.
fn add_clip(data: &Path, file: &Path, c: &Clip, created: &str) -> VideoLine {
    let n = frame_count(c.duration, 8, 4.0);
    let mut times = vec![0.0];
    times.extend(sample_times(c.duration, n - 1));
    let frames = photos::frames_from_file(file, &times, 1024, 0.88).unwrap();
    assert_eq!(frames.len(), n);
    for f in &frames {
        assert_eq!(
            f.height > f.width,
            c.rotated,
            "{}: frame is {}x{}; the preferred track transform must be applied",
            c.stem,
            f.width,
            f.height
        );
        assert!(f.width.max(f.height) <= 1024);
    }
    let parts: Vec<(&[u8], u32, u32, Option<f64>)> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            (
                f.jpeg.as_slice(),
                f.width,
                f.height,
                (i > 0).then_some(f.t.unwrap()),
            )
        })
        .collect();
    let fl = write_video_dir(&data.join("videos"), c.stem, &parts).unwrap();
    let line = VideoLine {
        id: format!("{}/L0/001", c.stem),
        created: Some(created.into()),
        duration: c.duration,
        width: fl[0].width,
        height: fl[0].height,
        favorite: false,
        subtypes: vec![],
        frames: fl,
    };
    let mut m = OpenOptions::new()
        .create(true)
        .append(true)
        .open(data.join("videos/manifest.jsonl"))
        .unwrap();
    writeln!(m, "{}", serde_json::to_string(&line).unwrap()).unwrap();
    line
}

fn page_data(index: &Path) -> serde_json::Value {
    let html = fs::read_to_string(index).unwrap();
    let start = html
        .find(r#"<script id="data" type="application/json">"#)
        .unwrap();
    let rest = &html[start..];
    let body = &rest[rest.find('>').unwrap() + 1..rest.find("</script>").unwrap()];
    serde_json::from_str(&body.replace("<\\/", "</")).unwrap()
}

const CPU: Backend = Backend::Cpu(CpuKernel::Accelerate);

#[test]
fn sample_videos_through_embed_cluster_gallery_albums() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIPPED: ffmpeg not found on PATH");
        return;
    }
    let d = tmp("e2e");
    let media = d.join("media");
    // Three clips: 6 s red-ish test pattern, 3 s blue bars shown rotated 90°, 1.5 s fractal.
    let a = media.join("a.mp4");
    let b0 = media.join("b0.mp4");
    let b = media.join("b.mp4");
    let c = media.join("c.mov");
    assert!(ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30:duration=6",
        "-vf",
        "colorchannelmixer=rr=1:gg=0.2:bb=0.2",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        a.to_str().unwrap(),
    ]));
    assert!(ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "smptebars=size=480x270:rate=25:duration=3",
        "-vf",
        "colorchannelmixer=rr=0.2:gg=0.3:bb=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        b0.to_str().unwrap(),
    ]));
    assert!(ffmpeg(&[
        "-display_rotation",
        "90",
        "-i",
        b0.to_str().unwrap(),
        "-c",
        "copy",
        b.to_str().unwrap(),
    ]));
    assert!(ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "mandelbrot=size=320x240:rate=25",
        "-t",
        "1.5",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        c.to_str().unwrap(),
    ]));

    // Photos first, embedded and clustered before any video exists.
    let n_photos = 40;
    add_photos(&d, n_photos);
    let meta = Meta::new("colour", "1", "f32");
    {
        let (mut st, _) = EmbedStore::open(&d.join("embed"), meta.clone(), &mut |_| {}).unwrap();
        let s = embed::run(
            &d.join("photos"),
            &mut st,
            &mut ColourModel,
            &EmbedOpts::default(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(s.embedded, n_photos);
    }
    let before = ClusterOpts {
        run: "before".into(),
        mode: Mode::K(2),
        backend: CPU,
        dedup: None,
        nepochs: 5.0,
        seed: 1,
        force: false,
    };
    cluster::cluster(&d, &before, &mut |_| {}).unwrap();

    // Videos: decode, write, embed.
    let clips = [
        Clip {
            stem: "VA00-AAAA",
            duration: 6.0,
            rotated: false,
        },
        Clip {
            stem: "VB00-AAAA",
            duration: 3.0,
            rotated: true,
        },
        Clip {
            stem: "VC00-AAAA",
            duration: 1.5,
            rotated: false,
        },
    ];
    let files = [&a, &b, &c];
    let lines: Vec<VideoLine> = clips
        .iter()
        .zip(files)
        .enumerate()
        .map(|(i, (cl, f))| add_clip(&d, f, cl, &format!("2021-06-0{}T10:00:00Z", i + 1)))
        .collect();
    assert_eq!(
        lines.iter().map(|l| l.frames.len()).collect::<Vec<_>>(),
        [8, 6, 3]
    );
    let (mut main, _) = EmbedStore::open(&d.join("embed"), meta.clone(), &mut |_| {}).unwrap();
    let (mut frames, _) =
        EmbedStore::open(&d.join("embed/frames"), meta.clone(), &mut |_| {}).unwrap();
    let s = embed::run_videos(
        &d.join("videos"),
        &mut main,
        &mut frames,
        &mut ColourModel,
        &EmbedOpts::default(),
        &mut GpuLock::none(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!((s.frames, s.videos, s.failed), (17, 3, 0));
    drop((main, frames));
    let emb = store::load(&d.join("embed")).unwrap();
    assert_eq!(emb.n(), n_photos + 3);
    assert_eq!(emb.kinds.iter().filter(|&&k| k == Kind::Video).count(), 3);

    // The run made before the videos still builds its gallery (prefix of the store).
    let old = gallery::build(&d, "before", &GalleryOpts::default(), &mut |_| {}).unwrap();
    let ov = page_data(&old);
    assert_eq!(ov["photos"].as_array().unwrap().len(), n_photos);
    assert_eq!(ov["n_videos"], 0);

    // Photos and videos together, with dedup and a time rule.
    let opts = ClusterOpts {
        run: "av".into(),
        mode: Mode::K(2),
        backend: CPU,
        dedup: Some(DedupConfig {
            max_gap_secs: Some(600.0),
            ..DedupConfig::default()
        }),
        nepochs: 5.0,
        seed: 1,
        force: false,
    };
    let run = cluster::cluster(&d, &opts, &mut |_| {}).unwrap();
    let rep = cluster::read_report(&run).unwrap();
    assert_eq!((rep.n, rep.n_videos), (n_photos + 3, 3));
    assert_eq!(
        cluster::read_i32(&run.join("labels.i32")).unwrap().len(),
        n_photos + 3
    );

    let index = gallery::build(&d, "av", &GalleryOpts::default(), &mut |_| {}).unwrap();
    let v = page_data(&index);
    assert_eq!(v["n_videos"], 3);
    let rows = v["photos"].as_array().unwrap();
    assert_eq!(rows.len(), n_photos + 3);
    for (i, l) in lines.iter().enumerate() {
        let r = &rows[n_photos + i];
        assert_eq!(r[1], l.id.as_str());
        assert_eq!(r[0], format!("v-{}.jpg", clips[i].stem).as_str());
        assert_eq!(r[5], format!("videos/{}/f00.jpg", clips[i].stem).as_str());
        assert_eq!(r[6][0].as_f64().unwrap(), clips[i].duration);
        assert_eq!(r[6][1].as_array().unwrap().len(), l.frames.len());
        let thumb = fs::read(
            index
                .parent()
                .unwrap()
                .join("thumbs")
                .join(r[0].as_str().unwrap()),
        )
        .unwrap();
        let (w, h) = photos::store::jpeg_size(&thumb).unwrap();
        assert_eq!(w.max(h), 256);
        assert_eq!(h > w, clips[i].rotated);
        // Paths in the page resolve from gallery/ (../../../ = $EMEANS_DATA).
        for f in r[6][1].as_array().unwrap() {
            assert!(d.join(f.as_str().unwrap()).is_file());
        }
    }
    assert!(rows[0][6].is_null() && rows[0][5] == "photos/P000-AAAA.jpg");
    let html = fs::read_to_string(&index).unwrap();
    assert!(html.contains("class=\"strip\"") && html.contains("\"play\""));

    // approved.json as the page writes it: every row of the first card plus one video.
    let card = &v["cards"].as_array().unwrap()[0];
    let mut ids: Vec<String> = card["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            rows[r.as_u64().unwrap() as usize][1]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    if !ids.contains(&lines[1].id) {
        ids.push(lines[1].id.clone());
    }
    let approved = serde_json::json!([
        {"cluster": card["key"], "title": "Mixed", "asset_ids": ids},
        {"cluster": "v", "title": "Clips", "asset_ids": [lines[0].id, lines[2].id]},
    ]);
    let path = d.join("approved.json");
    fs::write(&path, serde_json::to_vec(&approved).unwrap()).unwrap();
    let mut out = Vec::new();
    albums::run(
        &d,
        &path,
        "emeans",
        false,
        std::time::Duration::from_secs(1),
        &mut |l| out.push(l.to_string()),
    )
    .unwrap();
    assert!(out[0].contains("videos)"), "{}", out[0]);
    assert!(
        out.iter()
            .any(|l| l.contains("Clips") && l.contains("(incl. 2 videos)"))
    );
    assert!(
        out.iter()
            .any(|l| l.starts_with("Dry run: nothing was created"))
    );
    fs::remove_dir_all(&d).unwrap();
}
