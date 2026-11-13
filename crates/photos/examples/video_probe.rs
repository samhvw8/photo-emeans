//! Count the library's videos before fetching any: number, durations, original sizes, and how
//! many have the requested rendition on this Mac (network-off `requestAVAssetForVideo`).
//!
//! cargo run -p photos --example video_probe -- [--check-local] [--fast]
//!
//! Read-only: downloads nothing, writes nothing. Prints counts only, no asset ids.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use photos::{Rendition, VideoConfig, video_is_local};

#[derive(Parser)]
struct Args {
    /// Ask Photos (network off) whether each video's rendition is on this Mac.
    #[arg(long)]
    check_local: bool,
    /// Check the fast rendition instead of the medium one.
    #[arg(long)]
    fast: bool,
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let args = Args::parse();
    photos::ensure_authorized(Duration::from_secs(60))?;
    let t = Instant::now();
    let vids = photos::list_videos()?;
    println!(
        "listed {} videos in {:.2}s",
        vids.len(),
        t.elapsed().as_secs_f64()
    );
    let mut d: Vec<f64> = vids.iter().map(|v| v.duration).collect();
    d.sort_by(f64::total_cmp);
    let total: f64 = d.iter().sum();
    println!(
        "duration: total {:.1} h, median {:.1} s, p10 {:.1} s, p90 {:.1} s, max {:.0} s",
        total / 3600.0,
        pct(&d, 0.5),
        pct(&d, 0.1),
        pct(&d, 0.9),
        d.last().copied().unwrap_or(0.0)
    );
    for (lo, hi) in [
        (0.0, 4.0),
        (4.0, 15.0),
        (15.0, 60.0),
        (60.0, 300.0),
        (300.0, f64::MAX),
    ] {
        println!(
            "  {lo:>5.0}..{:<5} s: {}",
            if hi == f64::MAX {
                "".into()
            } else {
                format!("{hi:.0}")
            },
            d.iter().filter(|&&x| x >= lo && x < hi).count()
        );
    }
    let frames: usize = vids
        .iter()
        .map(|v| photos::video::frame_count(v.duration, 8, 4.0))
        .sum();
    println!("frames at K = 8 (fewer under 4 s): {frames}");
    let mut by_sub: BTreeMap<String, usize> = BTreeMap::new();
    for v in &vids {
        for s in &v.subtypes {
            *by_sub.entry(s.clone()).or_default() += 1;
        }
    }
    println!("by subtype: {by_sub:?}");
    let mut res: BTreeMap<String, usize> = BTreeMap::new();
    for v in &vids {
        let long = v.pixel_width.max(v.pixel_height);
        let k = match long {
            0..=720 => "<=720",
            721..=1280 => "<=1280",
            1281..=1920 => "<=1920",
            1921..=3840 => "<=3840",
            _ => ">3840",
        };
        *res.entry(k.into()).or_default() += 1;
    }
    println!("long side px: {res:?}");
    println!(
        "favorites {}, undated {}",
        vids.iter().filter(|v| v.favorite).count(),
        vids.iter().filter(|v| v.created.is_none()).count()
    );
    let sizes: Vec<u64> = vids.iter().filter_map(|v| v.original_bytes()).collect();
    println!(
        "original size (fileSize key, {} of {} known): total {:.1} GB, {:.1} MB per video",
        sizes.len(),
        vids.len(),
        sizes.iter().sum::<u64>() as f64 / 1e9,
        sizes.iter().sum::<u64>() as f64 / 1e6 / sizes.len().max(1) as f64
    );
    if args.check_local {
        let cfg = VideoConfig {
            rendition: if args.fast {
                Rendition::Fast
            } else {
                Rendition::Medium
            },
            idle_timeout: Duration::from_secs(30),
            hard_timeout: Duration::from_secs(60),
            ..VideoConfig::default()
        };
        let t = Instant::now();
        let (mut local, mut cloud, mut err) = (0usize, 0usize, 0usize);
        let (mut cloud_secs, mut cloud_bytes) = (0f64, 0u64);
        for (i, v) in vids.iter().enumerate() {
            match video_is_local(v, &cfg) {
                Ok(true) => local += 1,
                Ok(false) => {
                    cloud += 1;
                    cloud_secs += v.duration;
                    cloud_bytes += v.original_bytes().unwrap_or(0);
                }
                Err(e) => {
                    err += 1;
                    if err <= 3 {
                        println!("  check error: {}", e.to_string().replace(&v.id, "<id>"));
                    }
                }
            }
            if (i + 1) % 250 == 0 {
                eprintln!("  checked {}/{}", i + 1, vids.len());
            }
        }
        println!(
            "{} rendition on this Mac: {local} local, {cloud} only in iCloud ({:.1} h, originals {:.1} GB), {err} errors; {:.1}s",
            if args.fast { "fast" } else { "medium" },
            cloud_secs / 3600.0,
            cloud_bytes as f64 / 1e9,
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
