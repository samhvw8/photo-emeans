//! Probe the Photos library: authorize, list, count, and fetch a small sample.
//!
//! cargo run -p photos --example probe -- --limit 20 --out /path/to/dir
//!
//! Read-only. Prints counts and per-photo timing; never prints photo content.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use photos::{FetchAllConfig, FetchConfig, FetchEvent, fetch_all, is_local};

#[derive(Clone, Copy, ValueEnum)]
enum Pick {
    /// Evenly spaced across the library (oldest to newest).
    Spread,
    /// The oldest photos.
    First,
    /// The newest photos.
    Newest,
}

#[derive(Parser)]
struct Args {
    /// Photos to fetch.
    #[arg(long, default_value_t = 20)]
    limit: usize,
    /// Output directory (gets `*.jpg` and `manifest.jsonl`).
    #[arg(long)]
    out: PathBuf,
    /// Which photos to fetch.
    #[arg(long, value_enum, default_value_t = Pick::Spread)]
    pick: Pick,
    /// Requests in flight.
    #[arg(long, default_value_t = 6)]
    concurrency: usize,
    /// Before fetching, ask Photos (network off) whether each sample photo is local at 1024 px.
    #[arg(long)]
    check_local: bool,
    /// Seconds to wait for the permission prompt.
    #[arg(long, default_value_t = 300)]
    auth_timeout: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "photos access before request: {:?}",
        photos::auth::current()
    );
    photos::ensure_authorized(Duration::from_secs(args.auth_timeout))?;
    println!("photos access: Authorized");

    let t = Instant::now();
    let all = photos::list_photos()?;
    println!(
        "listed {} image assets in {:.2}s",
        all.len(),
        t.elapsed().as_secs_f64()
    );
    println!(
        "excluded: {} iCloud Shared Album photos, hidden photos, non-representative burst frames",
        photos::count_shared_album_photos()
    );
    let mut by_sub: BTreeMap<String, usize> = BTreeMap::new();
    for a in &all {
        for s in &a.subtypes {
            *by_sub.entry(s.clone()).or_default() += 1;
        }
    }
    let fav = all.iter().filter(|a| a.favorite).count();
    let burst = all.iter().filter(|a| a.burst_id.is_some()).count();
    let undated = all.iter().filter(|a| a.created.is_none()).count();
    let portrait = all
        .iter()
        .filter(|a| a.pixel_height > a.pixel_width)
        .count();
    println!("by subtype: {by_sub:?}");
    println!(
        "favorites {fav}, burst representatives {burst}, undated {undated}, portrait {portrait}"
    );
    if let (Some(f), Some(l)) = (all.first(), all.last()) {
        println!(
            "created range: {} .. {}",
            f.created.as_deref().unwrap_or("?"),
            l.created.as_deref().unwrap_or("?")
        );
    }

    let n = args.limit.min(all.len());
    let sample: Vec<_> = match args.pick {
        Pick::First => all[..n].to_vec(),
        Pick::Newest => all[all.len() - n..].to_vec(),
        Pick::Spread => (0..n)
            .map(|i| all[i * all.len() / n.max(1)].clone())
            .collect(),
    };
    let mut utis: BTreeMap<String, usize> = BTreeMap::new();
    for a in &sample {
        *utis
            .entry(a.original_uti().unwrap_or_else(|| "?".into()))
            .or_default() += 1;
    }
    println!("sample of {n}: original types {utis:?}");
    if args.check_local {
        let mut local = 0;
        let t = Instant::now();
        for a in &sample {
            if is_local(a, 1024)? {
                local += 1;
            }
        }
        println!(
            "sample local at 1024 px (network off): {local} local, {} need iCloud ({:.2}s)",
            n - local,
            t.elapsed().as_secs_f64()
        );
    }

    let cfg = FetchAllConfig {
        concurrency: args.concurrency,
        fetch: FetchConfig::default(),
    };
    let by_id: BTreeMap<&str, &photos::AssetInfo> =
        sample.iter().map(|a| (a.id.as_str(), a)).collect();
    let mut rows = 0;
    let report = fetch_all(&args.out, &sample, &cfg, |ev| match ev {
        FetchEvent::Done {
            line,
            bytes,
            from_icloud,
            attempts,
            elapsed,
        } => {
            rows += 1;
            let a = by_id[line.id.as_str()];
            let src_portrait = a.pixel_height > a.pixel_width;
            let out_portrait = line.height > line.width;
            println!(
                "  {rows:>3} {} {:>5}x{:<5} -> {:>4}x{:<4} {:>7} B {:>6} ms icloud={} attempts={} orient={} type={}",
                &line.file[..8],
                a.pixel_width,
                a.pixel_height,
                line.width,
                line.height,
                bytes,
                elapsed.as_millis(),
                from_icloud,
                attempts,
                if src_portrait == out_portrait || a.pixel_width == a.pixel_height {
                    "ok"
                } else {
                    "MISMATCH"
                },
                a.original_uti().unwrap_or_default(),
            );
        }
        FetchEvent::Failed { id, error } => {
            println!("  FAILED {}: {error}", &id[..8.min(id.len())])
        }
    })?;
    let secs = report.elapsed.as_secs_f64();
    println!(
        "fetched {} (from iCloud {}), skipped {} (already done), failed {}, repaired {}; {:.1}s total, {:.2} s/photo wall, {:.0} KB/photo avg",
        report.fetched,
        report.from_icloud,
        report.skipped,
        report.failed.len(),
        report.repaired,
        secs,
        if report.fetched > 0 {
            secs / report.fetched as f64
        } else {
            0.0
        },
        if report.fetched > 0 {
            report.bytes as f64 / report.fetched as f64 / 1024.0
        } else {
            0.0
        },
    );
    Ok(())
}
