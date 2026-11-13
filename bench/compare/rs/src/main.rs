//! Train the Rust e-means once on a raw f32 file and write the run for bench/compare.
//!
//! emeans-compare --data X.f32 --d 768 --k 256 --seed 1 --backend metal-fused --out DIR
//!
//! Writes DIR/centroids.f32 (k x d, LE f32) and DIR/impl.json (load_s, setup_s,
//! warm_s, upload_s, train_s, threads, recipe, objectives as the model reports
//! them). train_s covers `Emeans::train` only: the engine is built, warmed up and
//! holds the data set on the device (upload_s) before the timer starts.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use emeans::{Assigner, Backend, Config, Emeans, SharedAssigner, make_assigner};
use serde_json::json;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&args, "--data").context("--data")?);
    let d: usize = arg(&args, "--d").context("--d")?.parse()?;
    let k: usize = arg(&args, "--k").context("--k")?.parse()?;
    let seed: u32 = arg(&args, "--seed").context("--seed")?.parse()?;
    let nepochs: f64 = arg(&args, "--nepochs").unwrap_or("5".into()).parse()?;
    let backend_name = arg(&args, "--backend").unwrap_or("cpu".into());
    // Library backends and the candle baselines (feature `baselines`).
    let backend: Backend = emeans_bench::backend(&backend_name)?;
    let out = PathBuf::from(arg(&args, "--out").context("--out")?);
    std::fs::create_dir_all(&out)?;

    let t = Instant::now();
    // Shared, so an engine can keep it and read it in place (`bind_shared`).
    let x = std::sync::Arc::new(emeans::io::read_f32_matrix(&data, d)?);
    let load_s = t.elapsed().as_secs_f64();
    let n = x.len() / d;

    let mut cfg = Config::new(d, k);
    cfg.nepochs = nepochs;
    cfg.seed = seed;
    cfg.spherical = true;
    let t = Instant::now();
    // Shared, so the warm-up model and the timed model use one engine.
    let mut assigner = SharedAssigner::new(make_assigner(backend)?);
    let setup_s = t.elapsed().as_secs_f64();

    // In-process warm-up, not timed (the Python runs do the same): one epoch on the
    // first max(4k, 4096) rows with the same backend and k. It pays one-time costs
    // (CUDA context and lazy module loads, Metal pipelines, rayon pool) that are not
    // training work.
    let t = Instant::now();
    let wr = n.min((4 * k).max(4096));
    if wr >= k {
        let mut wc = cfg.clone();
        wc.nepochs = 1.0;
        Emeans::with_assigner(wc, Box::new(assigner.clone()))?.train(&x[..wr * d])?;
    }
    let warm_s = t.elapsed().as_secs_f64();

    // Putting the data set on the device, timed apart like the Python harness's
    // `.to(device)` (in its load_s): CUDA copies it, Metal maps it in place
    // (`bind_shared`). A no-op on the CPU backends.
    let t = Instant::now();
    assigner.bind_shared(x.clone(), d)?;
    let upload_s = t.elapsed().as_secs_f64();
    let mut m = Emeans::with_assigner(cfg, Box::new(assigner.clone()))?;

    let t = Instant::now();
    m.train(&x)?;
    let train_s = t.elapsed().as_secs_f64();

    emeans::io::write_f32(
        out.join("centroids.f32"),
        m.centroids().context("no centroids")?,
    )?;
    let st = m.iteration_stats();
    let report = json!({
        "n": n, "d": d, "k": k, "seed": seed, "nepochs": nepochs,
        "backend": backend_name,
        "assigner": m.assigner_name(),
        "load_s": load_s, "setup_s": setup_s, "warm_s": warm_s, "upload_s": upload_s, "train_s": train_s,
        "threads": {"rayon": rayon::current_num_threads()},
        "recipe": m.recipe(),
        "n_splits": m.n_splits(),
        "obj": st.iter().map(|s| s.obj).collect::<Vec<_>>(),
    });
    std::fs::write(out.join("impl.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
