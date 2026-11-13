//! CubeCL spike: fused nearest-centroid (argmax of x . c_j) timing and exactness.
//!
//! cargo run --release --features metal-native -- --runtime metal --data ../../data/s768_200k.f32 \
//!     --d 768 --kernels fma,mma-f32 --ks 64,256,1024 --bs 4096,65536,all --reps 7
//!
//! Prints one markdown row per (kernel, shape). `b = all` is a full pass over the file
//! (rows bound on the device in 65,536-row chunks, centroids uploaded and labels read back
//! per pass, as in `crates/emeans/examples/bench_assign.rs`).

mod kernels;

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::{Device, future::block_on};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn device(name: &str) -> Result<Device> {
    Ok(match name {
        #[cfg(feature = "metal-native")]
        "metal" => Device::Metal(Default::default()),
        #[cfg(any(feature = "wgpu", feature = "wgpu-msl", feature = "vulkan"))]
        "wgpu" => Device::Wgpu(Default::default()),
        #[cfg(feature = "cuda")]
        "cuda" => Device::Cuda(Default::default()),
        _ => bail!("runtime {name} not compiled in"),
    })
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Fma,
    MmaF32,
    MmaTf32,
    MmaBf16,
}

impl Kernel {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "fma" => Kernel::Fma,
            "mma-f32" => Kernel::MmaF32,
            "mma-tf32" => Kernel::MmaTf32,
            "mma-bf16" => Kernel::MmaBf16,
            _ => bail!("unknown kernel {s}"),
        })
    }
}

/// Launch one kernel on `rows` rows of `x` (device handle) against `c`.
#[allow(clippy::too_many_arguments)]
fn launch(
    client: &Client,
    kernel: Kernel,
    x: &Handle,
    c: &Handle,
    labels: &Handle,
    scores: &Handle,
    rows: usize,
    k: usize,
    d: usize,
) {
    let xa = || unsafe { BufferArg::from_raw_parts(x.clone(), rows * d / 4) };
    let ca = || unsafe { BufferArg::from_raw_parts(c.clone(), k * d / 4) };
    let la = || unsafe { BufferArg::from_raw_parts(labels.clone(), rows) };
    let sa = || unsafe { BufferArg::from_raw_parts(scores.clone(), rows) };
    match kernel {
        Kernel::Fma => {
            let (tm, tn, bk) = (4usize, 4usize, 16usize);
            let cubes = rows.div_ceil(16 * tm) as u32;
            unsafe {
                kernels::assign_fma::launch_unchecked(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_2d(16, 16),
                    xa(),
                    ca(),
                    la(),
                    sa(),
                    rows as u32,
                    k as u32,
                    d as u32,
                    tm,
                    tn,
                    bk,
                )
            }
        }
        Kernel::MmaF32 | Kernel::MmaTf32 | Kernel::MmaBf16 => {
            let (bm, bn, bk, wm, wn, mm, nn, kk, pad) = match kernel {
                Kernel::MmaF32 => (64, 64, 16, 2, 2, 8, 8, 8, 4),
                Kernel::MmaTf32 => (64, 64, 16, 2, 2, 16, 16, 8, 4),
                _ => (64, 64, 32, 2, 2, 16, 16, 16, 8),
            };
            // Tile override for tuning: EMEANS_TILE=bm,bn,bk,wm,wn
            let (bm, bn, bk, wm, wn) = match std::env::var("EMEANS_TILE") {
                Ok(t) => {
                    let v: Vec<usize> = t.split(',').map(|x| x.parse().unwrap()).collect();
                    (v[0], v[1], v[2], v[3], v[4])
                }
                Err(_) => (bm, bn, bk, wm, wn),
            };
            assert!(wm * wn * 32 >= 2 * bm);
            let cubes = rows.div_ceil(bm) as u32;
            let count = CubeCount::Static(cubes, 1, 1);
            let dim = CubeDim::new_1d((wm * wn * 32) as u32);
            macro_rules! go {
                ($t:ty) => {
                    unsafe {
                        kernels::assign_mma::launch_unchecked::<$t>(
                            client,
                            count,
                            dim,
                            xa(),
                            ca(),
                            la(),
                            sa(),
                            rows as u32,
                            k as u32,
                            d as u32,
                            bm,
                            bn,
                            bk,
                            wm,
                            wn,
                            mm,
                            nn,
                            kk,
                            pad,
                        )
                    }
                };
            }
            match kernel {
                Kernel::MmaF32 => go!(f32),
                Kernel::MmaTf32 => go!(tf32),
                _ => go!(half::bf16),
            }
        }
    }
}

/// f64 argmax (lowest id on ties) per row, multithreaded.
fn reference(x: &[f32], c: &[f32], d: usize, threads: usize) -> Vec<(u32, f64)> {
    let n = x.len() / d;
    let per = n.div_ceil(threads);
    let mut out = vec![(0u32, 0f64); n];
    std::thread::scope(|s| {
        for (ti, chunk) in out.chunks_mut(per).enumerate() {
            s.spawn(move || {
                for (r, o) in chunk.iter_mut().enumerate() {
                    let xi = &x[(ti * per + r) * d..(ti * per + r + 1) * d];
                    let mut best = (0u32, f64::NEG_INFINITY);
                    for (j, cj) in c.chunks(d).enumerate() {
                        let v = dot64(xi, cj);
                        if v > best.1 {
                            best = (j as u32, v);
                        }
                    }
                    *o = best;
                }
            });
        }
    });
    out
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    let mut s = [0f64; 4];
    for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for t in 0..4 {
            s[t] += p[t] as f64 * q[t] as f64;
        }
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

struct Exact {
    mism: usize,
    nontie: usize,
    max_gap: f64,
    max_score_err: f64,
}

/// Compare GPU labels/scores with the f64 reference. A mismatch is a tie when the
/// exact scores of the two centroids differ by at most 1e-5 (the bench_assign rule).
fn exactness(x: &[f32], c: &[f32], d: usize, reff: &[(u32, f64)], labels: &[u32], scores: &[f32]) -> Exact {
    let mut e = Exact { mism: 0, nontie: 0, max_gap: 0.0, max_score_err: 0.0 };
    for (i, (&l, &s)) in labels.iter().zip(scores).enumerate() {
        let xi = &x[i * d..(i + 1) * d];
        let exact_l = dot64(xi, &c[l as usize * d..(l as usize + 1) * d]);
        e.max_score_err = e.max_score_err.max((s as f64 - exact_l).abs());
        if l != reff[i].0 {
            e.mism += 1;
            let gap = reff[i].1 - exact_l;
            e.max_gap = e.max_gap.max(gap);
            if gap > 1e-5 {
                e.nontie += 1;
            }
        }
    }
    e
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = arg(&args, "--data").context("--data <file.f32>")?;
    let d: usize = arg(&args, "--d").unwrap_or("768".into()).parse()?;
    let runtime = arg(&args, "--runtime").context("--runtime metal|wgpu|cuda")?;
    let kernels: Vec<Kernel> = arg(&args, "--kernels")
        .unwrap_or("fma".into())
        .split(',')
        .map(Kernel::parse)
        .collect::<Result<_>>()?;
    let ks: Vec<usize> = arg(&args, "--ks")
        .unwrap_or("64,256,1024".into())
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let bs: Vec<String> = arg(&args, "--bs")
        .unwrap_or("4096,65536,all".into())
        .split(',')
        .map(String::from)
        .collect();
    let reps: usize = arg(&args, "--reps").unwrap_or("7".into()).parse()?;
    let max_rows: Option<usize> = arg(&args, "--rows").map(|s| s.parse()).transpose()?;
    let threads: usize = arg(&args, "--threads").unwrap_or("4".into()).parse()?;
    let check = !args.iter().any(|a| a == "--no-check");
    ensure!(d % 4 == 0, "d must be a multiple of 4");

    let bytes = std::fs::read(&path).with_context(|| path.clone())?;
    ensure!(bytes.len() % (4 * d) == 0, "file size is not a multiple of 4 * d");
    let mut x: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(bytes);
    if let Some(m) = max_rows {
        x.truncate(m * d);
    }
    let n = x.len() / d;

    let dev = device(&runtime)?;
    let client = dev.client();
    eprintln!("runtime {} on {:?}, n {n}, d {d}", client.name(), dev);

    // Bind the dataset once, in chunks of 65,536 rows.
    let chunk = 65536usize;
    let t = Instant::now();
    let xh: Vec<(Handle, usize)> = x
        .chunks(chunk * d)
        .map(|ch| (client.create_from_slice(f32::as_bytes(ch)), ch.len() / d))
        .collect();
    block_on(client.sync()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    eprintln!("bind {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    println!(
        "| runtime | kernel | b | k | ms (median of {reps}) | GFLOP/s | first pass ms (incl. JIT) | mismatches | non-tie | max gap | max score err |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for &k in &ks {
        // Centroids: k rows spread over the file (deterministic).
        let mut c = vec![0f32; k * d];
        for j in 0..k {
            let r = (j * 7919 + 13) % n;
            c[j * d..(j + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
        }
        let t = Instant::now();
        let reff = if check { reference(&x, &c, d, threads) } else { Vec::new() };
        if check {
            eprintln!("f64 reference k={k}: {:.1} s", t.elapsed().as_secs_f64());
        }
        for &kernel in &kernels {
            for b in &bs {
                let full = b == "all";
                let rows = if full { n } else { b.parse::<usize>()?.min(n) };
                ensure!(full || rows <= chunk, "b must be <= {chunk} or all");
                let lab = client.empty(chunk * 4);
                let sco = client.empty(chunk * 4);
                // One pass: upload centroids, launch each chunk, read labels and scores back.
                let pass = |want: bool| -> Result<(f64, Vec<u32>, Vec<f32>)> {
                    let t = Instant::now();
                    let ch = client.create_from_slice(f32::as_bytes(&c));
                    let mut ls = Vec::with_capacity(rows);
                    let mut ss = Vec::with_capacity(rows);
                    let mut done = 0;
                    for (xh, nr) in &xh {
                        if done >= rows {
                            break;
                        }
                        let nr = (*nr).min(rows - done);
                        launch(&client, kernel, xh, &ch, &lab, &sco, nr, k, d);
                        let lb = client.read_one(lab.clone()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                        let sb = client.read_one(sco.clone()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                        if want {
                            ls.extend_from_slice(&u32::from_bytes(&lb)[..nr]);
                            ss.extend_from_slice(&f32::from_bytes(&sb)[..nr]);
                        }
                        done += nr;
                    }
                    Ok((t.elapsed().as_secs_f64() * 1e3, ls, ss))
                };
                let (first, labels, scores) = pass(true)?;
                let times: Vec<f64> = (0..reps).map(|_| pass(false).map(|p| p.0)).collect::<Result<_>>()?;
                let ms = median(times);
                let gflops = 2.0 * rows as f64 * k as f64 * d as f64 / (ms * 1e6);
                let ex = if check {
                    let e = exactness(&x[..rows * d], &c, d, &reff[..rows], &labels, &scores);
                    format!("{} | {} | {:.2e} | {:.2e}", e.mism, e.nontie, e.max_gap, e.max_score_err)
                } else {
                    "- | - | - | -".into()
                };
                let bl = if full { format!("all ({rows})") } else { rows.to_string() };
                println!("| {runtime} | {kernel:?} | {bl} | {k} | {ms:.2} | {gflops:.0} | {first:.0} | {ex} |");
            }
        }
    }
    Ok(())
}
