//! Image embedding throughput of the candle EmbeddingGemma 2 port, for bench/compare.
//!
//! embed-compare <cpu|metal|cuda> <f32|bf16|f16> <batch> <n> <out-dir>
//!
//! Embeds bench/gemma/images (sorted *.png) cycled to n, end to end (decode, EXIF,
//! preprocess, model), after one warm-up batch. Writes OUT/embed.json (load_s,
//! seconds, images_per_s) and OUT/embeddings.f32 (the first pass over the unique
//! images, in sorted file order) for the cosine check against Python.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use encoder::Encoder;
use encoder::families::embeddinggemma2::{EmbeddingGemma2, MODEL_ID};
use serde_json::json;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dev_name = args.get(1).map(String::as_str).unwrap_or("metal");
    let device = match dev_name {
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        "cpu" => Device::Cpu,
        d => bail!("unknown device {d}"),
    };
    let dtype = match args.get(2).map(String::as_str).unwrap_or("f32") {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        d => bail!("unknown dtype {d}"),
    };
    let batch: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let n: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let out = PathBuf::from(args.get(5).cloned().unwrap_or("embed-out".into()));
    std::fs::create_dir_all(&out)?;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../gemma/images");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no images in {}", dir.display());
    }
    let paths: Vec<PathBuf> = files.iter().cycle().take(n).cloned().collect();

    let t = Instant::now();
    let emb = EmbeddingGemma2::load_with(MODEL_ID, &device, dtype)?.with_batch_size(batch);
    let load_s = t.elapsed().as_secs_f64();
    emb.embed_image_paths(&paths[..batch.min(n)], None)?;

    let t = Instant::now();
    let v = emb.embed_image_paths(&paths, None)?;
    let secs = t.elapsed().as_secs_f64();
    assert_eq!(v.len(), n);

    let uniq = files.len().min(n);
    let mut bytes = Vec::with_capacity(uniq * 768 * 4);
    for e in &v[..uniq] {
        for x in e.iter() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(out.join("embeddings.f32"), bytes)?;
    let names: Vec<String> = files[..uniq]
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let report = json!({
        "impl": "candle", "device": dev_name, "dtype": format!("{dtype:?}").to_lowercase(),
        "batch": batch, "n": n, "load_s": load_s, "seconds": secs,
        "images_per_s": n as f64 / secs, "files": names,
    });
    std::fs::write(out.join("embed.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
