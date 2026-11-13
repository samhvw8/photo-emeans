"""Image embedding throughput of sentence-transformers EmbeddingGemma 2, for bench/compare.

  python embed_py.py <cpu|mps|cuda> <f32|bf16> <batch> <n> <out-dir>

Same work as rs/src/bin/embed-compare.rs: bench/gemma/images (sorted *.png) cycled
to n, end to end (file decode, processor, model), after one warm-up batch.
Writes OUT/embed.json and OUT/embeddings.f32 (first pass over the unique images).
Model: google/embeddinggemma-2 at the revision crates/encoder (family embeddinggemma2) pins, loaded as
bench/gemma/reference.py does (audio tower off).
"""

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
IMAGES = pathlib.Path(__file__).resolve().parents[2] / "gemma/images"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def main():
    dev, dt, batch, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), pathlib.Path(sys.argv[5])
    out.mkdir(parents=True, exist_ok=True)
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[dt]
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    import transformers

    files = sorted(IMAGES.glob("*.png"))
    paths = [files[i % len(files)] for i in range(n)]
    t = time.perf_counter()
    model = SentenceTransformer("google/embeddinggemma-2", revision=REVISION, device=dev,
                                model_kwargs={"dtype": dtype}, config_kwargs={"audio_config": None})
    model.eval()
    sync(dev)
    load_s = time.perf_counter() - t
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def run(ps):
        imgs = [Image.open(p).convert("RGB") for p in ps]
        with torch.no_grad():
            e = model.encode(imgs, batch_size=batch, convert_to_numpy=True)
        sync(dev)
        return e

    run(paths[:batch])
    t = time.perf_counter()
    emb = run(paths)
    secs = time.perf_counter() - t
    uniq = min(len(files), n)
    np.ascontiguousarray(emb[:uniq], dtype="<f4").tofile(out / "embeddings.f32")
    rep = {"impl": "sentence-transformers", "device": dev, "dtype": dt, "batch": batch, "n": n,
           "load_s": load_s, "seconds": secs, "images_per_s": n / secs,
           "files": [f.name for f in files[:uniq]], "threads": torch.get_num_threads(),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "sentence_transformers": sentence_transformers.__version__}}
    if dev == "cuda":
        rep["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    if dev == "mps":
        rep["mps_driver_allocated_bytes_end"] = int(torch.mps.driver_allocated_memory())
    (out / "embed.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
