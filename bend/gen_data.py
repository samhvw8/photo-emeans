# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2"]
# ///
"""Write synthetic unit-sphere Gaussian mixtures for the Bend e-means tests.

Each dataset is a folder under bend/testdata/ in the project's data contract
format: embeddings.f32 (raw LE f32, n x d, rows L2-normalized) and meta.json.

    uv run bend/gen_data.py            # all four datasets
    uv run bend/gen_data.py --only small
"""

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "testdata"

# name: (n, d, true_clusters, noise per coordinate)
# The noise keeps each point's cosine to its center near 0.75 in both
# dimensions, so clusters overlap like real embeddings do.
DATASETS = {
    "n20k_d16": (20_000, 16, 100, 0.20),
    "n20k_d768": (20_000, 768, 100, 0.030),
    "n100k_d16": (100_000, 16, 100, 0.20),
    "n100k_d768": (100_000, 768, 100, 0.030),
}
SMALL = ["n20k_d16", "n20k_d768"]


def make(n, d, true_k, noise, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((true_k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Uneven cluster sizes (Dirichlet), so some clusters are small.
    weights = rng.dirichlet(np.full(true_k, 0.7))
    labels = rng.choice(true_k, size=n, p=weights)
    x = centers[labels] + noise * rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype("<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["small", "all"], default="all")
    args = ap.parse_args()
    names = SMALL if args.only == "small" else list(DATASETS)
    for i, name in enumerate(names):
        n, d, true_k, noise = DATASETS[name]
        folder = OUT / name
        path = folder / "embeddings.f32"
        meta = folder / "meta.json"
        if path.exists() and path.stat().st_size == n * d * 4 and meta.exists():
            print("exists", name)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        x = make(n, d, true_k, noise, seed=1000 + list(DATASETS).index(name))
        tmp = path.with_suffix(".tmp")
        x.tofile(tmp)
        tmp.replace(path)
        meta.write_text(json.dumps({
            "model": "synthetic-sphere-mixture", "d": d, "n": n,
            "dtype": "f32", "true_clusters": true_k, "noise": noise,
        }) + "\n")
        print("wrote", name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
