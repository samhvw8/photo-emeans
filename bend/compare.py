# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "faiss-cpu>=1.9"]
# ///
"""Compare the Bend e-means against the vendored faiss NumpyEmeans reference.

For each dataset x k x seed it runs reference/faiss_e_means NumpyEmeans and
bend/build/emeans with the same settings, then checks:

- the recipe: batch size, steps per pass, epoch rows and total steps must be
  equal; alpha0 is compared as float32 (Bend has no F64, see the report);
- the objective: the last pass's moving objective (em.obj[-1]) and the error
  of the final centroids (every row against its nearest final centroid,
  computed here in float64 for both), mean +- sd over the seeds.

    uv run bend/compare.py --small            # the mise bend:test gate
    uv run bend/compare.py --full --seeds 5   # the results table
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "reference"))
import faiss  # noqa: E402
from faiss_e_means import NumpyEmeans  # noqa: E402

BIN = HERE / "build" / "emeans"
DATA = HERE / "testdata"
OUT = HERE / "out"

SHAPES = {
    "n20k_d16": (20_000, 16),
    "n20k_d768": (20_000, 768),
    "n100k_d16": (100_000, 16),
    "n100k_d768": (100_000, 768),
}


def load(name):
    n, d = SHAPES[name]
    x = np.fromfile(DATA / name / "embeddings.f32", dtype="<f4")
    assert x.size == n * d, (name, x.size)
    return x.reshape(n, d)


def final_objective(x, centroids, spherical):
    """Error of every row against its nearest final centroid, in float64."""
    total = 0.0
    c64 = centroids.astype("f8")
    cn = (c64 * c64).sum(1)
    for i in range(0, len(x), 8192):
        xb = x[i:i + 8192].astype("f8")
        ip = xb @ c64.T
        if spherical:
            total += (2.0 - 2.0 * ip.max(1)).sum()
        else:
            dist = (xb * xb).sum(1)[:, None] + cn[None, :] - 2.0 * ip
            total += np.maximum(dist.min(1), 0).sum()
    return total


def nearest(x, centroids, spherical):
    out = np.empty(len(x), dtype="i8")
    c = centroids.astype("f4")
    cn = (c * c).sum(1)
    for i in range(0, len(x), 8192):
        ip = x[i:i + 8192] @ c.T
        out[i:i + 8192] = ip.argmax(1) if spherical else (cn[None, :] - 2 * ip).argmin(1)
    return out


def run_ref(x, k, nepochs, seed, spherical):
    em = NumpyEmeans(x.shape[1], k, nepochs=nepochs, seed=seed, spherical=spherical)
    t = time.perf_counter()
    centroids, labels = em.train(x)
    wall = time.perf_counter() - t
    r = em.recipe
    return {
        "recipe": {
            "batch_size": r["batch_size"], "alpha0": r["alpha0"],
            "steps_per_pass": r["steps_per_pass"], "total_steps": r["total_steps"],
            "epoch_rows": list(r["epoch_rows"]), "batch_law": r["batch_law"],
        },
        "obj": [float(o) for o in em.obj],
        "final": final_objective(x, centroids, spherical),
        "wall_s": wall, "splits": int(em.n_splits),
        "label_agree": float((nearest(x, centroids, spherical) == labels).mean()),
    }


def run_bend(name, x, k, nepochs, seed, spherical, gpu):
    n, d = x.shape
    out = OUT / f"{name}_k{k}_s{seed}_{'gpu' if gpu else 'cpu'}"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("labels.i32", "centroids.f32", "report.json"):
        (out / f).unlink(missing_ok=True)
    cmd = [str(BIN), str(DATA / name / "embeddings.f32"), str(n), str(d), str(k),
           str(nepochs), str(seed), "1" if spherical else "0", str(out), "1" if gpu else "0"]
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t
    if proc.returncode != 0:
        raise SystemExit(f"bend failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    rep = json.loads((out / "report.json").read_text())
    centroids = np.fromfile(out / "centroids.f32", dtype="<f4")
    labels = np.fromfile(out / "labels.i32", dtype="<i4")
    assert centroids.size == k * d, ("centroids", centroids.size)
    assert labels.size == n, ("labels", labels.size)
    assert np.isfinite(centroids).all(), "non-finite centroid"
    assert labels.min() >= -1 and labels.max() < k, ("labels range", labels.min(), labels.max())
    centroids = centroids.reshape(k, d)
    if spherical:
        norms = np.linalg.norm(centroids, axis=1)
        assert np.allclose(norms[norms > 0], 1, atol=1e-5), "spherical centroids not unit"
    rep["final"] = final_objective(x, centroids, spherical)
    rep["wall_s"] = wall
    # Last-pass labels move with the centroids, so they need not all be the
    # nearest final centroid; most should be.
    seen = labels >= 0
    near = nearest(x[seen], centroids, spherical)
    rep["label_agree"] = float((near == labels[seen]).mean())
    rep["labelled"] = int(seen.sum())
    return rep


def ulps(a, b):
    ia = np.array([a], dtype="f4").view("i4")[0]
    ib = np.array([b], dtype="u4").view("i4")[0]
    return abs(int(ia) - int(ib))


def check_recipe(ref, bend):
    r, b = ref["recipe"], bend["recipe"]
    problems = []
    for key in ("batch_size", "steps_per_pass", "total_steps", "epoch_rows", "batch_law"):
        if r[key] != b[key]:
            problems.append(f"{key}: ref {r[key]} bend {b[key]}")
    du = ulps(r["alpha0"], b["alpha0_bits"])
    rel = abs(float(np.float32(r["alpha0"])) - float(b["alpha0"])) / r["alpha0"]
    return problems, du, rel


def ms(xs):
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def fmt(m, s, digits=1):
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--small", action="store_true", help="n20k datasets, k=16, 2 seeds (the test gate)")
    g.add_argument("--full", action="store_true", help="every dataset and k")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--ks", nargs="*", type=int)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--nepochs", default="5")
    ap.add_argument("--euclidean", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="run Bend's assignment with ! (Metal)")
    ap.add_argument("--tol", type=float, default=0.02, help="largest relative objective gap")
    ap.add_argument("--json", type=pathlib.Path, help="write all runs here")
    args = ap.parse_args()

    if args.small:
        datasets, ks, seeds = ["n20k_d16", "n20k_d768"], [16], 2
    else:
        datasets = args.datasets or list(SHAPES)
        ks = args.ks or [16, 64, 256]
        seeds = args.seeds
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}: run `mise run bend:build`")
    nepochs = float(args.nepochs)
    nepochs = int(nepochs) if nepochs.is_integer() else nepochs
    spherical = not args.euclidean
    print(f"faiss {faiss.__version__}, threads {faiss.omp_get_max_threads()}; bend {'gpu' if args.gpu else 'cpu'}", flush=True)

    rows, runs, failures = [], [], []
    for name in datasets:
        x = load(name)
        for k in ks:
            ref_obj, ref_fin, ref_t, b_obj, b_fin, b_t, b_wall = [], [], [], [], [], [], []
            for seed in range(1, seeds + 1):
                ref = run_ref(x, k, nepochs, seed, spherical)
                bend = run_bend(name, x, k, nepochs, seed, spherical, args.gpu)
                problems, du, rel = check_recipe(ref, bend)
                if problems:
                    failures.append(f"{name} k={k} seed={seed}: recipe differs: {problems}")
                if rel > 1e-6:
                    failures.append(f"{name} k={k} seed={seed}: alpha0 off by {rel:.2e} ({du} ulp)")
                runs.append({"dataset": name, "k": k, "seed": seed, "ref": ref, "bend": bend,
                             "alpha0_ulps": du})
                ref_obj.append(ref["obj"][-1]); ref_fin.append(ref["final"]); ref_t.append(ref["wall_s"])
                b_obj.append(bend["obj"][-1]); b_fin.append(bend["final"])
                b_t.append(bend["train_ms"] / 1000); b_wall.append(bend["wall_s"])
                print(f"  {name} k={k} seed={seed}: ref obj {ref['obj'][-1]:.2f} final {ref['final']:.2f} "
                      f"{ref['wall_s']:.2f}s | bend obj {bend['obj'][-1]:.2f} final {bend['final']:.2f} "
                      f"train {bend['train_ms'] / 1000:.2f}s wall {bend['wall_s']:.2f}s | "
                      f"batch {bend['recipe']['batch_size']} alpha0 {du} ulp | splits {ref['splits']}/{bend['n_splits']} "
                      f"labels agree {ref['label_agree']:.3f}/{bend['label_agree']:.3f} ({bend['labelled']} labelled)", flush=True)
            gap = (statistics.mean(b_fin) - statistics.mean(ref_fin)) / statistics.mean(ref_fin)
            gap_obj = (statistics.mean(b_obj) - statistics.mean(ref_obj)) / statistics.mean(ref_obj)
            if gap > args.tol:
                failures.append(f"{name} k={k}: final objective gap {gap:+.2%} > {args.tol:.0%}")
            rows.append((name, k, ms(b_obj), ms(ref_obj), gap_obj, ms(b_fin), ms(ref_fin), gap,
                         ms(b_t), ms(b_wall), ms(ref_t), runs[-1]["bend"]["recipe"]))

    print()
    print(f"nepochs {nepochs}, {seeds} seeds, {'spherical' if spherical else 'Euclidean'}, "
          f"Bend backend: {'Metal GPU (!)' if args.gpu else 'C, CPU threads'}")
    print("| dataset | k | batch | alpha0 | Bend last-pass obj | faiss last-pass obj | gap | "
          "Bend final obj | faiss final obj | gap | Bend train s | Bend wall s | faiss train s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (name, k, bo, ro, go, bf, rf, gf, bt, bw, rt, rec) in rows:
        print(f"| {name} | {k} | {rec['batch_size']} | {rec['alpha0']:.6g} | {fmt(*bo)} | {fmt(*ro)} | {go:+.2%} | "
              f"{fmt(*bf)} | {fmt(*rf)} | {gf:+.2%} | {fmt(*bt, 2)} | {fmt(*bw, 2)} | {fmt(*rt, 2)} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"nepochs": nepochs, "gpu": args.gpu, "runs": runs}, indent=1))
    if failures:
        print("\nFAIL")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
