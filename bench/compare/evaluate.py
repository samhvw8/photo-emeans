"""Shared evaluator for bench/compare: one objective and one ARI for every impl.

  python evaluate.py results/<host> [--data-dir ../data]

For every run dir results/<host>/runs/<data>_k<k>/<impl>/s<seed>/ that has
centroids.f32, it loads the data set and the final centroids and computes, the
same way for all impls (the impls' own reports are not used):

  objective = mean over all rows of (2 - 2 * max_j cos(x, c_j)), in float64
  labels    = argmax_j cos(x, c_j)   (nearest final centroid)
  ARI       = adjusted Rand index of labels vs py-numpy's labels, same data, k, seed

Writes eval.json next to each run.json, then results/<host>/summary.json and
results/<host>/table.md.
"""

import argparse
import json
import pathlib
import statistics

import numpy as np

CHUNK = 16384


def objective_and_labels(x, c):
    """x: n x d f32 rows; c: k x d centroids. Returns (objective f64, labels int32)."""
    c64 = c.astype(np.float64)
    norms = np.linalg.norm(c64, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("zero centroid")
    cn64 = c64 / norms
    cn32 = cn64.astype(np.float32)
    n = len(x)
    labels = np.empty(n, dtype=np.int32)
    total = 0.0
    for b in range(0, n, CHUNK):
        xb = x[b:b + CHUNK]
        lab = np.argmax(xb @ cn32.T, axis=1)
        x64 = xb.astype(np.float64)
        cos = np.einsum("ij,ij->i", x64, cn64[lab]) / np.linalg.norm(x64, axis=1)
        total += float(np.sum(2.0 - 2.0 * cos))
        labels[b:b + CHUNK] = lab
    return total / n, labels


def ari(a, b):
    """Adjusted Rand index of two label vectors (non-negative ints)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    nb = int(b.max()) + 1
    cont = np.bincount(a * nb + b, minlength=(int(a.max()) + 1) * nb).astype(np.float64)
    comb = lambda v: float(np.sum(v * (v - 1) / 2))
    sij = comb(cont)
    sa = comb(np.bincount(a).astype(np.float64))
    sb = comb(np.bincount(b).astype(np.float64))
    n = len(a)
    tot = n * (n - 1) / 2
    exp = sa * sb / tot
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (sij - exp) / (mx - exp)


def med_iqr(v):
    if not v:
        return None, None
    if len(v) < 2:
        return v[0], 0.0
    q = statistics.quantiles(v, n=4, method="inclusive")
    return statistics.median(v), q[2] - q[0]


def mean_sd(v):
    if not v:
        return None, None
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("host_dir")
    p.add_argument("--data-dir", default=str(pathlib.Path(__file__).resolve().parents[1] / "data"))
    p.add_argument("--force", action="store_true", help="recompute existing eval.json")
    a = p.parse_args()
    host = pathlib.Path(a.host_dir)
    data_dir = pathlib.Path(a.data_dir)
    cases = sorted((host / "runs").glob("*_k*"))
    rows = []
    for case in cases:
        data, k = case.name.rsplit("_k", 1)
        meta = json.loads((data_dir / f"{data}.json").read_text())
        x = None
        runs = sorted(case.glob("*/s*/run.json"))
        labels = {}
        evals = {}
        for rj in runs:
            d = rj.parent
            impl, seed = d.parent.name, int(d.name[1:])
            r = json.loads(rj.read_text())
            if r.get("warmup") or not r.get("ok"):
                continue
            ej = d / "eval.json"
            lab_path = d / "labels_eval.i32"
            if ej.exists() and lab_path.exists() and not a.force:
                evals[(impl, seed)] = json.loads(ej.read_text())
                labels[(impl, seed)] = np.fromfile(lab_path, dtype="<i4")
                continue
            if x is None:
                x = np.fromfile(data_dir / f"{data}.f32", dtype="<f4").reshape(meta["n"], meta["d"])
            c = np.fromfile(d / "centroids.f32", dtype="<f4").reshape(-1, meta["d"])
            assert len(c) == int(k), (rj, c.shape)
            obj, lab = objective_and_labels(x, c)
            lab.astype("<i4").tofile(lab_path)
            labels[(impl, seed)] = lab
            evals[(impl, seed)] = {"objective": obj}
        for (impl, seed), e in evals.items():
            ref = labels.get(("py-numpy", seed))
            e["ari_vs_py_numpy"] = None if ref is None else ari(labels[(impl, seed)], ref)
            run_dir = case / impl / f"s{seed}"
            (run_dir / "eval.json").write_text(json.dumps(e))
        # Noise floor: ARI between py-numpy runs of consecutive seeds (same data and k,
        # different seed). Two equally good runs agree only this much.
        ref_seeds = sorted(sd for (im, sd) in labels if im == "py-numpy")
        floor = [ari(labels[("py-numpy", a)], labels[("py-numpy", b)])
                 for a, b in zip(ref_seeds, ref_seeds[1:])]
        by_impl = {}
        for rj in runs:
            r = json.loads(rj.read_text())
            if r.get("warmup"):
                continue
            impl, seed = rj.parent.parent.name, int(rj.parent.name[1:])
            by_impl.setdefault(impl, []).append((seed, r, evals.get((impl, seed))))
        for impl, items in sorted(by_impl.items()):
            ok = [(s, r, e) for s, r, e in items if r.get("ok") and e]
            objs = [e["objective"] for _, _, e in ok]
            aris = [e["ari_vs_py_numpy"] for _, _, e in ok if e["ari_vs_py_numpy"] is not None]
            train = [r["train_s"] for _, r, _ in ok]
            load = [r["load_s"] for _, r, _ in ok]
            rss = [r["peak_rss_bytes"] for _, r, _ in ok if r.get("peak_rss_bytes")]
            foot = [r["peak_footprint_bytes"] for _, r, _ in ok if r.get("peak_footprint_bytes")]
            gpu = [r["gpu_peak_bytes"] for _, r, _ in ok if r.get("gpu_peak_bytes")]
            la = [r["loadavg_before"][0] for _, r, _ in items if r.get("loadavg_before")]
            om, osd = mean_sd(objs)
            tm, tiqr = med_iqr(train)
            am, asd = mean_sd(aris)
            rows.append({
                "data": data, "n": meta["n"], "k": int(k), "impl": impl,
                "runs_ok": len(ok), "runs_failed": len(items) - len(ok),
                "errors": sorted({r.get("error", "")[-300:] for _, r, _ in items if not r.get("ok")}),
                "objective_mean": om, "objective_sd": osd,
                "train_s_median": tm, "train_s_iqr": tiqr,
                "load_s_median": med_iqr(load)[0],
                "peak_rss_bytes_max": max(rss) if rss else None,
                "peak_footprint_bytes_max": max(foot) if foot else None,
                "gpu_peak_bytes_max": max(gpu) if gpu else None,
                "ari_mean": am, "ari_sd": asd,
                "ari_seed_floor": statistics.fmean(floor) if floor else None,
                "loadavg_1m_range": [min(la), max(la)] if la else None,
                "threads": next((r.get("threads") for _, r, _ in ok if r.get("threads")), None),
            })
    (host / "summary.json").write_text(json.dumps(rows, indent=1))
    md = render(rows) + "\n" + matrix(rows, "train_s_median", "{:.2f}", "Train s, median of 5 seeds") \
        + "\n" + matrix(rows, "objective_mean", "{:.5f}", "Objective, mean of 5 seeds") \
        + "\n" + matrix(rows, "ari_mean", "{:.3f}", "ARI vs py-numpy (same seed), mean of 5 seeds") \
        + "\n" + floor_table(rows) \
        + "\n" + mem_matrix(rows)
    (host / "table.md").write_text(md)
    print(md)


def matrix(rows, key, fmt, title):
    impls = sorted({r["impl"] for r in rows})
    cases = sorted({(r["n"], r["k"]) for r in rows})
    by = {(r["n"], r["k"], r["impl"]): r for r in rows}
    out = [f"{title}:", "", "| n | k | " + " | ".join(impls) + " |", "|---|---|" + "---|" * len(impls)]
    for n, k in cases:
        cells = []
        for i in impls:
            r = by.get((n, k, i))
            v = None if r is None else r.get(key)
            cells.append("-" if r is None else ("failed" if r["runs_ok"] == 0 else ("-" if v is None else fmt.format(v))))
        out.append(f"| {n:,} | {k} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def mem_matrix(rows):
    """Peak memory per impl: RSS / (GPU on Linux, footprint on macOS), GiB, max over runs."""
    for r in rows:
        dev = r["gpu_peak_bytes_max"] or r["peak_footprint_bytes_max"]
        r["_mem"] = f"{gb(r['peak_rss_bytes_max'])} / {gb(dev)}"
    out = matrix(rows, "_mem", "{}", "Peak memory GiB, RSS / GPU (Linux, nvidia-smi) or footprint (macOS)")
    for r in rows:
        r.pop("_mem")
    return out


def floor_table(rows):
    seen = {}
    for r in rows:
        if r["impl"] == "py-numpy" and r.get("ari_seed_floor") is not None:
            seen[(r["n"], r["k"])] = r["ari_seed_floor"]
    out = ["ARI noise floor (py-numpy seed s vs seed s+1, mean of 4 pairs):", "", "| n | k | ARI |", "|---|---|---|"]
    out += [f"| {n:,} | {k} | {v:.3f} |" for (n, k), v in sorted(seen.items())]
    return "\n".join(out) + "\n"


def gb(b):
    return "-" if b is None else f"{b / 2**30:.2f}"


def render(rows):
    out = ["| n | k | impl | objective mean ± sd | train s median (IQR) | load s | peak RSS GiB | peak GPU GiB (Linux) / footprint GiB (macOS) | ARI vs py-numpy | runs |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["n"], r["k"], r["impl"])):
        if r["runs_ok"] == 0:
            out.append(f"| {r['n']:,} | {r['k']} | {r['impl']} | failed | | | | | | 0/{r['runs_failed']} |")
            continue
        ari_s = "-" if r["ari_mean"] is None else f"{r['ari_mean']:.4f} ± {r['ari_sd']:.4f}"
        out.append(
            f"| {r['n']:,} | {r['k']} | {r['impl']} | {r['objective_mean']:.6f} ± {r['objective_sd']:.6f} | "
            f"{r['train_s_median']:.2f} ({r['train_s_iqr']:.2f}) | {r['load_s_median']:.2f} | "
            f"{gb(r['peak_rss_bytes_max'])} | {gb(r['gpu_peak_bytes_max'] or r['peak_footprint_bytes_max'])} | {ari_s} | "
            f"{r['runs_ok']}/{r['runs_ok'] + r['runs_failed']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
