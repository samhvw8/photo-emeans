"""Run the embedding throughput comparison and the cosine sanity check.

  python3 embed_run.py --host mac --cases candle:metal:f32,st:mps:f32,st:cpu:f32 --batches 1,8 --n 64

Each case runs in a fresh process under /usr/bin/time (peak RSS, and on macOS the
peak footprint, which includes Metal/MPS buffers). On Linux a sampler records the
process tree's peak GPU memory. Then every case's embeddings of the unique images
are compared by cosine against the first sentence-transformers f32 case
(and against bench/gemma/ref/images.f32 when present: the CPU f32 reference).
Writes results/<host>/embed/<impl>_<dev>_<dtype>_b<batch>/ and results/<host>/embed.md.
"""

import argparse
import json
import os
import pathlib
import subprocess
import time

import run as R

HERE = R.HERE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--batches", default="1,8")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--max-footprint-gb", type=float, default=0,
                   help="macOS: kill a case whose process tree passes this physical footprint (0 = off)")
    a = p.parse_args()
    base = HERE / "results" / a.host / "embed"
    rs_embed = R.TARGET / "release/embed-compare"
    for case in a.cases.split(","):
        impl, dev, dt = case.split(":")
        for b in [int(x) for x in a.batches.split(",")]:
            out = base / f"{impl}_{dev}_{dt}_b{b}"
            if (out / "run.json").exists() and json.loads((out / "run.json").read_text()).get("ok"):
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = ([str(rs_embed), dev, dt, str(b), str(a.n), str(out)] if impl == "candle" else
                   [str(R.PY), str(HERE / "py/embed_py.py"), dev, dt, str(b), str(a.n), str(out)])
            timer = ["/usr/bin/time", "-v"] if R.LINUX else ["/usr/bin/time", "-l"]
            rec = {"case": case, "batch": b, "cmd": cmd, "loadavg_before": list(os.getloadavg())}
            with open(out / "stdout.txt", "w") as so, open(out / "stderr.txt", "w") as se:
                proc = subprocess.Popen(timer + cmd, stdout=so, stderr=se)
                s = R.GpuSampler(proc.pid) if R.LINUX and dev == "cuda" else None
                if s:
                    s.start()
                rc = None
                while rc is None:
                    try:
                        rc = proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        fp = tree_footprint(proc.pid) if a.max_footprint_gb and not R.LINUX else 0
                        if fp > a.max_footprint_gb * 2**30:
                            rec["killed"] = f"footprint {fp / 2**30:.1f} GB > {a.max_footprint_gb} GB"
                            for pid in tree(proc.pid):
                                subprocess.run(["kill", "-9", str(pid)])
                if s:
                    s.stop.set(); s.join(); rec["gpu_peak_bytes"] = s.peak
            err = (out / "stderr.txt").read_text(errors="replace")
            rec.update(R.parse_time(err))
            rec["ok"] = rc == 0 and (out / "embed.json").exists()
            if rec["ok"]:
                rec.update(json.loads((out / "embed.json").read_text()))
            else:
                rec["error"] = rec.get("killed") or f"exit {rc}: " + " | ".join(err.strip().splitlines()[-8:])
            (out / "run.json").write_text(json.dumps(rec, indent=1))
            print(time.strftime("%H:%M:%S"), case, f"b{b}", f"{rec.get('images_per_s', 0):.2f} img/s" if rec["ok"] else rec["error"][:500], flush=True)
    report(base, a.host)


def tree(root):
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = {}
    for line in out.splitlines():
        pid, ppid = map(int, line.split())
        kids.setdefault(ppid, []).append(pid)
    res, todo = [root], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            res.append(c)
            todo.append(c)
    return res


def tree_footprint(root):
    """Sum of phys_footprint (bytes) over a process tree, from macOS `footprint`."""
    import re
    total = 0
    for pid in tree(root):
        o = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s+([\d.]+)\s*(KB|MB|GB)", o)
        if m:
            total += float(m.group(1)) * {"KB": 2**10, "MB": 2**20, "GB": 2**30}[m.group(2)]
    return total


def report(base, host):
    import numpy as np
    runs = []
    for rj in sorted(base.glob("*/run.json")):
        r = json.loads(rj.read_text())
        r["dir"] = rj.parent
        runs.append(r)
    ok = [r for r in runs if r.get("ok")]
    ref_run = next((r for r in ok if r["impl"] == "sentence-transformers" and r["dtype"] == "f32"), None)
    load = lambda r: np.fromfile(r["dir"] / "embeddings.f32", dtype="<f4").reshape(-1, 768).astype(np.float64)
    refv = load(ref_run) if ref_run else None
    cpu_ref = HERE.parents[0] / "gemma/ref/images.f32"
    cpu_json = HERE.parents[0] / "gemma/ref/images.json"
    cpu_map = None
    if cpu_ref.exists() and cpu_json.exists():
        meta = json.loads(cpu_json.read_text())
        names = [pathlib.Path(m["file"]).name for m in (meta["images"] if isinstance(meta, dict) else meta)]
        cpu_map = dict(zip(names, np.fromfile(cpu_ref, dtype="<f4").reshape(-1, 768).astype(np.float64)))
    cos = lambda a, b: np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    lines = ["| impl | device | dtype | batch | images/s | load s | peak RSS GiB | peak footprint / GPU GiB | cos vs ST f32 (min) | cos vs CPU ref (min) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if not r.get("ok"):
            lines.append(f"| {r['case']} | | | {r['batch']} | failed: {r['error'][:120]} | | | | | |")
            continue
        v = load(r)
        c1 = f"{cos(v, refv).min():.6f}" if refv is not None and len(refv) == len(v) else "-"
        c2 = "-"
        if cpu_map and all(f[:-4] + ".png" in cpu_map or f in cpu_map for f in r["files"]):
            ref2 = np.stack([cpu_map[f] for f in r["files"]])
            c2 = f"{cos(v, ref2).min():.6f}"
        mem = r.get("gpu_peak_bytes") or r.get("peak_footprint_bytes")
        lines.append(f"| {r['impl']} | {r['device']} | {r['dtype']} | {r['batch']} | {r['images_per_s']:.2f} | "
                     f"{r['load_s']:.1f} | {r.get('peak_rss_bytes', 0) / 2**30:.2f} | "
                     f"{(mem or 0) / 2**30:.2f} | {c1} | {c2} |")
    md = "\n".join(lines) + "\n"
    (base.parent / "embed.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
