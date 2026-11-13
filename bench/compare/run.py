"""Run the e-means comparison: every impl, data set, k and seed, interleaved.

  python3 run.py --host mac --impls rs-metal-fused,bend-metal,py-torch-mps \
      --datasets s768_16k,s768_100k,s768_200k --ks 64,256,1024 --seeds 1,2,3,4,5

Order: per (data set, k), one warm-up run per impl (seed 1000, discarded), then
for each seed every impl once, the impl order rotated by one each seed
(round-robin, never all of one impl then the next). Each run is a fresh process
under /usr/bin/time (-l on macOS, -v on Linux) for peak RSS; on Linux a sampler
polls nvidia-smi for the run's GPU memory. Before each run the 1-minute load
average is recorded. A run that already has an ok run.json is skipped (resume).

Output: results/<host>/runs/<data>_k<k>/<impl>/s<seed>/run.json (+ centroids.f32,
gitignored). Then run evaluate.py for the shared objective and ARI.
"""

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "bench/data"
LINUX = platform.system() == "Linux"
TARGET = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target/bench"))
RS_BIN = TARGET / "release/emeans-compare"
BEND_BIN = ROOT / "bend/build/emeans"
PY = HERE / "py/.venv/bin/python"
GPU_IMPLS = {"rs-metal-fused", "rs-metal-unfused", "bend-metal", "py-torch-mps",
             "rs-cuda", "rs-cuda-unfused", "rs-cuda-bf16", "bend-cuda", "py-torch-cuda-native", "py-torch-cuda-triton",
             "py-torch-cuda-native-f32", "py-torch-cuda-triton-f32"}


def command(impl, data, n, d, k, seed, out):
    f = str(DATA / f"{data}.f32")
    if impl.startswith("py-"):
        return [str(PY), str(HERE / "py/run_py.py"), "--impl", impl, "--data", f, "--d", str(d),
                "--k", str(k), "--seed", str(seed), "--out", str(out)]
    if impl.startswith("rs-"):
        backend = {"rs-cpu": "cpu", "rs-cpu-accelerate": "cpu-accelerate", "rs-cpu-blocked": "cpu-blocked",
                   "rs-cpu-candle-gemm": "cpu-candle-gemm", "rs-cpu-fused": "cpu-fused", "rs-metal-fused": "metal-fused",
                   "rs-metal-unfused": "metal-unfused", "rs-cuda": "cuda",
                   "rs-cuda-unfused": "cuda-unfused", "rs-cuda-bf16": "cuda-bf16"}[impl]
        return [str(RS_BIN), "--data", f, "--d", str(d), "--k", str(k), "--seed", str(seed),
                "--backend", backend, "--out", str(out)]
    if impl.startswith("bend-"):
        gpu = "0" if impl == "bend-cpu" else "1"
        return [str(BEND_BIN), f, str(n), str(d), str(k), "5", str(seed), "1", str(out), gpu]
    raise ValueError(impl)


def parse_time(text):
    r = {}
    if LINUX:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        if m:
            r["peak_rss_bytes"] = int(m.group(1)) * 1024
    else:
        m = re.search(r"(\d+)\s+maximum resident set size", text)
        if m:
            r["peak_rss_bytes"] = int(m.group(1))
        m = re.search(r"(\d+)\s+peak memory footprint", text)
        if m:
            r["peak_footprint_bytes"] = int(m.group(1))
    return r


def descendants(pid):
    kids = {}
    for p in pathlib.Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(p.name))
    out, todo = {pid}, [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.add(c)
            todo.append(c)
    return out


class GpuSampler(threading.Thread):
    """Peak GPU memory of a process tree, from nvidia-smi every 0.2 s (Linux)."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop = pid, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                tree = descendants(self.pid)
                used = sum(int(m) for p, m in (l.split(", ") for l in q.stdout.strip().splitlines() if l)
                           if int(p) in tree)
                self.peak = max(self.peak, used * 2**20)
            except Exception:
                pass
            self.stop.wait(0.2)


def bend_result(out):
    rep = json.loads((out / "report.json").read_text())
    return {"load_s": rep["load_ms"] / 1000,
            "train_s": (rep.get("init_ms", 0) + rep["train_ms"] + rep.get("finalize_ms", 0)) / 1000,
            "threads": {"bend_pool": os.cpu_count()}, "recipe": rep.get("recipe"),
            "n_splits": rep.get("n_splits"), "obj": rep.get("obj"), "backend": rep.get("backend")}


def run_one(impl, data, meta, k, seed, case_dir, warmup, timeout):
    out = case_dir / impl / f"s{seed}"
    rj = out / "run.json"
    if rj.exists() and not warmup:
        prev = json.loads(rj.read_text())
        if prev.get("ok"):
            return prev
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd = command(impl, data, meta["n"], meta["d"], k, seed, out)
    timer = ["/usr/bin/time", "-v"] if LINUX else ["/usr/bin/time", "-l"]
    rec = {"impl": impl, "data": data, "n": meta["n"], "d": meta["d"], "k": k, "seed": seed,
           "warmup": warmup, "cmd": cmd, "loadavg_before": list(os.getloadavg()),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    t = time.perf_counter()
    with open(out / "stdout.txt", "w") as so, open(out / "stderr.txt", "w") as se:
        proc = subprocess.Popen(timer + cmd, stdout=so, stderr=se, env=env)
        sampler = None
        if LINUX and impl in GPU_IMPLS and shutil.which("nvidia-smi"):
            sampler = GpuSampler(proc.pid)
            sampler.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = "timeout"
        if sampler:
            sampler.stop.set()
            sampler.join()
            rec["gpu_peak_bytes"] = sampler.peak
    rec["wall_s"] = time.perf_counter() - t
    err = (out / "stderr.txt").read_text(errors="replace")
    rec.update(parse_time(err))
    rec["rc"] = rc
    try:
        if rc != 0:
            raise RuntimeError(f"exit {rc}")
        if impl.startswith("bend-"):
            rec.update(bend_result(out))
        else:
            rec.update(json.loads((out / "impl.json").read_text()))
            if rec.get("gpu_peak_allocated_bytes") and not rec.get("gpu_peak_bytes"):
                rec["gpu_peak_bytes"] = rec["gpu_peak_allocated_bytes"]
        if not (out / "centroids.f32").exists():
            raise RuntimeError("no centroids.f32")
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - the error is recorded, the sweep goes on
        rec["ok"] = False
        tail = [l for l in err.strip().splitlines() if not re.match(r"^\s*\d+\s+[a-z]", l)][-12:]
        rec["error"] = f"{e}: " + " | ".join(tail)
    if warmup:
        for f in ("centroids.f32", "labels.i32"):
            (out / f).unlink(missing_ok=True)
    rj.write_text(json.dumps(rec, indent=1))
    status = "ok" if rec["ok"] else "FAIL " + rec["error"][:400]
    print(f"{time.strftime('%H:%M:%S')} {data} k={k} {impl:22s} s{seed}{' (warm-up)' if warmup else ''}: "
          f"train {rec.get('train_s', float('nan')):.2f}s load {rec.get('load_s', float('nan')):.2f}s "
          f"la {rec['loadavg_before'][0]:.1f} -> {status}", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--impls", required=True)
    p.add_argument("--datasets", default="s768_16k,s768_100k,s768_200k")
    p.add_argument("--ks", default="64,256,1024")
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=int, default=3600)
    a = p.parse_args()
    impls = a.impls.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    host_dir = HERE / "results" / a.host
    for data in a.datasets.split(","):
        meta = json.loads((DATA / f"{data}.json").read_text())
        for k in [int(s) for s in a.ks.split(",")]:
            case = host_dir / "runs" / f"{data}_k{k}"
            for w in range(a.warmup):
                for impl in impls:
                    run_one(impl, data, meta, k, 1000 + w, case, True, a.timeout)
            for i, seed in enumerate(seeds):
                order = impls[i % len(impls):] + impls[:i % len(impls)]
                for impl in order:
                    run_one(impl, data, meta, k, seed, case, False, a.timeout)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
