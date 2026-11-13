"""faiss Triton assignment-only timing and exactness, same shapes and protocol as the CubeCL spike.

python triton_assign.py --data s768_200k.f32 --d 768 --upstream <dir with faiss_e_means> [--reps 7]

One pass = upload centroids from the host, run `assign_rows` on each 65,536-row chunk of the
device-resident rows, copy labels and scores back to the host. `kernel ms` times the same
launches without the uploads and copies (CUDA events). Ranking is inner product: the centroid
norm vector is zero, so the kernel's `norm - 2 x.c` is `-2 x.c`.
"""

import argparse
import sys
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--d", type=int, default=768)
p.add_argument("--upstream", required=True)
p.add_argument("--ks", default="64,256,1024")
p.add_argument("--bs", default="4096,65536,all")
p.add_argument("--reps", type=int, default=7)
p.add_argument("--dtypes", default="f32,bf16,bf16-both")
a = p.parse_args()
sys.path.insert(0, a.upstream)
from faiss_e_means.triton_assignment import assign_rows  # noqa: E402

d = a.d
x_host = np.fromfile(a.data, dtype="<f4").reshape(-1, d)
n = x_host.shape[0]
x = torch.from_numpy(x_host).cuda()
CHUNK = 65536
chunks = [x[i:i + CHUNK] for i in range(0, n, CHUNK)]
print(f"| impl | b | k | ms (median of {a.reps}) | kernel ms | GFLOP/s (pass) | first pass ms (incl. JIT) | mismatches | non-tie | max gap |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in map(int, a.ks.split(",")):
    idx = [(j * 7919 + 13) % n for j in range(k)]
    c_host = np.ascontiguousarray(x_host[idx])
    c64 = torch.from_numpy(c_host).cuda().double()
    ref_s, ref_l = [], []
    for ch in chunks:
        s = ch.double() @ c64.T
        v, l = s.max(dim=1)  # first max on ties = lowest id
        ref_s.append(s)
        ref_l.append(l)
    for dt in a.dtypes.split(","):
        tdt = torch.float32 if dt == "f32" else torch.bfloat16
        xcast = dt == "bf16-both"  # faiss default on CUDA: rows and centroids in BF16 (cast timed)
        for b in a.bs.split(","):
            rows = n if b == "all" else min(int(b), n)

            def run(want):
                c = torch.from_numpy(c_host).cuda().to(tdt)
                zero = torch.zeros(k, dtype=torch.float32, device="cuda")
                ls, ss, done = [], [], 0
                for ch in chunks:
                    if done >= rows:
                        break
                    nr = min(ch.shape[0], rows - done)
                    xr = ch[:nr].to(tdt) if xcast else ch[:nr]
                    dist, lab = assign_rows(xr, c, zero, spherical=True)
                    ls.append(lab.cpu())
                    ss.append(dist.cpu())
                    done += nr
                return (torch.cat(ls), torch.cat(ss)) if want else None

            def kernel_ms():
                c = torch.from_numpy(c_host).cuda().to(tdt)
                zero = torch.zeros(k, dtype=torch.float32, device="cuda")
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                done = 0
                for ch in chunks:
                    if done >= rows:
                        break
                    nr = min(ch.shape[0], rows - done)
                    assign_rows(ch[:nr].to(tdt) if xcast else ch[:nr], c, zero, spherical=True)
                    done += nr
                e1.record()
                torch.cuda.synchronize()
                return e0.elapsed_time(e1)

            bl = f"all ({rows})" if b == "all" else str(rows)
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                labels, _ = run(True)
            except Exception as err:  # faiss picks a config the GPU cannot hold
                msg = str(err).splitlines()[0][:90]
                print(f"| triton-{dt} | {bl} | {k} | fails: {type(err).__name__}: {msg} | | | | | | |", flush=True)
                torch.cuda.empty_cache()
                continue
            first = (time.perf_counter() - t) * 1e3
            times = []
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                run(False)
                times.append((time.perf_counter() - t) * 1e3)
            ms = float(np.median(times))
            kms = float(np.median([kernel_ms() for _ in range(a.reps)]))
            # Exactness against the f64 reference (labels only; Triton scores are -2 x.c).
            mism = nontie = 0
            gap = 0.0
            done = 0
            for ci, ch in enumerate(chunks):
                if done >= rows:
                    break
                nr = min(ch.shape[0], rows - done)
                lab = labels[done:done + nr].cuda()
                rl = ref_l[ci][:nr]
                bad = (lab != rl).nonzero().flatten()
                if bad.numel():
                    s = ref_s[ci]
                    g = s[bad, rl[bad]] - s[bad, lab[bad]]
                    mism += bad.numel()
                    nontie += int((g > 1e-5).sum())
                    gap = max(gap, float(g.max()))
                done += nr
            gf = 2.0 * rows * k * d / (ms * 1e6)
            print(f"| triton-{dt} | {bl} | {k} | {ms:.2f} | {kms:.2f} | {gf:.0f} | {first:.0f} | {mism} | {nontie} | {gap:.2e} |", flush=True)
    del ref_s, ref_l, c64
    torch.cuda.empty_cache()
