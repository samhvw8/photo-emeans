"""Summarize the training window of an nsys cuda_gpu_trace CSV.
Window = from the end of the largest host-to-device copy (bind_data) to the last op."""
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
ev = []
for r in rows:
    s = int(r["Start (ns)"]); d = int(r["Duration (ns)"]); name = r["Name"]
    ev.append((s, d, name, int(r.get("Bytes (MB)") and float(r["Bytes (MB)"]) * 1e6 or 0)))
ev.sort()
big = max((e for e in ev if "Host-to-Device" in e[2]), key=lambda e: e[3])
t0 = big[0] + big[1]
win = [e for e in ev if e[0] >= t0]
t1 = max(s + d for s, d, *_ in win)
tot = collections.defaultdict(lambda: [0, 0])
for s, d, n, _ in win:
    key = n.split("(")[0][:40]
    tot[key][0] += d; tot[key][1] += 1
busy = 0; idle = 0; gaps = []; end = t0
for s, d, n, _ in win:
    if s > end:
        idle += s - end; gaps.append((s - end, n))
    busy += max(0, s + d - max(s, end)); end = max(end, s + d)
wall = t1 - t0
print(f"window {wall/1e6:.2f} ms, gpu busy {busy/1e6:.2f} ms, idle {idle/1e6:.2f} ms")
for k, (d, c) in sorted(tot.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:42s} {d/1e6:9.2f} ms  {c:6d} x  {d/c/1e3:8.1f} us")
gaps.sort(reverse=True)
print("  largest gaps (us):", ", ".join(f"{g/1e3:.0f} before {n.split('(')[0][:18]}" for g, n in gaps[:8]))
print(f"  gaps > 10us: {sum(1 for g,_ in gaps if g>1e4)}, total {sum(g for g,_ in gaps if g>1e4)/1e6:.2f} ms")
