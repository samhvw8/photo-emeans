import csv, sys
ev = sorted((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"][:30], float(r["Bytes (MB)"] or 0)) for r in csv.DictReader(open(sys.argv[1])))
big = max((e for e in ev if "Host-to-Device" in e[2]), key=lambda e: e[3]); t0 = big[0] + big[1]
win = [e for e in ev if e[0] >= t0]; end = t0; prev = "start"
gaps = []
for s, d, n, b in win:
    if s - end > 1e6: gaps.append(((s - end) / 1e6, (end - t0) / 1e6, prev, n, b))
    end = max(end, s + d); prev = n
print(f"window {(end-t0)/1e6:.1f} ms")
for g, at, a, b, mb in gaps: print(f"  gap {g:6.2f} ms at {at:7.1f} ms: after {a} -> before {b} ({mb} MB)")
