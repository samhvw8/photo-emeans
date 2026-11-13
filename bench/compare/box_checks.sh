#!/usr/bin/env bash
# Side checks on the box after the sweep (results/ubuntu-checks/):
#  1. Does bend-cuda run its ! calls on the GPU? SM utilization during a run, and a
#     control run with BENDRTS="--gpu off".
#  2. py-numpy (faiss.knn) with 64 vs 32 vs 16 OpenMP threads.
#  3. clippy of the emeans library (github.com/samhvw8/emeans, checked out at ../../../../libs/emeans
#     relative to this folder) with the cuda feature.
cd "$(dirname "$0")"
source ./box_env.sh
out=results/ubuntu-checks; mkdir -p "$out"
D=../data/s768_100k.f32
echo "=== bend-cuda SM utilization (100k, k=256, seed 1)"
nvidia-smi dmon -s u -d 1 > "$out/bend_cuda_dmon.txt" & DM=$!
/usr/bin/time -v ../../bend/build/emeans $D 100000 768 256 5 1 1 "$out/bend-cuda" 1 2> "$out/bend_cuda_time.txt" | tail -1
kill $DM; awk 'NR>2 {s+=$2; n++; if ($2>m) m=$2} END {printf "sm%% mean %.1f max %d over %d s\n", s/n, m, n}' "$out/bend_cuda_dmon.txt"
grep -E "Percent of CPU|Elapsed" "$out/bend_cuda_time.txt"
echo "=== bend gpu=1 with BENDRTS=--gpu off"
BENDRTS="--gpu off" /usr/bin/time -v ../../bend/build/emeans $D 100000 768 256 5 1 1 "$out/bend-gpuoff" 1 2> "$out/bend_gpuoff_time.txt" | tail -1
grep -E "Percent of CPU|Elapsed" "$out/bend_gpuoff_time.txt"
echo "=== bend-cpu"
../../bend/build/emeans $D 100000 768 256 5 1 1 "$out/bend-cpu" 0 | tail -1
for f in bend-cuda bend-gpuoff bend-cpu; do python3 -c "import json;r=json.load(open('$out/$f/report.json'));print('$f', r['train_ms'], 'ms', r['obj'][-1])"; done
echo "=== py-numpy OMP threads (100k, k=256, seed 1)"
for t in 64 32 16; do
  OMP_NUM_THREADS=$t py/.venv/bin/python py/run_py.py --impl py-numpy --data $D --d 768 --k 256 --seed 1 --out "$out/numpy-omp$t" >/dev/null
  python3 -c "import json;r=json.load(open('$out/numpy-omp$t/impl.json'));print('omp', r['threads']['faiss_omp'], 'train_s %.2f' % r['train_s'])"
done
echo "=== clippy emeans --features cuda"
(cd ../../../../libs/emeans && cargo clippy --workspace --features emeans/cuda,emeans-cluster/cuda --lib -- -D warnings 2>&1 | grep -E "^(warning|error)|-->|Finished")
