#!/usr/bin/env bash
# Profile emeans-compare training with nsys. Usage: prof.sh TAG backend1,backend2 data1,data2
set -uo pipefail
cd ~/workspace/tools/emeans
source bench/compare/box_env.sh
export CARGO_TARGET_DIR=$PWD/target/cuda2
TAG=$1; BACKENDS=${2:-cuda,cuda-bf16}; DATAS=${3:-s768_200k,s768_1m}
OUT=/tmp/emeans-prof/$TAG; mkdir -p $OUT
BIN=${BIN:-$CARGO_TARGET_DIR/release/emeans-compare}
for data in ${DATAS//,/ }; do for b in ${BACKENDS//,/ }; do
  f=$OUT/$b-$data
  flock /tmp/emeans-box-gpu.lock env ${EXTRA:-EMEANS_X=0} nsys profile -o $f --force-overwrite true -t cuda --sample none --cpuctxsw none \
    $BIN --data bench/data/$data.f32 --d 768 --k 1024 --seed 1 --backend $b --out $OUT/run-$b-$data >/dev/null 2>&1
  /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i $f.qdstrm >/dev/null 2>&1; nsys stats --report cuda_gpu_trace --format csv --output $f --force-export true $f.nsys-rep >/dev/null 2>&1
  echo "== $b $data train_s=$(python3 -c "import json;print(round(json.load(open('$OUT/run-$b-$data/impl.json'))['train_s'],4))")"
  python3 ~/emeans-cuda/analyze.py ${f}_cuda_gpu_trace.csv
done; done
