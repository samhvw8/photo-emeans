#!/usr/bin/env bash
# One run of every Linux impl on s768_16k, k=64 (results/box-smoke).
cd "$(dirname "$0")"
source ./box_env.sh
python3 run.py --host box-smoke --impls py-numpy,py-torch-cpu,py-torch-cuda-native,py-torch-cuda-triton,py-torch-cuda-native-f32,py-torch-cuda-triton-f32,rs-cpu-blocked,rs-cpu-candle-gemm,rs-cuda,bend-cpu,bend-cuda \
  --datasets s768_16k --ks 64 --seeds 1 --warmup 0
