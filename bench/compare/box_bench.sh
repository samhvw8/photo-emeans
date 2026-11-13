#!/usr/bin/env bash
# Full Linux sweep: 11 impls x {16k, 100k, 200k} x k {64, 256, 1024} x 5 seeds, then n = 1M, k = 1024.
cd "$(dirname "$0")"
source ./box_env.sh
IMPLS=py-numpy,py-torch-cpu,py-torch-cuda-native,py-torch-cuda-triton,py-torch-cuda-native-f32,py-torch-cuda-triton-f32,rs-cpu-blocked,rs-cpu-candle-gemm,rs-cuda,bend-cpu,bend-cuda
python3 run.py --host ubuntu --impls "$IMPLS" --datasets s768_16k,s768_100k,s768_200k --ks 64,256,1024 --seeds 1,2,3,4,5 --timeout 1800
echo "=== main sweep finished $(date +%T)"
python3 run.py --host ubuntu --impls "$IMPLS" --datasets s768_1m --ks 1024 --seeds 1,2,3,4,5 --timeout 2400
