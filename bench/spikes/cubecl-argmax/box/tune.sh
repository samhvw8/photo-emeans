#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH CUDA_HOME=/usr CUDA_COMPUTE_CAP=86 CARGO_TARGET_DIR=$PWD/target/spike
cargo +1.99.1 build --release --features cuda 2>&1 | tail -1
B=target/spike/release/cubecl-argmax; D=../../data/s768_200k.f32
for kern in mma-bf16 mma-tf32; do
for t in 64,64,32,2,2 128,64,32,4,2 64,128,32,2,4 128,128,32,4,4 128,64,64,4,2; do
  [ $kern = mma-tf32 ] && t=$(echo $t | awk -F, '{print $1","$2",16,"$4","$5}')
  echo "== $kern tile $t"
  EMEANS_TILE=$t flock /tmp/emeans-box-gpu.lock $B --runtime cuda --data $D --kernels $kern --ks 1024 --bs all --reps 5 --threads 32 2>&1 | grep -E "^\| cuda|Error|error|Caused" | head -4
done; done
echo DONE
