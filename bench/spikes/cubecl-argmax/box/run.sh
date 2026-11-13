#!/usr/bin/env bash
# Box run of the CubeCL spike: CubeCL CUDA vs faiss Triton (assignment only) vs emeans candle CUDA.
# Run inside tmux: tmux new-session -d -s emeans-cubecl "bash bench/spikes/cubecl-argmax/box/run.sh 2>&1 | tee .../run.log; exec bash"
# Synthetic data only (bench/py/gen_data.py, the project generator). Prints DONE at the end.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SPIKE="$ROOT/bench/spikes/cubecl-argmax"
OUT="$SPIKE/results"
mkdir -p "$OUT" "$SPIKE/target/data"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$ROOT"

# 1. Data: the same synthetic s768_200k as on the Mac (seed 5), plus 4,096 / 65,536-row prefixes.
[ -s bench/data/s768_200k.f32 ] || uv run --python 3.12 bench/py/gen_data.py --name s768_200k --n 200000 --d 768 --centers 500 --seed 5
sha256sum bench/data/s768_200k.f32 | tee "$OUT/box-data.sha256"
for b in 4096 65536; do
  head -c $((b * 768 * 4)) bench/data/s768_200k.f32 > "$SPIKE/target/data/b$b.f32"
  echo "{\"n\": $b, \"d\": 768, \"spherical\": true}" > "$SPIKE/target/data/b$b.json"
done

# 2. Build the spike (CUDA runtime) and record build time and binary size.
export CUDA_HOME=/usr CUDA_COMPUTE_CAP=86
( cd "$SPIKE" && export CARGO_TARGET_DIR="$SPIKE/target/spike" && /usr/bin/time -v cargo +1.99.1 build --release --features cuda 2> "$OUT/box-build.time" | tail -2 )
grep -E "Elapsed|Maximum resident" "$OUT/box-build.time"
ls -la "$SPIKE/target/spike/release/cubecl-argmax" | tee "$OUT/box-binary.txt"

# 3. Build emeans bench_assign with candle CUDA (the box's existing crates copy).
( source bench/compare/box_env.sh && export CARGO_TARGET_DIR="$SPIKE/target/spike" && cargo +1.99.1 build --release -p emeans --no-default-features --features cuda --example bench_assign 2>&1 | tail -2 )

LOCK=/tmp/emeans-box-gpu.lock
BIN="$SPIKE/target/spike/release/cubecl-argmax"
# 4. CubeCL CUDA: scalar FMA (exact f32), cmma tf32 and bf16 (tensor cores), and cmma f32 (expected unsupported).
for kern in ${KERNELS:-fma mma-tf32 mma-bf16 mma-f32}; do
  flock "$LOCK" "$BIN" --runtime cuda --data bench/data/s768_200k.f32 --kernels "$kern" --ks 64,256,1024 \
    --bs 4096,65536,all --reps 7 --threads 32 > "$OUT/box-cubecl-$kern.md" 2> "$OUT/box-cubecl-$kern.log"
  echo "cubecl $kern exit $?"; cat "$OUT/box-cubecl-$kern.md"; tail -3 "$OUT/box-cubecl-$kern.log"
done
[ "${SKIP_TRITON:-0}" = 1 ] || {
# 5. faiss Triton, assignment only.
flock "$LOCK" bench/compare/py/.venv/bin/python "$SPIKE/box/triton_assign.py" --data bench/data/s768_200k.f32 \
  --upstream bench/compare/.upstream --reps 7 > "$OUT/box-triton.md" 2> "$OUT/box-triton.log"
echo "triton exit $?"; cat "$OUT/box-triton.md"; tail -3 "$OUT/box-triton.log"
}
# 6. emeans candle CUDA (unfused matmul + argmax).
rm -f "$OUT/box-candle-cuda.md"
BA="$SPIKE/target/spike/release/examples/bench_assign"
for ds in "$SPIKE/target/data/b4096" "$SPIKE/target/data/b65536" "$ROOT/bench/data/s768_200k"; do
  flock "$LOCK" "$BA" --data "$ds" --ks 64,256,1024 --backends cuda --reps 7 >> "$OUT/box-candle-cuda.md" 2>&1
done
cat "$OUT/box-candle-cuda.md"
echo DONE
