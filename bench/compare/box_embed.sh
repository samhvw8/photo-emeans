#!/usr/bin/env bash
# Embedding throughput on the box: candle CUDA vs sentence-transformers CUDA, f32 and bf16.
# Builds into its own target dir so the e-means binaries stay untouched.
set -uo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
export CARGO_TARGET_DIR="$(cd ../.. && pwd)/target/bench-embed"
cargo build --release --manifest-path rs/Cargo.toml --features cuda,embed 2>&1 | grep -E "^error|-->|Finished"
py/.venv/bin/python embed_run.py --host ubuntu --cases candle:cuda:f32,st:cuda:f32,candle:cuda:bf16,st:cuda:bf16 --batches 1,8 --n 64
