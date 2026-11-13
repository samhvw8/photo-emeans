#!/usr/bin/env bash
# Rebuild the bench binaries on the box.
# emeans-compare gets `cuda` only: candle's CUDA (from `embed` or `cuda-baselines`)
# links the CUDA libraries at load time and slows the CPU backends ~10%. For the
# `rs-cuda-unfused` baseline, build it with `--features cuda-baselines` in another
# CARGO_TARGET_DIR and do not time CPU backends with that binary.
set -euo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
cargo build --release --manifest-path rs/Cargo.toml --features cuda --bin emeans-compare 2>&1 | grep -E "^(warning|error)|-->|Finished"
cargo build --release --manifest-path rs/Cargo.toml --features cuda,embed --bin embed-compare 2>&1 | grep -E "^(warning|error)|-->|Finished"
