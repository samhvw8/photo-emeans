#!/usr/bin/env bash
# One-time setup of bench/compare on the Linux GPU box. Run from the repo root in tmux:
#   tmux new-session -d -s emeans-setup "bash bench/compare/box_setup.sh 2>&1 | tee bench/compare/setup.log; echo DONE; exec bash"
# Installs nothing system-wide: Bend 2.0.40 via `mise install` (no config edit),
# a project venv (bench/compare/py/.venv), cargo builds under target/bench.
# Data: synthetic only (bench/py/gen_data.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
step() { echo; echo "=== $(date +%T) $*"; }

step "bend and clang"
mise install github:bendlang/bend@2.0.40
mise install conda:clang@19.1.7
# One environment for setup and later builds: Bend, clang, CUDA paths, CARGO_TARGET_DIR.
source bench/compare/box_env.sh
BEND="$(mise where github:bendlang/bend@2.0.40)/bin/bend"
"$BEND" --version || true

step "python env"
(cd bench/compare/py && uv sync)
bench/compare/py/.venv/bin/python -c "import torch, faiss, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), 'triton', triton.__version__, 'faiss', faiss.__version__)"

step "upstream e_means"
bench/compare/fetch_upstream.sh

step "rust bench binaries (cuda, embed)"
cargo build --release --manifest-path bench/compare/rs/Cargo.toml --features cuda,embed 2>&1 | tail -30 || {
  echo "BUILD cuda+embed FAILED; trying cuda only"
  cargo build --release --manifest-path bench/compare/rs/Cargo.toml --features cuda 2>&1 | tail -30
}

step "bend build"
(cd bend && mkdir -p build && "$BEND" emeans.bend -o build/emeans 2>&1 | tail -20)
ls -la bend/build

step "data"
for spec in "s768_16k 16000 200 17" "s768_100k 100000 400 3" "s768_200k 200000 500 5" "s768_1m 1000000 1000 23"; do
  set -- $spec
  [ -s "bench/data/$1.f32" ] || uv run --python 3.12 bench/py/gen_data.py --name "$1" --n "$2" --d 768 --centers "$3" --seed "$4"
done
ls -la bench/data
