#!/usr/bin/env bash
# Synthetic data sets for bench/compare (same generator, sizes and seeds as on the Mac).
set -euo pipefail
cd "$(dirname "$0")/../.."
while read -r name n centers seed; do
  [ -s "bench/data/$name.f32" ] || uv run --python 3.12 bench/py/gen_data.py --name "$name" --n "$n" --d 768 --centers "$centers" --seed "$seed"
done <<'LIST'
s768_16k 16000 200 17
s768_100k 100000 400 3
s768_200k 200000 500 5
s768_1m 1000000 1000 23
LIST
sha256sum bench/data/*.f32
