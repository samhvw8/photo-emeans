#!/usr/bin/env bash
# Fetch the full faiss contrib/e_means package at the commit pinned in
# reference/faiss_e_means/UPSTREAM_COMMIT. The vendored copy holds only the NumPy
# path; the torch Emeans (e_means_torch.py, triton_assignment.py, ...) comes from here.
# Output: bench/compare/.upstream/faiss_e_means/ (gitignored). Files the vendored copy
# also has must be byte-identical, or this fails.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
commit="$(tr -d '[:space:]' < "$root/reference/faiss_e_means/UPSTREAM_COMMIT")"
out="$here/.upstream/faiss_e_means"
mkdir -p "$out"
base="https://raw.githubusercontent.com/facebookresearch/faiss/$commit/contrib/e_means"
for f in __init__.py base.py cute.py cute_argmin.py cute_grouped.py cute_kernel.py cute_row_owned.py \
         e_means.py e_means_distributed.py e_means_grouping.py e_means_quantized.py e_means_spherical.py \
         e_means_torch.py ivf_flat.py quantizer.py quantizer_fp8.py quantizer_int8.py sharded_centroids.py \
         sharded_centroids_ivf.py torch_backend.py triton_assignment.py; do
  [ -s "$out/$f" ] || curl -fsSL "$base/$f" -o "$out/$f"
done
for f in "$root"/reference/faiss_e_means/*.py; do
  cmp -s "$f" "$out/$(basename "$f")" || { echo "MISMATCH vs vendored: $(basename "$f")" >&2; exit 1; }
done
echo "$commit" > "$out/UPSTREAM_COMMIT"
echo "fetched faiss contrib/e_means @ $commit -> $out"
