"""Train one faiss e-means (Python) run for bench/compare.

  python run_py.py --impl py-numpy --data X.f32 --d 768 --k 256 --seed 1 --out DIR

Impls:
  py-numpy               NumpyEmeans (faiss.knn on the CPU)
  py-torch-cpu           Emeans on CPU tensors (assignment_backend="native")
  py-torch-mps           Emeans on MPS tensors (assignment_backend="native")
  py-torch-cuda-native   Emeans on CUDA f32 tensors (assignment_backend="native")
  py-torch-cuda-triton   Emeans on CUDA f32 tensors (assignment_backend="triton"); faiss converts to BF16 by default
  py-torch-cuda-*-f32    QuantizedEmeans(quantizer=None): native FP32 operands

The package is faiss contrib/e_means at the pinned commit, fetched by
fetch_upstream.sh (its NumPy files are byte-identical to reference/faiss_e_means).
Writes DIR/centroids.f32 (k x d LE f32) and DIR/impl.json. Load (file read and
host-to-device copy) and train are timed apart; GPU work is synchronized before
the train timer stops.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / ".upstream"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--nepochs", type=float, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    info = {"impl": a.impl, "seed": a.seed, "k": a.k, "d": a.d, "nepochs": a.nepochs,
            "versions": {"numpy": np.__version__}, "threads": {}}
    if a.impl == "py-numpy":
        import faiss
        info["versions"]["faiss"] = faiss.__version__
        info["threads"]["faiss_omp"] = faiss.omp_get_max_threads()
    else:
        # The torch path never calls faiss, but the package __init__ imports
        # e_means.py, which imports faiss. On macOS faiss-cpu and torch each bundle
        # libomp, and loading both aborts (OMP Error #15). A placeholder module
        # keeps faiss out of torch runs; any faiss call would raise.
        import types
        stub = types.ModuleType("faiss")
        def _no_faiss(name):
            raise AttributeError(f"faiss.{name}: faiss is not loaded in torch runs")
        stub.__getattr__ = _no_faiss
        sys.modules["faiss"] = stub

    t = time.perf_counter()
    x = np.fromfile(a.data, dtype="<f4").reshape(-1, a.d)
    sync = lambda: None
    dev = None
    if a.impl != "py-numpy":
        import torch
        info["versions"]["torch"] = torch.__version__
        info["versions"]["torch_cuda"] = torch.version.cuda
        info["threads"]["torch"] = torch.get_num_threads()
        dev = {"py-torch-cpu": "cpu", "py-torch-mps": "mps"}.get(a.impl, "cuda")
        x = torch.from_numpy(x).to(dev)
        if dev == "mps":
            sync = torch.mps.synchronize
        elif dev == "cuda":
            sync = torch.cuda.synchronize
            torch.cuda.reset_peak_memory_stats()
            info["gpu"] = torch.cuda.get_device_name()
            import triton
            info["versions"]["triton"] = triton.__version__
        sync()
    info["load_s"] = time.perf_counter() - t
    n = x.shape[0]

    def make(nepochs):
        if a.impl == "py-numpy":
            from faiss_e_means import NumpyEmeans
            return NumpyEmeans(a.d, a.k, nepochs=nepochs, seed=a.seed, spherical=True)
        backend = "triton" if "triton" in a.impl else "native"
        if a.impl.endswith("-f32"):
            # Native FP32 operands on CUDA (faiss's default converts FP32 CUDA input to BF16).
            from faiss_e_means import QuantizedEmeans
            return QuantizedEmeans(a.d, a.k, nepochs=nepochs, seed=a.seed, spherical=True,
                                   assignment_backend=backend, quantizer=None)
        from faiss_e_means import Emeans
        return Emeans(a.d, a.k, nepochs=nepochs, seed=a.seed, spherical=True,
                      assignment_backend=backend)

    # In-process warm-up, not timed: one epoch on the first max(4k, 4096) rows with the
    # same class, k, backend and representation. It pays one-time costs (CUDA context
    # and cuBLAS handles, Triton JIT/autotune, MPS graphs, BLAS thread pools) that are
    # not training work. Every Python and Rust impl does the same.
    t = time.perf_counter()
    make(1).train(x[: min(n, max(4 * a.k, 4096))])
    sync()
    info["warm_s"] = time.perf_counter() - t

    m = make(a.nepochs)
    t = time.perf_counter()
    m.train(x)
    sync()
    info["train_s"] = time.perf_counter() - t

    c = m.centroids
    if hasattr(c, "detach"):
        c = c.detach().float().cpu().numpy()
    np.ascontiguousarray(c, dtype="<f4").tofile(out / "centroids.f32")
    info["n"] = n
    info["recipe"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                      for k, v in (m.recipe or {}).items()}
    info["n_splits"] = int(m.n_splits)
    info["obj"] = [float(s["obj"]) for s in m.iteration_stats]
    if dev == "cuda":
        import torch
        info["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        info["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    if dev == "mps":
        import torch
        info["mps_driver_allocated_bytes_end"] = int(torch.mps.driver_allocated_memory())
    (out / "impl.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
