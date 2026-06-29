"""
CHORUS Protocol – bench_cipher.py   [v0.2.0]
==============================================
Local cipher-throughput benchmark: CPU vs GPU.

The CHORUS cipher *is* a matrix multiply (V_enc = V_raw @ K), so it runs
unchanged on a GPU. This script quantifies the "zero-overhead" claim by timing
the cipher across embedding dimensions on every available device, reporting
vectors/sec, throughput (GiB/s) and per-vector latency.

Unlike the cross-region gRPC benchmark (which measures network RTT), this
isolates the *compute* cost of the cipher itself — the part the patent calls
"zero overhead". On a CUDA box it also proves the GPU fast-path scales to high
dimensions that are impractical on CPU.

Usage:
    python bench_cipher.py
    CHORUS_BENCH_DIMS=128,512,1536,4096 python bench_cipher.py
"""

import json
import os
from datetime import datetime, timezone

import torch

import crypto_engine as ce

DIMS   = [int(x) for x in os.getenv("CHORUS_BENCH_DIMS", "128,512,1536,4096").split(",")]
BATCH  = int(os.getenv("CHORUS_BENCH_BATCH", "1024"))
ITERS  = int(os.getenv("CHORUS_BENCH_ITERS", "100"))
OUT    = os.getenv("CHORUS_BENCH_OUT", "results/cipher_throughput.json")


def _devices():
    devs = ["cpu"]
    if ce.cuda_available():
        devs.append("cuda")
    return devs


def run():
    print("=" * 78)
    print("  CHORUS Cipher Throughput Benchmark  (compute-only, no network)")
    print(f"  torch {torch.__version__}   CUDA available: {ce.cuda_available()}")
    print(f"  batch={BATCH}  iters={ITERS}  dims={DIMS}")
    print("=" * 78)

    devices = _devices()
    if "cuda" not in devices:
        print("  NOTE: no CUDA device detected - GPU column omitted. The GPU")
        print("        fast-path is exercised automatically on a CUDA machine.")
    print()

    header = f"  {'dim':>6}  {'device':>6}  {'vec/sec':>14}  {'GiB/s':>9}  {'us/vector':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for dim in DIMS:
        for dev in devices:
            r = ce.cipher_throughput(dim=dim, batch=BATCH, iters=ITERS, device=dev)
            rows.append(r)
            print(f"  {dim:>6}  {dev:>6}  {r['vectors_per_sec']:>14,.0f}  "
                  f"{r['gib_per_sec']:>9.2f}  {r['us_per_vector']:>10.4f}")
        if "cuda" in devices:  # blank line between dim groups when comparing
            print()

    # Speedup summary if both devices present
    if "cuda" in devices:
        print("  " + "-" * (len(header) - 2))
        print(f"  {'dim':>6}  {'GPU speedup vs CPU':>22}")
        for dim in DIMS:
            cpu = next(x for x in rows if x["dim"] == dim and x["device"] == "cpu")
            gpu = next(x for x in rows if x["dim"] == dim and x["device"] == "cuda")
            sx = gpu["vectors_per_sec"] / cpu["vectors_per_sec"]
            print(f"  {dim:>6}  {sx:>21.1f}x")

    print("\n  Per-vector latency is the cipher's full compute cost. Compare to the")
    print("  179 ms transatlantic RTT: the cipher is orders of magnitude smaller,")
    print("  which is what 'zero measurable overhead' means in the benchmark report.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "torch_version": torch.__version__,
        "cuda_available": ce.cuda_available(),
        "batch": BATCH,
        "iters": ITERS,
        "results": rows,
    }
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results written to {OUT}")
    print("=" * 78)


if __name__ == "__main__":
    run()
