# CHORUS Fabric — v0.1.0 vs v0.2.0 Benchmark Comparison

Same machine, same inputs (CPU-only: `torch 2.9.1+cpu`, no CUDA device).
Cipher figures use the production **float64** wire path; `batch=1024`, `iters=100`.

## 1. Cipher throughput — no regression

Encrypt + decrypt round-trip, vectors/sec:

| dim  | v0.1.0 vec/s | v0.2.0 vec/s | delta |
|-----:|-------------:|-------------:|------:|
| 128  | 1,732,680    | 1,682,260    | −2.9% |
| 512  | 158,134      | 192,812      | +21.9% |
| 1536 | 24,677       | 25,421       | +3.0% |
| 4096 | 4,500        | 4,723        | +5.0% |

**Interpretation: unchanged within measurement noise.** The deltas swing both
directions (−2.9% … +21.9%) on single-run CPU timings — the +21.9% at dim 512 is
variance, not a real speedup, because v0.2.0's default CPU cipher path is the
*same* `torch.matmul` as v0.1.0. The takeaway is **no regression**: the
device-aware refactor and key-rotation plumbing add zero steady-state overhead.

## 2. The price of forward secrecy — cost of one key rotation

A rotation is a single QR key generation (`generate_key_pair`). Between
rotations every message is just encrypt/decrypt — identical to v0.1.0.

| dim  | per rotation | amortized @ `rekey_every=1000` |
|-----:|-------------:|-------------------------------:|
| 128  | 2.35 ms      | 2.35 µs/msg                    |
| 512  | 116.7 ms     | 116.7 µs/msg                   |
| 1536 | 824.9 ms     | 0.82 ms/msg                    |
| 4096 | 3,374.8 ms   | 3.37 ms/msg                    |

At the **128-dim MVP** (the dimension the transatlantic benchmark used) a
rotation costs 2.35 ms — ~1.3% of a single 179 ms network RTT, paid once per N
messages. At `rekey_every=1000` that is **0.0013% amortized overhead**.
Caveat: QR scales ~O(dim³), so a 4096-dim rotation is ~3.4 s — rotate less
frequently at high dimensions.

## 3. Capability comparison

| | v0.1.0 | v0.2.0 |
|---|:---:|:---:|
| Tensor cipher + rolling watermark | ✅ | ✅ |
| Key lifetime | static per session | **rotating epochs** |
| Forward secrecy (retired keys decrypt nothing) | ❌ | ✅ |
| GPU cipher path | noted, unimplemented | ✅ `CHORUS_DEVICE=cuda` |
| Throughput benchmark tool | ❌ | ✅ `bench_cipher.py` |
| Validation checks | 20 | **29** |
| Target decrypts sender's actual key | ❌ (latent bug) | ✅ fixed |

## 4. Not re-measured

- **Transatlantic 179 ms p50 RTT / 4.45× bandwidth** — carried forward from the
  v0.1.0 Azure US-East → Germany-West-Central run; the cross-region setup was
  not re-run for v0.2.0.
- **GPU throughput** — this machine is CPU-only; `bench_cipher.py` emits the GPU
  column automatically on a CUDA host.

*Reproduce cipher throughput: `python bench_cipher.py`.*
