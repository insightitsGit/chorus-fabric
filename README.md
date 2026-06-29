# The Chorus Fabric

**Patent-Pending AI-to-AI Tensor Communication Protocol**

> Stream raw float32 embeddings between AI agents over gRPC — no tokenization, no JSON, zero cipher overhead.

[![PyPI version](https://img.shields.io/pypi/v/chorus-fabric?color=1A365D&label=chorus-fabric)](https://pypi.org/project/chorus-fabric/)
[![Python](https://img.shields.io/pypi/pyversions/chorus-fabric?color=9A7B3A)](https://pypi.org/project/chorus-fabric/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/USPTO-Patent%20Pending%2064%2F096%2C156-1A365D)](https://pypi.org/project/chorus-fabric/)

```bash
pip install chorus-fabric
```

---

## The Problem

Every multi-agent AI system today wastes bandwidth and latency on the **token boundary**:

```
Agent A output → serialize embeddings to JSON text
               → send over HTTP (2,440 bytes for 128-dim)
               → Agent B receives text
               → re-embed back to float32
               → process
```

CHORUS removes that entire round-trip.

---

## The Solution

```
Agent A output → float32 vector
               → encrypt: V_enc = V_raw @ K   (matrix multiply — zero overhead)
               → gRPC binary stream (548 bytes for 128-dim)
               → Agent B decrypts: V_dec = V_enc @ K_inv
               → verify SHA-256 watermark
               → process
```

The cipher **is** a matrix multiply — the same operation every neural network already performs. No separate crypto co-processor. CUDA-accelerable natively.

---

## New in v0.2.0 — Zero-Overhead Encryption **+ Forward Secrecy**

**Dynamic key rotation.** `K` is no longer static. Each key generation is an
*epoch*; rotating mints a fresh QR key, broadcasts it over the existing gRPC
channel, and swaps it in atomically. Every payload carries the `key_epoch` it
was sealed under, and rotated-away keys are retired after a short grace window —
so a key captured after rotation decrypts nothing, past or future.

```python
client.rotate_key()                                   # rotate on demand
client = ChorusClient(..., rekey_every=64)            # or auto-rotate every 64 msgs
```

**CUDA cipher path.** The cipher is a matrix multiply, so it now runs unchanged
on a GPU (`CHORUS_DEVICE=cuda`, or `auto` to detect). The bundled
`bench_cipher.py` benchmarks cipher throughput on CPU vs GPU across embedding
dimensions, proving the zero-overhead claim scales past 4096-dim vectors.

> Rotation is one extra QR decomposition (microseconds). Forward secrecy is
> added at **zero steady-state overhead** — the v0.2.0 story in one line:
> *zero-overhead encryption + forward secrecy.*

See [CHANGELOG.md](CHANGELOG.md) for the full v0.2.0 change list.

**v0.1.0 → v0.2.0 (CPU):** cipher throughput unchanged within noise (e.g. 128-dim
~1.7M vec/s both versions); forward secrecy costs only the rotation event — 2.35 ms
at 128-dim, ~0.0013% amortized at `rekey_every=1000`. Full table:
[results/v010_vs_v020_comparison.md](results/v010_vs_v020_comparison.md).

---

## Live Benchmark

Tested transatlantic: **US East (Virginia) → Germany West Central (Frankfurt)** on Azure Container Instances.

| Metric | CHORUS Fabric | HTTP / REST | LLM API Calls |
|--------|:------------:|:-----------:|:-------------:|
| p50 Round-Trip Latency | **179 ms** | ~320 ms | ~800+ ms |
| Payload (128-dim float32) | **548 bytes** | 2,440 bytes | ~3,450 bytes |
| Bandwidth savings | **baseline** | 4.45× more | 7.1× more |
| Cipher overhead | **0 ms** | N/A | N/A |
| Watermark verification | **100%** | N/A | N/A |
| Transmissions tested | **7,766** | — | — |

> 179 ms matches the physical minimum for US–EU distance. The cipher and watermark add **zero measurable latency**.

---

## Installation

```bash
pip install chorus-fabric
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, gRPC ≥ 1.64

---

## Quick Start

```python
from chorus_fabric import ChorusClient
import numpy as np

client = ChorusClient(control_plane_host="your-control-plane", port=50051)
await client.handshake()

vector = np.random.randn(128).astype(np.float32)

# Mode: Direct — encrypted point-to-point
response = await client.send_direct(vector, target="agent-b")

# Mode A: Orthogonal Isolation — two agents, one channel, 0.000006% crosstalk
response = await client.send_isolation(vector, channel="shared-ch-1")

# Mode B: Holographic Superposition — collective broadcast
response = await client.send_superposition(vector, swarm="agent-cluster-1")
```

---

## Architecture

### Four-Node Topology

```
┌─────────────────────────────────────┐
│       Control Plane  :50051         │
│  SessionKeyBundle · TTL · QR keys   │
│  ProjectionBundle (Mode A)          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Source Pod / Client                │
│  handshake() → encrypt → stream     │
└──────────────┬──────────────────────┘
               │  gRPC binary stream
┌──────────────▼──────────────────────┐
│  Relay Node  :50052                 │
│  V_amp = factor × V_enc             │
│  holds NO K_inv · SHA-256 audit log │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Target Pod  :50053                 │
│  decrypt · verify watermark · use   │
└─────────────────────────────────────┘
```

### Three Communication Modes

#### Direct
Standard encrypted point-to-point. The cipher uses a QR-decomposed orthogonal key pair:
```
Encrypt:  V_enc = V_raw @ K
Decrypt:  V_dec = V_enc @ K_inv
```

#### Mode A — Orthogonal Isolation
Two agents share one gRPC channel with mathematically guaranteed separation:
```
W_A @ W_B ≈ 0   →   0.000006% crosstalk (measured)
```
Each agent gets a projection matrix from the Control Plane. Perfect signal recovery from a shared channel. Analogous to wavelength-division multiplexing in fiber optics, but for AI signals.

#### Mode B — Holographic Superposition
Multiple agent signals combined into a single collective vector:
```
V_collective = V_A + V_B
```
~0.70 cosine similarity recovery per agent. Enables broadcast and swarm architectures.

### Rolling SHA-256 Neural Watermark
Every message carries a watermark injected into the vector itself — not a header, not a separate field:
```
watermark = SHA-256(session_seed ‖ seq_number) → seeded RNG → unit vector
verify:     cosine_similarity(received, expected) ≥ 0.95
```
**100% verification rate** across 7,766 consecutive transmissions. Any tampering breaks the cosine check immediately at the math layer.

### Relay Node — Zero-Knowledge Middle Layer
The relay amplifies signals without ever holding decryption keys:
- Receives and forwards: `V_amp = factor × V_enc`
- Holds: **no K_inv**
- Produces: SHA-256 audit fingerprint per relay event
- Use case: multi-tenant AI infrastructure where the relay operator must not see content

### Dynamic Key Rotation — Forward Secrecy
The Control Plane tracks a rolling set of key generations (*epochs*) per session:
```
RotateKey(session)  →  mint fresh QR key, epoch += 1, retire epochs beyond grace
payload.key_epoch   →  receiver fetches that exact generation to decrypt
```
- Rotation is driven by the client (`rotate_key()`) or automatically (`rekey_every=N`).
- A short grace window keeps the previous epochs decryptable so in-flight messages survive a rotation; older epochs are dropped and can never decrypt again.
- Result: **forward secrecy** — compromising the current key exposes neither past nor future traffic.

### CUDA Cipher Path
The cipher is `V_enc = V_raw @ K`, so the GPU fast-path is the same `torch.matmul` on a CUDA tensor:
```
CHORUS_DEVICE = auto | cpu | cuda      # auto detects a GPU, falls back to CPU
```
`bench_cipher.py` measures cipher throughput (vectors/sec, GiB/s, µs/vector) on every available device. On CPU the cipher costs ~0.08 µs/vector at 128-dim and ~9 µs at 1536-dim — orders of magnitude below the 179 ms network RTT, which is exactly what "zero measurable overhead" means.

---

## Use Cases

### 1. Multi-Agent AI Pipelines (LangGraph, AutoGen, CrewAI)
Replace HTTP calls between agents with CHORUS streams. Remove the tokenization round-trip entirely. Every agent-to-agent message is cryptographically watermarked with zero overhead.

### 2. Real-Time Inference Clusters
Connect inference nodes with a persistent bidirectional gRPC fabric. The cipher runs on the same GPU doing inference — it's just a matrix multiply.

### 3. Multi-Tenant AI Infrastructure
Relay node operates on ciphertext only. SHA-256 audit fingerprint on every relay event. Orthogonal isolation gives each tenant their own signal lane on shared hardware.

### 4. Agent Authentication (Watermark-First)
Prove message origin without PKI certificates. The watermark is woven into the vector itself — tamper-evident at the math layer. No headers. No tokens. No separate auth layer.

### 5. Distributed AI Research
Mode B superposition enables swarm architectures where multiple agents contribute to a single collective signal with recoverable individual contributions.

---

## Competitive Comparison

| | CHORUS Fabric | Standard gRPC | HTTP/REST | LLM API |
|--|:---:|:---:|:---:|:---:|
| Tensor-native (no serialization) | ✅ | ❌ | ❌ | ❌ |
| Built-in encryption (in the math) | ✅ | ❌ | ❌ | ❌ |
| Per-message watermark / auth | ✅ | ❌ | ❌ | ❌ |
| Orthogonal channel sharing | ✅ | ❌ | ❌ | ❌ |
| Zero-knowledge relay | ✅ | ❌ | ❌ | ❌ |
| Bandwidth vs HTTP/REST | **4.45× less** | ~2× less | baseline | 7.1× more |
| Cipher overhead | **0 ms** | N/A | N/A | N/A |

---

## Patent

**USPTO Provisional Patent Application No. 64/096,156**
- Title: *The Chorus Fabric: High-Dimensional Signal Orchestration for Machine-to-Machine Communication*
- Inventor: Amin Parva
- Filed: June 22, 2026
- Confirmation Number: 7452

**Protected claims include:**
1. Tensor multiplication cipher using QR-decomposed orthogonal key pairs
2. Rolling SHA-256 neural watermark for per-message authentication
3. Orthogonal isolation mode — shared-channel multi-agent communication with crosstalk elimination
4. Holographic superposition mode — collective vector construction and signal recovery
5. Zero-knowledge relay node architecture
6. Full 4-node control plane topology with ephemeral SessionKeyBundle and TTL

---

## PyPI

**https://pypi.org/project/chorus-fabric/**

```bash
pip install chorus-fabric
```

---

## Author

**Amin Parva** — AI Solution Architect & Inventor
Insight IT Solutions LLC · Mission Viejo, CA
parvaamin@gmail.com
https://insightits.com

Also inventor of:
- [PrismLang](https://insightits.com/prismlang) — deterministic vector protocol for LangGraph (~60% token reduction)
- [PrismRAG](https://prismrag.insightits.com) — mapping-first enterprise Graph RAG replacement

---

## License

MIT — see [LICENSE](LICENSE) for details.

Patent rights reserved. Commercial use of the patented protocol requires a license.
Contact: **parvaamin@gmail.com**

---

## Licensing & Partnerships

Interested in integrating CHORUS Fabric into your AI infrastructure?
Open to: research licenses, commercial licenses, integration partnerships, and acquisition discussions.

**Contact: parvaamin@gmail.com**
