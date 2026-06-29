# chorus-fabric

**High-dimensional tensor communication fabric for AI-to-AI signaling.**

[![PyPI version](https://img.shields.io/pypi/v/chorus-fabric.svg)](https://pypi.org/project/chorus-fabric/)
[![Python](https://img.shields.io/pypi/pyversions/chorus-fabric.svg)](https://pypi.org/project/chorus-fabric/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending%2064%2F096%2C156-blue.svg)]()

---

Most AI agents talk to each other using HTTP and JSON — serializing embeddings to text, sending them over REST, and parsing them back. That process is slow, wasteful, and lossy.

**CHORUS Fabric** is a different approach: AI agents stream raw `float32` embedding vectors directly over bidirectional gRPC, with a built-in tensor multiplication cipher and per-message cryptographic watermark. No tokenization. No JSON. No overhead.

> Deployed transatlantic (US East → Germany West Central): **179 ms p50 RTT**, **4.45× less bandwidth than HTTP/REST**, **100% watermark verification** across 7,766 transmissions.

> *Patent Pending — US Provisional Application No. 64/096,156*

---

## Installation

```bash
pip install chorus-fabric
```

PyTorch is required. If you don't have it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chorus-fabric
```

---

## Quick Start

### 1. Start the services (Docker — easiest)

```bash
git clone https://github.com/aminparva84/chorus-fabric
cd chorus-fabric
docker-compose up
```

This starts the Control Plane (`:50051`), Relay Node (`:50052`), and Target Pod (`:50053`) locally.

### 2. Send your first tensor

```python
import torch
from chorus_fabric import ChorusClient

# Connect to the fabric
client = ChorusClient(
    pod_id="my-agent",
    control_plane="localhost:50051",
    relay="localhost:50052",
)

# Get an ephemeral session key from the Control Plane
client.handshake()

# Send a 128-dimensional embedding vector
signal = torch.randn(128)
acks = client.send_direct(signal)

print(acks)
# [{'seq': 0, 'forwarded': True, 'status': 'ok'}]
```

That's it. The tensor is encrypted with a session-specific matrix cipher, watermarked with a SHA-256-derived authentication vector, streamed over gRPC, and verified at the receiver — all in one call.

---

## Core Concepts

### Tensor Multiplication Cipher

Every signal is encrypted by matrix multiplication before transmission:

```
V_enc = V_raw @ K        # at the sender
V_raw = V_enc @ K_inv    # at the receiver
```

Keys `K` and `K_inv` are generated via QR decomposition — mathematically guaranteed to invert. Both are ephemeral (TTL: 1 hour by default) and issued fresh per session by the Control Plane.

### Rolling Neural Watermark

Every payload carries a deterministic authentication vector:

```
watermark(n) = normalize( seeded_random( SHA-256(session_seed ‖ seq_num) ) )
```

The receiver recomputes the expected watermark independently and checks cosine similarity ≥ 0.95. A replayed or tampered payload fails immediately — the watermark changes every sequence number.

### Three Transmission Modes

| Mode | API call | Use case |
|------|----------|----------|
| **Direct** | `send_direct(tensor)` | Single agent → single target |
| **Isolation (Mode A)** | `send_isolation(tensor_a, tensor_b)` | Two agents share one channel, zero crosstalk |
| **Superposition (Mode B)** | `send_superposition(tensor_a, tensor_b)` | Consensus / ensemble blend |

### Dynamic Key Rotation — Forward Secrecy  *(new in v0.2.0)*

`K` is no longer a static per-session matrix. Each key generation is an
**epoch**; rotating mints a fresh QR key, broadcasts it over the existing gRPC
channel (`ControlPlane.RotateKey`), and swaps it in atomically. Every payload is
tagged with the `key_epoch` it was encrypted under, so the receiver decrypts
with the exact generation that sealed it. Rotated-away keys are retired after a
short grace window and can no longer decrypt past **or** future traffic — a key
captured after rotation is worthless. That is forward secrecy.

```python
# Rotate on demand
client.rotate_key()

# …or auto-rotate every N messages
client = ChorusClient(pod_id="agent", control_plane="cp:50051",
                      relay="relay:50052", rekey_every=64)
client.handshake()
client.send_direct(torch.randn(128))   # epoch advances automatically
```

The rotation itself is one more QR decomposition — microseconds — so forward
secrecy is added at **zero steady-state overhead**.

### CUDA Cipher Path  *(new in v0.2.0)*

The cipher *is* a matrix multiply, so it runs unchanged on a GPU. Set the device
explicitly or let it auto-detect:

```python
import os
os.environ["CHORUS_DEVICE"] = "cuda"   # "auto" (default) | "cpu" | "cuda"

from chorus_fabric import cipher_throughput
print(cipher_throughput(dim=1536, batch=1024))   # vectors/sec, GiB/s, µs/vector
```

`resolve_device("cuda")` falls back to CPU with a warning when no GPU is present,
so the same code runs everywhere. The repository's `bench_cipher.py` benchmarks
the cipher across embedding dimensions on CPU vs GPU to quantify the
zero-overhead claim at high dimensions.

---

## Use Cases

### Use Case 1 — AI Agent-to-Agent Communication

Replace HTTP REST calls between LangChain / AutoGen agents with direct tensor streams.

```python
# Agent 1 (source) — e.g. a retrieval agent
import torch
from chorus_fabric import ChorusClient

retrieval_agent = ChorusClient(pod_id="retrieval", control_plane="cp:50051", relay="relay:50052")
retrieval_agent.handshake()

# Tap the last hidden state from your LLM instead of generating random
hidden_state = torch.randn(128)  # replace with model.last_hidden_state
acks = retrieval_agent.send_direct(hidden_state)
```

```python
# Agent 2 (target) — receives the embedding directly
# Run: python -m chorus_fabric.servers target
# The target pod decrypts, verifies, and processes the vector
```

No JSON serialization. No tokenization round-trip. The embedding arrives at the receiving agent exactly as it left the sender.

---

### Use Case 2 — Dual-Agent Isolation (Mode A)

Two agents share a single encrypted channel. Each recovers only its own signal — measured crosstalk: **0.000006%**.

```python
from chorus_fabric import ChorusClient
import torch

client = ChorusClient(pod_id="dual-agent", control_plane="cp:50051", relay="relay:50052")
client.handshake(isolation_mode=True)  # fetches orthogonal projection matrices

agent_a_signal = torch.randn(128)  # Agent A's embedding
agent_b_signal = torch.randn(128)  # Agent B's embedding

# Both signals travel as a single encrypted payload
acks = client.send_isolation(agent_a_signal, agent_b_signal)
# At the receiver: W_A @ V_dec recovers A's signal, W_B @ V_dec recovers B's
```

**When to use this:** Multi-agent pipelines where two agents need to coordinate over the same network channel without exposing each other's signals to the relay.

---

### Use Case 3 — Ensemble / Consensus Voting (Mode B)

Blend multiple agent signals into one collective transmission.

```python
from chorus_fabric import ChorusClient
import torch

client = ChorusClient(pod_id="ensemble", control_plane="cp:50051", relay="relay:50052")
client.handshake()

# Three models vote on a decision — blend their hidden states
model_a_vote = torch.randn(128)
model_b_vote = torch.randn(128)

# Send the superposed collective state in a single payload
acks = client.send_superposition(model_a_vote, model_b_vote)
```

**When to use this:** Ensemble inference, multi-model voting, distributed sensor fusion — any case where the aggregate state matters more than individual signals.

---

### Use Case 4 — Secure Relay (Confidential Multi-Hop)

A Relay Node amplifies and forwards signals without ever decrypting them. The relay logs a SHA-256 fingerprint of each ciphertext for audit — but the plaintext is invisible to it.

```python
# Relay operates transparently — no code changes needed on the client
# The relay sits between source and target:
#   source -> relay (amplify + audit log) -> target

# From the client's perspective it's identical to direct:
client = ChorusClient(pod_id="source", control_plane="cp:50051", relay="relay:50052")
client.handshake()
acks = client.send_direct(my_tensor)
# relay logs SHA-256(ciphertext) but never sees V_raw
```

**When to use this:** Two companies running a joint AI pipeline — a neutral relay in the middle can audit traffic volume and timing without accessing the content.

---

### Use Case 5 — Crypto Primitives Standalone

Use just the crypto engine without the full gRPC stack:

```python
from chorus_fabric import generate_key_pair, encrypt, decrypt, inject_watermark, verify_watermark
import torch

# Generate a session key pair
K, K_inv = generate_key_pair(dim=128)

# Encrypt a signal
signal = torch.randn(128)
ciphertext = encrypt(signal, K)

# Add a rolling watermark
seed = b'\x00' * 32
authenticated = inject_watermark(signal, seed=seed, seq_num=0)

# Decrypt and verify
recovered = decrypt(ciphertext, K_inv)
is_valid = verify_watermark(recovered, seed=seed, seq_num=0)
print(f"Verified: {is_valid}")  # True
```

---

## Architecture

```
┌─────────────┐     RegisterAndRequestKey      ┌──────────────────┐
│  Your Agent │ ──────────────────────────────► │  Control Plane   │
│  (Source)   │ ◄────────────────────────────── │  :50051          │
│             │      SessionKeyBundle            │  (key issuance)  │
│             │      { K, K_inv, seed, TTL }     └──────────────────┘
│             │
│             │  TensorPayload (V_enc)           ┌──────────────────┐
│             │ ──────────────────────────────► │   Relay Node     │
│             │                                  │   :50052         │
│             │                                  │ • amplify V_enc  │
│             │                                  │ • log SHA-256    │
│             │                                  │ • no decryption  │
│             │                                  └────────┬─────────┘
│             │                                           │ V_amp
│             │                                           ▼
│             │  RelayAck                        ┌──────────────────┐
│             │ ◄──────────────────────────────  │   Target Pod     │
└─────────────┘                                  │   :50053         │
                                                 │ • decrypt        │
                                                 │ • verify wm      │
                                                 │ • mode dispatch  │
                                                 └──────────────────┘
```

---

## Benchmark Results

Live transatlantic deployment — US East (Virginia) → Germany West Central (Frankfurt), ~8,000 km.

| Metric | Value |
|--------|-------|
| Direct mode p50 RTT | **179 ms** |
| Direct mode p95 RTT | 300 ms |
| Mode A (Isolation) p50 | 311 ms |
| Mode B (Superposition) p50 | 1,274 ms |
| Watermark verification | **7,766 / 7,766 (100%)** |
| Cipher overhead vs raw RTT | **0 ms** (matches physical minimum) |

### Bandwidth comparison per 128-dim payload

| Protocol | Bytes/payload | vs CHORUS |
|----------|--------------|-----------|
| **CHORUS gRPC** | **548 B** | 1× |
| HTTP/REST JSON | 2,440 B | 4.45× more |
| LLM API call | 3,900 B | 7.1× more |

---

## Running Services Manually

### Control Plane
```bash
CHORUS_DIM=128 CONTROL_PLANE_PORT=50051 python -m chorus_fabric.servers control_plane
```

### Relay Node
```bash
RELAY_PORT=50052 CONTROL_PLANE_HOST=localhost python -m chorus_fabric.servers relay
```

### Target Pod
```bash
TARGET_PORT=50053 CONTROL_PLANE_HOST=localhost python -m chorus_fabric.servers target
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHORUS_DIM` | `128` | Embedding dimension |
| `CONTROL_PLANE_HOST` | `localhost` | Control plane hostname |
| `CONTROL_PLANE_PORT` | `50051` | Control plane port |
| `RELAY_HOST` | `localhost` | Relay node hostname |
| `RELAY_PORT` | `50052` | Relay node port |
| `CHORUS_TARGET_HOST` | `localhost` | Target pod hostname |
| `CHORUS_TARGET_PORT` | `50053` | Target pod port |
| `CHORUS_SESSION_TTL` | `3600` | Session key TTL in seconds |
| `CHORUS_AMPLIFY_FACTOR` | `1.0` | Relay amplification factor |

---

## Docker Compose (Full Stack)

```yaml
# docker-compose.yml included in repo
services:
  control-plane:
    build: .
    command: python -m chorus_fabric.servers control_plane
    ports: ["50051:50051"]

  relay-node:
    build: .
    command: python -m chorus_fabric.servers relay
    ports: ["50052:50052"]

  target-pod:
    build: .
    command: python -m chorus_fabric.servers target
    ports: ["50053:50053"]
```

---

## Patent Notice

The CHORUS Fabric protocol — including the tensor multiplication cipher, rolling neural watermark, orthogonal isolation mode, holographic superposition mode, relay confidentiality architecture, and control plane key management — is protected under:

**US Provisional Patent Application No. 64/096,156**
*The Chorus Fabric: High-Dimensional Signal Orchestration for Machine-to-Machine Communication*
Filed: June 22, 2026 — Inventor: Amin Parva

This library is released under the MIT License for use by developers. Commercial licensing for embedding the protocol into proprietary products is available — contact **parvaamin@gmail.com**.

---

## License

MIT License — Copyright (c) 2026 Amin Parva

See [LICENSE](LICENSE) for full text.

---

## Contact & Commercial Licensing

**Amin Parva**
parvaamin@gmail.com

For licensing inquiries, enterprise support, or research collaboration, please reach out directly.
