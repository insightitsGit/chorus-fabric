# The Chorus Fabric — Product Information Brief
> Use this file to build a product landing page. All numbers are from live transatlantic benchmarks — not estimates.

---

## What It Is

**The Chorus Fabric** is a patent-pending communication protocol that lets AI agents talk to each other using raw math — no text, no tokens, no JSON.

Today every AI agent system converts embeddings (float32 vectors) to text, sends that text over HTTP, and the receiving agent converts it back to embeddings. CHORUS skips that entire round-trip by streaming the float32 vectors directly over a persistent gRPC channel with military-grade encryption baked in at the linear-algebra layer.

**One-line pitch:** CHORUS Fabric is the first tensor-native communication protocol for AI agents — 4.45× less bandwidth, cipher overhead of zero, and a cryptographic watermark on every single message.

**New in v0.2.0 — zero-overhead encryption + forward secrecy.** The session key
is no longer static: it rotates per-session or every N messages (a fresh QR key
broadcast over the existing gRPC channel and swapped atomically), and retired
keys can no longer decrypt past or future traffic. The matrix-multiply cipher
also gained a CUDA fast-path, so throughput can be benchmarked on GPU vs CPU at
high embedding dimensions. Rotation adds forward secrecy at zero steady-state
overhead.

---

## Why It Matters (The Problem)

Modern AI agent systems waste enormous resources on the token boundary:

| Step | What happens today | What CHORUS does |
|------|--------------------|-----------------|
| Agent A generates a response | Float32 embedding → serialize to JSON text | Float32 embedding stays as float32 |
| Send over network | HTTP/REST JSON payload (2,440 bytes for 128-dim) | gRPC binary stream (548 bytes for 128-dim) |
| Agent B receives | Deserialize JSON → re-embed text back to float32 | Float32 arrives directly — done |
| Authentication | None, or separate token/header | Cryptographic watermark embedded in the vector itself |

Every agent-to-agent call in today's systems converts math to words and back to math. CHORUS removes that boundary entirely.

---

## Live Benchmark Results

Tested transatlantic: **US East (Virginia) → Germany West Central (Frankfurt)** on Azure Container Instances.

| Metric | CHORUS Fabric | HTTP / REST | LLM API Calls | CHORUS Wins By |
|--------|--------------|-------------|---------------|----------------|
| **p50 Round-Trip Latency** | **179 ms** | ~320 ms | ~800+ ms | 1.8× – 4.5× |
| **Payload size (128-dim float32)** | **548 bytes** | 2,440 bytes | ~3,450 bytes | **4.45× – 7.1×** |
| **Cipher overhead** | **0 ms** | N/A | N/A | Zero cost |
| **Watermark verification rate** | **100%** | N/A | N/A | 7,766 / 7,766 transmissions |

> The 179 ms transatlantic p50 matches the physical minimum for US–EU distance — meaning the cipher and watermark add literally zero measurable overhead.

---

## How It Works

### The Four-Node Architecture

```
[Control Plane :50051]  ←── Session keys, TTL, ProjectionBundles
        │
[Source Pod / Client]  ──→  [Relay Node :50052]  ──→  [Target Pod :50053]
     (encrypts)              (amplifies ciphertext,        (decrypts,
                              holds no K_inv,               verifies watermark)
                              SHA-256 audit fingerprint)
```

### Tensor Multiplication Cipher
- Key generation: QR decomposition of a random matrix → orthogonal key pair (K, K_inv)
- Encrypt: `V_enc = V_raw @ K`
- Decrypt: `V_dec = V_enc @ K_inv`
- The cipher IS a matrix multiply — the same operation every neural network uses. No separate crypto co-processor needed. CUDA-accelerable natively.

### Rolling SHA-256 Neural Watermark
- Every message carries a watermark: `SHA-256(session_seed ‖ seq_number)` → seeded RNG → unit vector injected into the embedding
- Verification: cosine similarity ≥ 0.95 threshold
- Result: 100% verified across 7,766 consecutive transmissions
- Tamper-evident: any modification breaks the cosine similarity check immediately

### Three Communication Modes

**Mode: Direct**
Standard encrypted point-to-point. Agent A → encrypt → gRPC stream → Agent B decrypts.

**Mode A: Orthogonal Isolation**
Two agents share a single gRPC channel with zero crosstalk.
- Each agent gets a projection matrix: `W_A`, `W_B`
- Mathematical guarantee: `W_A @ W_B ≈ 0`
- Achieved **0.000006% crosstalk** in live tests
- Perfect signal recovery from a shared channel
- Analogy: wavelength-division multiplexing in fiber optics, but for AI signals

**Mode B: Holographic Superposition**
Multiple agent signals are combined into a single collective vector.
- `V_collective = V_A + V_B`
- ~0.70 cosine similarity recovery per agent
- Enables broadcast / swarm architectures

### Relay Node (Zero-Knowledge Middle Layer)
The relay node amplifies signals without ever holding decryption keys.
- Receives: `V_amp = factor × V_enc` (scalar multiply on ciphertext)
- Holds: no K_inv
- Produces: SHA-256 audit fingerprint of every relay event
- Use case: multi-tenant AI infrastructure where the relay operator must not see content

### Control Plane
- Manages ephemeral `SessionKeyBundle` objects with configurable TTL
- Issues QR-decomposed key pairs per session
- Issues `ProjectionBundle` for Mode A (orthogonal projection matrices)
- Handles handshake, key rotation, session teardown

---

## Install & Use

```bash
pip install chorus-fabric
```

**PyPI:** https://pypi.org/project/chorus-fabric/0.1.0/

**Basic usage:**

```python
from chorus_fabric import ChorusClient

client = ChorusClient(control_plane_host="your-control-plane", port=50051)
await client.handshake()

# Send a float32 embedding directly to another agent
import numpy as np
vector = np.random.randn(128).astype(np.float32)

# Direct encrypted send
response = await client.send_direct(vector, target="agent-b")

# Orthogonal isolation — two agents, one channel, zero crosstalk
response = await client.send_isolation(vector, channel="shared-ch-1")

# Holographic superposition — collective broadcast
response = await client.send_superposition(vector, swarm="agent-cluster-1")
```

**Requirements:** Python 3.10+, PyTorch 2.0+, gRPC 1.64+

---

## Use Cases

### 1. Multi-Agent AI Pipelines (LangGraph, AutoGen, CrewAI)
Replace HTTP calls between agents with CHORUS streams. Cut bandwidth by 4.45×. Remove the tokenization round-trip entirely. Every agent-to-agent message is cryptographically watermarked.

### 2. Real-Time AI Inference Clusters
Connect inference nodes with a persistent bidirectional gRPC fabric. The cipher is a linear layer — runs on the same GPU doing inference with zero scheduling overhead.

### 3. Multi-Tenant AI Infrastructure
Relay node operates on ciphertext only — the operator never sees content. SHA-256 audit fingerprint on every relay event. Orthogonal isolation gives each tenant their own signal lane on shared hardware.

### 4. AI Agent Security (Watermark-First Authentication)
Prove message origin without PKI certificates. The watermark is woven into the vector itself — not a header, not a signature, not a separate field. Any tampering breaks it immediately at the math layer.

### 5. Distributed AI Research
Mode B superposition enables researchers to study emergent collective behavior in swarms of AI agents — multiple agents contributing to a single collective signal with recoverable individual contributions.

---

## Patent Information

**USPTO Provisional Patent Application No. 64/096,156**
- Title: *The Chorus Fabric: High-Dimensional Signal Orchestration for Machine-to-Machine Communication*
- Inventor: Amin Parva
- Filed: June 22, 2026
- Confirmation Number: 7452
- Patent Center Reference: 77736418
- Entity Status: Micro Entity
- Non-provisional deadline: June 22, 2027

**What is protected:**
1. Tensor multiplication cipher using QR-decomposed orthogonal key pairs for embedding encryption
2. Rolling SHA-256 neural watermark scheme for per-message authentication
3. Orthogonal isolation mode (Mode A) — shared-channel multi-agent communication with mathematical crosstalk elimination
4. Holographic superposition mode (Mode B) — collective vector construction and individual signal recovery
5. Relay node architecture — ciphertext-only relay with SHA-256 audit fingerprint and no key possession
6. Full 4-node topology (Control Plane, Source, Relay, Target) with ephemeral SessionKeyBundle and TTL

---

## Company & Inventor

**Inventor:** Amin Parva  
**Company:** Insight IT Solutions LLC  
**Website:** https://insightits.com  
**Email:** parvaamin@gmail.com  
**Location:** Mission Viejo, California, USA  

Amin Parva is an AI Solution Architect and Director of Engineering with 20+ years across software, AI, and machine-learning systems. He also invented PrismLang (deterministic vector protocol for LangGraph, ~60% token reduction) and PrismRAG (mapping-first enterprise Graph RAG). CHORUS Fabric is the third foundational AI infrastructure invention from Insight IT Solutions.

---

## Competitive Differentiation

| Feature | CHORUS Fabric | gRPC (standard) | HTTP/REST | LLM API |
|---------|--------------|-----------------|-----------|---------|
| Tensor-native (no serialization) | ✅ | ❌ | ❌ | ❌ |
| Built-in encryption (cipher in the math) | ✅ | ❌ (needs TLS) | ❌ (needs TLS) | ❌ |
| Per-message watermark / auth | ✅ | ❌ | ❌ | ❌ |
| Orthogonal channel sharing | ✅ | ❌ | ❌ | ❌ |
| Relay with zero key possession | ✅ | ❌ | ❌ | ❌ |
| Bandwidth vs HTTP/REST | 4.45× less | ~2× less | baseline | 7.1× more |
| Cipher overhead | 0 ms | N/A | N/A | N/A |

---

## Tone & Positioning for the Landing Page

- **Audience:** AI engineers, ML platform teams, AI startup CTOs, enterprise AI architects
- **Tone:** Technical confidence. Real numbers, real code, real patent. Not hype.
- **Key proof points to emphasize:** The 0 ms cipher overhead (extraordinary claim with proof), 100% watermark rate across 7,766 transmissions, and the 4.45× bandwidth number
- **Call to action options:** `pip install chorus-fabric` · View on PyPI · Licensing inquiry (parvaamin@gmail.com) · Read the patent abstract
- **Color palette suggestion:** Deep navy (#1A365D) primary, warm gold (#9A7B3A) accent — matches Insight IT brand
- **Do NOT use:** vague superlatives, "revolutionary", "game-changing" — the numbers speak for themselves
- **Do use:** the exact benchmark numbers, the exact patent application number, real code snippets
