CHORUS FABRIC — PATENT FIGURES GUIDE
=====================================
These are the 8 figures referenced in the specification.
Each figure should be submitted as a separate PDF page (black & white).
Informal hand-drawn or computer-generated drawings are acceptable for provisional.

--------------------------------------------------------------------------------
FIG. 1 — SYSTEM ARCHITECTURE (Four-Node Topology)
--------------------------------------------------------------------------------
Draw as a block diagram with four boxes connected by arrows:

  [SOURCE POD / BENCHMARK CLIENT]          [CONTROL PLANE]
   US East - Virginia                       Germany West Central
   IP: dynamic                              IP: 4.182.111.109
   Port: client                             Port: 50051
        |                                        |
        |<--- (1) RegisterAndRequestKey -------->|
        |<--- (2) SessionKeyBundle (K, seed) ----|
        |
        |---(3) TensorPayload stream (V_enc)-->[ RELAY NODE ]
                                               US East - Virginia
                                               IP: 20.237.21.245
                                               Port: 50052
                                               [logs SHA-256(V_enc)]
                                                     |
                                        (4) V_amp = factor * V_enc
                                                     |
                                                     v
                                               [TARGET POD]
                                               Germany West Central
                                               IP: 98.67.201.195
                                               Port: 50053
                                               [decrypt V_dec = V_amp @ K_inv]
                                               [verify watermark]
                                               [dispatch Mode A / Mode B]
                                                     |
                                        (5) SignalAck <-----------
                                                     |
                                        (6) RelayAck <-- Relay <--

Label: FIG. 1 — CHORUS Fabric Four-Node Architecture

--------------------------------------------------------------------------------
FIG. 2 — SIGNAL ENCRYPTION AND WATERMARK INJECTION (Transmitter Side)
--------------------------------------------------------------------------------
Draw as a vertical flowchart:

  [Input: V_raw (d-dimensional float32 vector)]
            |
            v
  [Compute watermark vector W(n)]
  SHA-256(session_id || seq_num) -> seeded RNG -> unit vector W
            |
            v
  [Inject watermark]
  V_wm = normalize(V_raw + alpha * W(n))
            |
            v
  [Encrypt]
  V_enc = (V_wm.float64 @ K.float64).float32
            |
            v
  [Pack into TensorPayload protobuf]
  { data: V_enc bytes, seq_num: n, session_id, mode }
            |
            v
  [Transmit over gRPC bidirectional stream]

Label: FIG. 2 — Watermark Injection and Encryption at Transmitting Node

--------------------------------------------------------------------------------
FIG. 3 — SIGNAL DECRYPTION AND WATERMARK VERIFICATION (Receiver Side)
--------------------------------------------------------------------------------
Draw as a vertical flowchart with a decision diamond:

  [Receive TensorPayload from gRPC stream]
            |
            v
  [Unpack: V_enc, seq_num n, session_id, mode]
            |
            v
  [Decrypt]
  V_dec = (V_enc.float64 @ K_inv.float64).float32
            |
            v
  [Recompute expected watermark W(n)]
  SHA-256(session_id || n) -> unit vector W(n)
            |
            v
  [Compute cosine similarity]
  sim = dot(V_dec, W(n)) / (||V_dec|| * ||W(n)||)
            |
            v
  < sim >= threshold (0.95)? >
    YES ------> [verified=True] ------> [Mode Dispatch]
    NO  ------> [verified=False, reject payload, log tamper event]

  [Mode Dispatch]:
    mode=="direct"       -> use V_dec directly
    mode=="isolation"    -> FIG. 4 handler
    mode=="superposition"-> FIG. 5 handler

Label: FIG. 3 — Decryption, Watermark Verification, and Mode Dispatch

--------------------------------------------------------------------------------
FIG. 4 — MODE A: ORTHOGONAL ISOLATION
--------------------------------------------------------------------------------
Draw as two parallel columns merging then splitting:

  TRANSMITTER SIDE:               RECEIVER SIDE:

  [Agent A signal V_A]            [Decrypted V_dec]
        |                               |          |
        v                               v          v
  [Project: P_A = W_A @ V_A]   [W_A @ V_dec]  [W_B @ V_dec]
        |                               |          |
        v                               v          v
  [Agent B signal V_B]          [V_A_recovered] [V_B_recovered]
        |
        v
  [Project: P_B = W_B @ V_B]
        |
        v
  [Mix: V_tunnel = normalize(P_A + P_B)]
        |
        v
  [Watermark + Encrypt + Transmit]
        |
        -----------> [gRPC stream] ------->

  Key property: W_A @ W_B^T ≈ 0  (orthogonality condition)
  Measured crosstalk: 0.000006%

Label: FIG. 4 — Mode A Orthogonal Isolation: Dual-Signal Multiplexing

--------------------------------------------------------------------------------
FIG. 5 — MODE B: HOLOGRAPHIC SUPERPOSITION
--------------------------------------------------------------------------------
Draw as two inputs merging, transmitting, then approximately recovering:

  [Agent A: V_A]    [Agent B: V_B]
        |                 |
        +--------+--------+
                 |
                 v
  [V_collective = normalize(V_A + V_B)]
                 |
                 v
  [Watermark + Encrypt + Transmit]
                 |
                 v
  [Decrypted: V_dec ≈ V_collective]
                 |
         --------|--------
         |               |
         v               v
  [sim(V_dec, V_A)  [sim(V_dec, V_B)
     ≈ 0.70]           ≈ 0.70]

  Use case: consensus signaling, ensemble agreement, sensor fusion

Label: FIG. 5 — Mode B Holographic Superposition

--------------------------------------------------------------------------------
FIG. 6 — RELAY NODE ARCHITECTURE
--------------------------------------------------------------------------------
Draw as a horizontal flow:

  [Source Pod]                [Relay Node]                [Target Pod]
       |                           |                           |
       |---V_enc (gRPC stream)---->|                           |
       |                           |                           |
       |               [Amplify: V_amp = factor * V_enc]       |
       |               [Log: SHA-256(V_enc) -> audit store]    |
       |               [No decryption. K_inv NOT present.]     |
       |                           |                           |
       |                           |---V_amp (gRPC stream)---->|
       |                           |                           |
       |                           |<--SignalAck (seq, status)-|
       |<--RelayAck (forwarded)----|                           |

  Note: decrypt(V_amp) = factor * V_raw_watermarked
        (linear property of matrix multiplication)

Label: FIG. 6 — Relay Node: Ciphertext Amplification Without Decryption

--------------------------------------------------------------------------------
FIG. 7 — CONTROL PLANE KEY ISSUANCE SEQUENCE
--------------------------------------------------------------------------------
Draw as a sequence diagram (three vertical lifelines):

  Source Pod          Control Plane          Target Pod
      |                    |                     |
      |--RegisterAndRequestKey(pod_id, mode)----->|
      |                    |                     |
      |                    |[generate K, K_inv,  |
      |                    | watermark_seed,      |
      |                    | session_id, TTL]     |
      |                    |                     |
      |<--SessionKeyBundle(K, seed, session_id)---|
      |                    |                     |
      |                    |<--Register(session_id)-----|
      |                    |                     |
      |                    |--SessionKeyBundle(K_inv, seed)-->|
      |                    |                     |
  [TTL expires]            |                     |
      |--ReRegister------->|                     |
      |<--NewSession-------|                     |

Label: FIG. 7 — Control Plane Session Key Issuance and TTL Lifecycle

--------------------------------------------------------------------------------
FIG. 8 — BENCHMARK RESULTS CHART
--------------------------------------------------------------------------------
Draw as a horizontal bar chart (3 bars):

  p50 Round-Trip Latency — Transatlantic (US East to Germany West Central)
  Tensor dimension d=128, 7,766 total payloads, 100% watermark verified

  Direct (no mux)    |████████                    | 179 ms  (p95: 300 ms)
  Isolation (Mode A) |█████████████████           | 311 ms  (p95: 381 ms)
  Superposition (B)  |█████████████████████████████████████████| 1,274 ms (p95: 1,422 ms)

  Bandwidth comparison (per 128-dim payload):
  CHORUS gRPC   ████  548 B   (1x baseline)
  HTTP/REST     ████████████████████  2,440 B  (4.45x)
  LLM API       ████████████████████████████  3,900 B  (7.1x)

  Note: Direct mode p50 = 179 ms matches theoretical transatlantic physics
        minimum (~165-180 ms), confirming cipher adds zero measurable overhead.

Label: FIG. 8 — Benchmark Results: Latency and Bandwidth vs. Existing Protocols

================================================================================
DRAWING NOTES FOR FILING:
- All figures must be black and white (no grayscale shading for formal drawings)
- Minimum line weight: 0.3 mm
- Reference numbers must match those in the specification text
- Label each figure as "FIG. X" at the bottom
- Informal drawings are acceptable for provisional applications
- Acceptable tools: PowerPoint, draw.io, Visio, or hand-drawn scanned at 300 DPI
================================================================================
