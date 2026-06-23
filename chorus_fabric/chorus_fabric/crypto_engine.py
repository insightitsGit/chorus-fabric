"""
CHORUS Protocol – crypto_engine.py
====================================
All linear-algebra security primitives for the CHORUS signal layer.

Provides:
  • Ephemeral key matrix generation + inversion
  • Tensor multiplication cipher   V_enc = V_raw @ K
  • Orthogonal projection pairs    W_A @ W_B ≈ 0
  • Rolling neural watermark       inject / verify
  • Wire serialisation             tensor ↔ bytes

Float64 is used internally for numerical stability; float32 on the wire.
"""

import hashlib
import logging
import struct
from typing import Tuple

import torch

logger = logging.getLogger("chorus.crypto")

DEFAULT_DIM = 128          # MVP embedding dimension (scale to 1536 via CHORUS_DIM env)
WATERMARK_RATIO = 0.10     # 10 % of dimensions carry the rolling watermark


# ─────────────────────────────────────────────────────────────────────────────
# Key Matrix Management
# ─────────────────────────────────────────────────────────────────────────────

def generate_key_pair(dim: int = DEFAULT_DIM) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ephemeral invertible key matrix K and its exact inverse K_inv.

    Construction:
      1. Random matrix → QR decomposition yields orthogonal Q.
      2. Sign-correct Q so diagonal of R is positive (canonical).
      3. Scale by random scalar in [0.5, 2.5] to break det=±1 pattern.

    Returns (K, K_inv) both float64, shape [dim, dim].
    """
    A = torch.randn(dim, dim, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    signs = torch.diag(torch.sign(torch.diag(R)))
    K = Q @ signs
    scale = torch.rand(1, dtype=torch.float64).item() * 2.0 + 0.5
    K = K * scale
    K_inv = torch.linalg.inv(K)

    err = torch.max(torch.abs(K @ K_inv - torch.eye(dim, dtype=torch.float64))).item()
    if err > 1e-8:
        raise RuntimeError(f"Key inversion error {err:.2e} exceeds tolerance")
    logger.debug("Key pair generated dim=%d inversion_error=%.2e", dim, err)
    return K, K_inv


def encrypt(v_raw: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """V_enc = V_raw @ K  →  float32 output."""
    v = v_raw.to(torch.float64)
    return (v @ K).to(torch.float32)


def decrypt(v_enc: torch.Tensor, K_inv: torch.Tensor) -> torch.Tensor:
    """V_raw = V_enc @ K_inv  →  float32 output."""
    v = v_enc.to(torch.float64)
    return (v @ K_inv).to(torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal Projection Pairs  (Mode A – Space-Division Isolation)
# ─────────────────────────────────────────────────────────────────────────────

def generate_orthogonal_projections(dim: int = DEFAULT_DIM) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (W_A, W_B) such that W_A @ W_B ≈ 0 (mathematically orthogonal).

    Strategy:
      Build a full orthonormal basis via QR.
      W_A projects onto the first  dim//2 basis vectors.
      W_B projects onto the second dim//2 basis vectors.
      Orthogonality is exact by construction (P = U U^T).
    """
    half = dim // 2
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float64))
    W_A = Q[:, :half]  @ Q[:, :half].T
    W_B = Q[:, half:]  @ Q[:, half:].T
    ct = torch.max(torch.abs(W_A @ W_B)).item()
    logger.debug("Projection crosstalk=%.2e", ct)
    return W_A, W_B


def project_signal(v: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """v_proj = W @ v  (projects v into subspace W)."""
    return (W.to(torch.float64) @ v.to(torch.float64)).to(torch.float32)


def mix_signals(v_A_proj: torch.Tensor, v_B_proj: torch.Tensor) -> torch.Tensor:
    """Mode A wire signal:  V_tunnel = W_A·V_A + W_B·V_B."""
    return v_A_proj.float() + v_B_proj.float()


def isolate_signal(v_tunnel: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Recover one signal from the tunnel:  W @ V_tunnel = W^2 @ V_orig + 0
    (since W is an idempotent projector and the two subspaces are orthogonal).
    """
    return project_signal(v_tunnel, W)


def superpose_signals(v_A: torch.Tensor, v_B: torch.Tensor) -> torch.Tensor:
    """Mode B holographic blend:  V_collective = V_A + V_B."""
    return v_A.float() + v_B.float()


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Neural Watermark
# ─────────────────────────────────────────────────────────────────────────────

def generate_watermark_seed(dim: int = DEFAULT_DIM) -> bytes:
    """32-byte random seed for the watermark PRNG."""
    return torch.randint(0, 2**31, (8,), dtype=torch.int64).numpy().tobytes()


def compute_watermark(seed: bytes, seq_num: int, dim: int = DEFAULT_DIM) -> torch.Tensor:
    """
    Deterministic unit-vector watermark for (seed, seq_num).
    Uses SHA-256(seed || seq_num) to seed a PyTorch RNG so both source
    and target can independently compute the same value.
    """
    wdim = max(1, int(dim * WATERMARK_RATIO))
    digest = hashlib.sha256(seed + struct.pack(">q", seq_num)).digest()
    seed_int = int.from_bytes(digest[:8], "big") % (2 ** 31)
    gen = torch.Generator()
    gen.manual_seed(seed_int)
    wm = torch.randn(wdim, generator=gen)
    return wm / (wm.norm() + 1e-8)


def inject_watermark(v: torch.Tensor, seed: bytes, seq_num: int) -> torch.Tensor:
    """Overwrite the watermark slice of v with the expected watermark vector."""
    wdim = max(1, int(v.shape[-1] * WATERMARK_RATIO))
    wm = compute_watermark(seed, seq_num, v.shape[-1])
    out = v.clone()
    out[..., :wdim] = wm.to(v.dtype)
    return out


def verify_watermark(v: torch.Tensor, seed: bytes, seq_num: int,
                     threshold: float = 0.95) -> bool:
    """
    True if cosine similarity between the received and expected watermark
    exceeds threshold.  False → tampered / replayed / wrong key.
    """
    wdim = max(1, int(v.shape[-1] * WATERMARK_RATIO))
    expected = compute_watermark(seed, seq_num, v.shape[-1]).float()
    received = v[..., :wdim].float()
    sim = torch.nn.functional.cosine_similarity(
        received.unsqueeze(0), expected.unsqueeze(0)
    ).item()
    logger.debug("Watermark cosine_sim=%.4f threshold=%.2f", sim, threshold)
    return sim >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Wire Serialisation  (tensor ↔ bytes)
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Pack float32 tensor to little-endian raw bytes."""
    return t.to(torch.float32).numpy().tobytes()


def bytes_to_tensor(b: bytes, dim: int) -> torch.Tensor:
    """Unpack raw bytes to float32 tensor of shape [dim]."""
    import numpy as np
    arr = np.frombuffer(b, dtype=np.float32)
    return torch.from_numpy(arr.copy())


def matrix_to_bytes(m: torch.Tensor) -> bytes:
    """Pack 2-D matrix to raw bytes (float32)."""
    return m.to(torch.float32).numpy().tobytes()


def bytes_to_matrix(b: bytes, dim: int) -> torch.Tensor:
    """Unpack raw bytes to float32 matrix [dim, dim]."""
    import numpy as np
    arr = np.frombuffer(b, dtype=np.float32)
    return torch.from_numpy(arr.copy()).reshape(dim, dim)
