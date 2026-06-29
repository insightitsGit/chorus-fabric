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
  • Dynamic key rotation           SessionKeyManager (forward secrecy)   [v0.2.0]
  • CUDA acceleration path         resolve_device / cipher_throughput    [v0.2.0]

Float64 is used internally for numerical stability; float32 on the wire.

v0.2.0 notes
------------
The cipher *is* a matrix multiply, so it runs unchanged on a GPU — pass
``device="cuda"`` (or set ``CHORUS_DEVICE=cuda``) to ``encrypt`` / ``decrypt``
and the matmul executes on the GPU. ``cipher_throughput`` benchmarks the cipher
on CPU vs GPU to quantify the "zero-overhead" claim at high dimensions.

``SessionKeyManager`` adds per-session / per-N-message key rotation: each key
generation is an *epoch*; a payload is tagged with the epoch it was encrypted
under, so a rotated-away key can be retired and can no longer decrypt past or
future traffic (forward secrecy).
"""

import hashlib
import logging
import os
import struct
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger("chorus.crypto")

DEFAULT_DIM = 128          # MVP embedding dimension (scale to 1536 via CHORUS_DIM env)
WATERMARK_RATIO = 0.10     # 10 % of dimensions carry the rolling watermark

# How many past epochs a SessionKeyManager keeps decryptable after a rotation,
# so messages already in flight under the old key still verify. Older epochs are
# retired (forgotten) — they can no longer be decrypted. This is the knob that
# turns rotation into forward secrecy.
DEFAULT_KEY_GRACE = int(os.getenv("CHORUS_KEY_GRACE", "2"))


# ─────────────────────────────────────────────────────────────────────────────
# Device Management  (CUDA acceleration path)
# ─────────────────────────────────────────────────────────────────────────────

def cuda_available() -> bool:
    """True if a CUDA device is visible to torch."""
    try:
        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_device(prefer: Optional[str] = None) -> torch.device:
    """
    Resolve the compute device for cipher operations.

    Priority:
      1. explicit ``prefer`` argument ("cuda" | "cpu" | "auto")
      2. ``CHORUS_DEVICE`` environment variable
      3. "auto"  →  cuda if available else cpu

    A request for "cuda" on a machine without a GPU falls back to CPU with a
    warning rather than raising, so the same code runs everywhere.
    """
    choice = (prefer or os.getenv("CHORUS_DEVICE", "auto")).lower()
    if choice in ("cuda", "gpu"):
        if cuda_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if cuda_available() else "cpu")


def to_device(t: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor to a device (accepts str or torch.device)."""
    return t.to(torch.device(device) if isinstance(device, str) else device)


# ─────────────────────────────────────────────────────────────────────────────
# Key Matrix Management
# ─────────────────────────────────────────────────────────────────────────────

def generate_key_pair(
    dim: int = DEFAULT_DIM,
    device: Optional[object] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ephemeral invertible key matrix K and its exact inverse K_inv.

    Construction:
      1. Random matrix → QR decomposition yields orthogonal Q.
      2. Sign-correct Q so diagonal of R is positive (canonical).
      3. Scale by random scalar in [0.5, 2.5] to break det=±1 pattern.

    Returns (K, K_inv) both float64, shape [dim, dim], on ``device``
    (default CPU — backward compatible). QR/inversion run on CPU in float64
    for numerical stability, then the pair is moved to the requested device.
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

    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        K, K_inv = K.to(dev), K_inv.to(dev)
    return K, K_inv


def encrypt(v_raw: torch.Tensor, K: torch.Tensor,
            device: Optional[object] = None) -> torch.Tensor:
    """
    V_enc = V_raw @ K  →  float32 output.

    The matmul runs on ``device`` if given, otherwise on K's own device
    (CPU by default — backward compatible). Both operands are aligned to the
    same device and promoted to float64 for the multiply; the result is
    returned as float32 on that device.
    """
    dev = (torch.device(device) if device is not None
           else (K.device if isinstance(K, torch.Tensor) else torch.device("cpu")))
    v = v_raw.to(dev, torch.float64)
    Kd = K.to(dev, torch.float64)
    return (v @ Kd).to(torch.float32)


def decrypt(v_enc: torch.Tensor, K_inv: torch.Tensor,
            device: Optional[object] = None) -> torch.Tensor:
    """V_raw = V_enc @ K_inv  →  float32 output. See ``encrypt`` for device rules."""
    dev = (torch.device(device) if device is not None
           else (K_inv.device if isinstance(K_inv, torch.Tensor) else torch.device("cpu")))
    v = v_enc.to(dev, torch.float64)
    Ki = K_inv.to(dev, torch.float64)
    return (v @ Ki).to(torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Key Rotation  (Forward Secrecy)  [v0.2.0]
# ─────────────────────────────────────────────────────────────────────────────

class SessionKeyManager:
    """
    Holds a rolling set of QR key generations ("epochs") for one session.

    The current epoch encrypts new traffic. On ``rotate()`` a brand-new key
    pair is minted and the epoch counter increments; the previous ``grace``
    epochs are retained so in-flight messages still decrypt, and anything older
    is *retired* — its key material is dropped and can never decrypt again.
    That retirement is what gives forward secrecy: a key captured after it has
    rotated away yields nothing.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    grace : int
        Number of *past* epochs kept decryptable after a rotation (>= 0).
    device : optional
        Device to place key matrices on (CUDA path). Default CPU.
    """

    def __init__(self, dim: int = DEFAULT_DIM, grace: int = DEFAULT_KEY_GRACE,
                 device: Optional[object] = None):
        self.dim = dim
        self.grace = max(0, int(grace))
        self.device = (torch.device(device) if isinstance(device, str)
                       else device) if device is not None else None
        self._keys: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.current_epoch = -1
        self.rotate()  # mint epoch 0

    # -- lifecycle -----------------------------------------------------------

    def rotate(self) -> int:
        """Mint a fresh key pair as a new epoch; retire epochs past the grace
        window. Returns the new current epoch."""
        self.current_epoch += 1
        self._keys[self.current_epoch] = generate_key_pair(self.dim, device=self.device)
        self._retire_old()
        logger.info("Key rotated → epoch=%d  live_epochs=%s",
                    self.current_epoch, sorted(self._keys))
        return self.current_epoch

    def _retire_old(self) -> None:
        cutoff = self.current_epoch - self.grace
        for ep in [e for e in self._keys if e < cutoff]:
            del self._keys[ep]
            logger.debug("Epoch %d retired (no longer decryptable)", ep)

    # -- access --------------------------------------------------------------

    def live_epochs(self):
        return sorted(self._keys)

    def has_epoch(self, epoch: int) -> bool:
        return epoch in self._keys

    def current(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        K, K_inv = self._keys[self.current_epoch]
        return self.current_epoch, K, K_inv

    def get(self, epoch: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._keys.get(epoch)

    def install(self, epoch: int, K: torch.Tensor, K_inv: torch.Tensor) -> None:
        """Install an externally-supplied key pair for an epoch (used by the
        target side, which receives keys from the Control Plane)."""
        self._keys[epoch] = (K, K_inv)
        if epoch > self.current_epoch:
            self.current_epoch = epoch
        self._retire_old()

    # -- cipher --------------------------------------------------------------

    def encrypt(self, v_raw: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        ep = self.current_epoch if epoch is None else epoch
        K, _ = self._keys[ep]
        return encrypt(v_raw, K, device=self.device)

    def decrypt(self, v_enc: torch.Tensor, epoch: int) -> torch.Tensor:
        pair = self._keys.get(epoch)
        if pair is None:
            raise KeyError(
                f"epoch {epoch} retired or unknown — cannot decrypt "
                f"(live epochs: {self.live_epochs()})")
        _, K_inv = pair
        return decrypt(v_enc, K_inv, device=self.device)


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
    return t.to(torch.float32).cpu().numpy().tobytes()


def bytes_to_tensor(b: bytes, dim: int) -> torch.Tensor:
    """Unpack raw bytes to float32 tensor of shape [dim]."""
    import numpy as np
    arr = np.frombuffer(b, dtype=np.float32)
    return torch.from_numpy(arr.copy())


def matrix_to_bytes(m: torch.Tensor) -> bytes:
    """Pack 2-D matrix to raw bytes (float32)."""
    return m.to(torch.float32).cpu().numpy().tobytes()


def bytes_to_matrix(b: bytes, dim: int) -> torch.Tensor:
    """Unpack raw bytes to float32 matrix [dim, dim]."""
    import numpy as np
    arr = np.frombuffer(b, dtype=np.float32)
    return torch.from_numpy(arr.copy()).reshape(dim, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Cipher Throughput Benchmark  (CPU vs GPU)  [v0.2.0]
# ─────────────────────────────────────────────────────────────────────────────

def cipher_throughput(
    dim: int = 1536,
    batch: int = 1024,
    iters: int = 50,
    warmup: int = 5,
    device: Optional[object] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Measure raw cipher throughput: ``iters`` batched matmuls of a
    [batch, dim] signal block against a [dim, dim] key, on ``device``.

    Returns a dict with device, vectors/sec, GiB/sec, and per-vector latency.
    Used to quantify the "zero-overhead" claim and the CUDA fast-path: on a GPU
    the same matrix-multiply cipher scales to far higher dimensions at higher
    throughput than CPU, with no algorithmic change.
    """
    dev = resolve_device(device)
    K = torch.randn(dim, dim, dtype=dtype, device=dev)
    V = torch.randn(batch, dim, dtype=dtype, device=dev)

    use_cuda = dev.type == "cuda"

    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    # warmup (JIT/caches/allocator)
    for _ in range(max(1, warmup)):
        _ = V @ K
    _sync()

    import time as _time
    t0 = _time.perf_counter()
    for _ in range(iters):
        _ = V @ K
    _sync()
    elapsed = _time.perf_counter() - t0

    n_vectors = batch * iters
    bytes_moved = n_vectors * dim * torch.tensor([], dtype=dtype).element_size()
    vps = n_vectors / elapsed if elapsed > 0 else float("inf")
    return {
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "dim": dim,
        "batch": batch,
        "iters": iters,
        "elapsed_s": elapsed,
        "vectors_per_sec": vps,
        "gib_per_sec": (bytes_moved / elapsed / (1024 ** 3)) if elapsed > 0 else float("inf"),
        "us_per_vector": (elapsed / n_vectors * 1e6) if n_vectors else 0.0,
        "cuda_available": cuda_available(),
    }
