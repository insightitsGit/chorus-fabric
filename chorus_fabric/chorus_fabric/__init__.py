"""
chorus-fabric
=============
High-dimensional tensor communication fabric for AI-to-AI signaling.
Patent Pending — US Provisional Application 64/096,156

Quick start:
    from chorus_fabric import ChorusClient
    client = ChorusClient(pod_id="agent-1", control_plane="localhost:50051")
    client.handshake()
    acks = client.send_direct(my_tensor)

v0.2.0 — zero-overhead encryption + forward secrecy:
    client = ChorusClient(..., rekey_every=64)   # auto-rotate the key every 64 msgs
    client.rotate_key()                           # or rotate on demand
    from chorus_fabric import cipher_throughput   # benchmark cipher CPU vs GPU
"""

from chorus_fabric.client import ChorusClient
from chorus_fabric.crypto_engine import (
    generate_key_pair,
    encrypt,
    decrypt,
    inject_watermark,
    verify_watermark,
    generate_orthogonal_projections,
    superpose_signals,
    SessionKeyManager,
    resolve_device,
    cuda_available,
    cipher_throughput,
    DEFAULT_DIM,
)

__version__ = "0.2.0"
__author__  = "Amin Parva"
__email__   = "parvaamin@gmail.com"
__patent__  = "US Provisional Application 64/096,156"

__all__ = [
    "ChorusClient",
    "generate_key_pair",
    "encrypt",
    "decrypt",
    "inject_watermark",
    "verify_watermark",
    "generate_orthogonal_projections",
    "superpose_signals",
    "SessionKeyManager",
    "resolve_device",
    "cuda_available",
    "cipher_throughput",
    "DEFAULT_DIM",
]
