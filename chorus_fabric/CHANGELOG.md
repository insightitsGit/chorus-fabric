# Changelog

All notable changes to **The Chorus Fabric** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-29

> Headline: **zero-overhead encryption + forward secrecy.**

### Added — Dynamic Key Rotation (forward secrecy)
- `crypto_engine.SessionKeyManager`: a rolling keyring of QR key *epochs* with a
  configurable grace window. Rotating mints a fresh key, advances the epoch, and
  retires keys past the grace window so they can no longer decrypt any traffic.
- Wire protocol (`fabric.proto` → v1.1):
  - `TensorPayload.key_epoch` and `SignalAck.key_epoch` — every payload is tagged
    with the key generation it was encrypted under.
  - `SessionKeyBundle.key_epoch`, `KeyRequest.key_epoch`.
  - New RPCs `ControlPlane.RotateKey` and `ControlPlane.GetSessionKey`.
- `ControlPlane` issues and tracks per-session epochs; `RotateKey` mints a new
  generation, `GetSessionKey` returns the exact epoch a receiver asks for.
- `ChorusClient.rotate_key()` (rotate on demand) and `rekey_every=N`
  (auto-rotate every N messages, also via `CHORUS_REKEY_EVERY`). The swap is
  atomic — no payload is ever encrypted with a half-updated key/epoch pair.
- Target pod decrypts each payload with its tagged epoch, fetching new epochs
  from the Control Plane on demand; the relay carries `key_epoch` through
  amplification.

### Added — CUDA cipher path
- `resolve_device()` / `cuda_available()` and a `CHORUS_DEVICE` env
  (`auto` | `cpu` | `cuda`); a `cuda` request falls back to CPU with a warning
  when no GPU is present, so the same code runs everywhere.
- `encrypt` / `decrypt` / `generate_key_pair` accept a `device=` argument and run
  the matmul on that device (defaults unchanged → fully backward compatible).
- `cipher_throughput()` helper and a new `bench_cipher.py` script that benchmarks
  cipher throughput (vectors/sec, GiB/s, µs/vector) on CPU vs GPU across
  embedding dimensions (128 → 4096).

### Fixed
- **Target now decrypts with the sender's actual key.** Previously the target
  pod registered its *own* random session key, which could not decrypt the
  sender's ciphertext; it now fetches the sender's key via `GetSessionKey`.
  Verified end-to-end over local gRPC across three key epochs.
- **Package import.** The generated `chorus_fabric/fabric_pb2_grpc.py` now uses a
  package-relative import (`from chorus_fabric import fabric_pb2`) instead of a
  flat `import fabric_pb2`, so the installed wheel imports without a stray
  top-level `fabric_pb2` on `sys.path`.
- Dependency floors corrected to match the checked-in generated stubs
  (`protobuf>=6.33.5,<7`, `grpcio>=1.81.0`); the old `protobuf>=5.26` floor could
  not load the shipped gencode.

### Tests
- Test suite grows from 20 → 29 checks: `TestKeyRotation` (5) covers epoch
  advance, per-epoch round-trip, in-flight grace, retired-key death, and grace
  window size; `TestDevicePath` (4) covers device resolution/fallback, device
  parity, the throughput smoke test, and GPU/CPU parity (skipped without CUDA).

## [0.1.0]
- Initial release: tensor-multiplication cipher, rolling SHA-256 watermark,
  Direct / Isolation (Mode A) / Superposition (Mode B) transmission modes,
  4-node topology (Control Plane, Source, Relay, Target), cross-region benchmark.
