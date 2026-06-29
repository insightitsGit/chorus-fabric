"""
CHORUS Protocol – server.py
=============================
Target Pod  –  final signal destination.

Per incoming TensorPayload:
  1. Resolve session keys for (session_id, key_epoch) from the Control Plane
     (cached locally after first fetch; a new key_epoch triggers a fresh fetch)
  2. Decrypt:          V_raw = V_enc @ K_inv   (using the payload's key epoch)
  3. Verify watermark  (drop + ack tampered=True if check fails)
  4. Mode dispatch:
       "isolation"    -> apply W_A / W_B projectors, log crosstalk %
       "superposition"-> process holographic consensus vector, log cosine sims
       "direct"       -> plain decrypted vector, no multiplexing
  5. Return SignalAck (verified, signal_norm, status, key_epoch)

Dynamic key rotation (v0.2.0): the sender tags every payload with the key_epoch
it was encrypted under. The target fetches that exact epoch from the Control
Plane via GetSessionKey, so it follows the sender's rotations in lock-step.
"""

import logging
import os
import time
from concurrent import futures

import grpc
import torch

import crypto_engine as ce
import fabric_pb2
import fabric_pb2_grpc

logger = logging.getLogger("chorus.server")

TARGET_PORT         = int(os.getenv("TARGET_PORT", "50053"))
CONTROL_PLANE_HOST  = os.getenv("CONTROL_PLANE_HOST", "control-plane")
CONTROL_PLANE_PORT  = int(os.getenv("CONTROL_PLANE_PORT", "50051"))
DIM                 = int(os.getenv("CHORUS_DIM", str(ce.DEFAULT_DIM)))


# ── Local session cache ────────────────────────────────────────────────────────

class _SessionCache:
    """Caches per-session metadata plus a map of key_epoch -> K_inv, so multiple
    key generations of the same session stay decryptable during rotation."""

    def __init__(self):
        self._c: dict = {}

    def get(self, sid: str):
        e = self._c.get(sid)
        return e if (e and time.time() < e["expires_at"]) else None

    def ensure(self, sid: str, bundle: fabric_pb2.SessionKeyBundle,
               W_A=None, W_B=None) -> dict:
        """Create-or-update the session entry and store this epoch's K_inv."""
        entry = self._c.get(sid)
        if entry is None:
            entry = {
                "keys":           {},   # epoch -> K_inv tensor
                "watermark_seed": bundle.watermark_seed,
                "expires_at":     bundle.expires_at,
                "dim":            bundle.dim,
                "W_A":            W_A,
                "W_B":            W_B,
            }
            self._c[sid] = entry
        entry["keys"][bundle.key_epoch] = ce.bytes_to_matrix(
            bundle.key_matrix_K_inv, bundle.dim)
        entry["expires_at"] = bundle.expires_at
        if W_A is not None:
            entry["W_A"], entry["W_B"] = W_A, W_B
        return entry


# ── Target Pod servicer ────────────────────────────────────────────────────────

class TargetPodServicer(fabric_pb2_grpc.TargetPodServicer):

    def __init__(self):
        self._cache = _SessionCache()
        cp_addr = f"{CONTROL_PLANE_HOST}:{CONTROL_PLANE_PORT}"
        ch = grpc.insecure_channel(cp_addr)
        self._cp = fabric_pb2_grpc.ControlPlaneStub(ch)
        logger.info("Target Pod -> Control Plane at %s", cp_addr)

    def StreamSignal(self, request_iterator, context):
        """Bidirectional stream: yield one SignalAck per TensorPayload."""
        for payload in request_iterator:
            yield self._process(payload)

    def _process(self, p: fabric_pb2.TensorPayload) -> fabric_pb2.SignalAck:
        sid     = p.session_id
        seq     = p.seq_num
        mode    = p.mode
        dim     = p.dim or DIM
        epoch   = p.key_epoch

        # 1 – resolve session + the specific key epoch this payload used
        session = self._cache.get(sid)
        if session is None or epoch not in session["keys"]:
            session = self._fetch_session(sid, p.pod_id, dim, mode, epoch)
        if session is None or epoch not in session["keys"]:
            return fabric_pb2.SignalAck(
                session_id=sid, seq_num=seq, verified=False, key_epoch=epoch,
                status="key_expired", message=f"No key for epoch {epoch}")

        # 2 – decrypt with the payload's key epoch
        try:
            v_enc = ce.bytes_to_tensor(p.data, dim)
            v_raw = ce.decrypt(v_enc, session["keys"][epoch])
        except Exception as exc:
            logger.error("Decrypt failed seq=%d epoch=%d: %s", seq, epoch, exc)
            return fabric_pb2.SignalAck(
                session_id=sid, seq_num=seq, verified=False, key_epoch=epoch,
                status="error", message=str(exc))

        # 3 – verify watermark
        if not ce.verify_watermark(v_raw, session["watermark_seed"], seq):
            logger.warning("TAMPER DETECTED session=%s seq=%d epoch=%d", sid, seq, epoch)
            return fabric_pb2.SignalAck(
                session_id=sid, seq_num=seq, verified=False, key_epoch=epoch,
                status="tampered", message="Watermark verification failed")

        # 4 – mode dispatch
        norm = v_raw.norm().item()
        if mode == "isolation":
            self._dispatch_isolation(v_raw, session, seq, sid)
        elif mode == "superposition":
            self._dispatch_superposition(v_raw, seq, sid)
        else:
            logger.info("[Direct] session=%s seq=%d epoch=%d norm=%.4f",
                        sid, seq, epoch, norm)

        return fabric_pb2.SignalAck(
            session_id=sid, seq_num=seq, verified=True, key_epoch=epoch,
            signal_norm=norm, status="ok", message=f"mode={mode}")

    # ── Mode handlers ─────────────────────────────────────────────────────────

    def _dispatch_isolation(self, v_tunnel, session, seq, sid):
        W_A, W_B = session.get("W_A"), session.get("W_B")
        if W_A is None:
            logger.warning("Isolation mode but no projections for session %s", sid)
            return
        v_A = ce.isolate_signal(v_tunnel, W_A)
        v_B = ce.isolate_signal(v_tunnel, W_B)
        # Crosstalk: fraction of A's energy landing in B's subspace
        ct_AB = (W_B.float() @ v_A).norm().item() / (v_A.norm().item() + 1e-8) * 100
        ct_BA = (W_A.float() @ v_B).norm().item() / (v_B.norm().item() + 1e-8) * 100
        logger.info("[Isolation] seq=%d norm_A=%.4f norm_B=%.4f "
                    "crosstalk_A->B=%.6f%% crosstalk_B->A=%.6f%%",
                    seq, v_A.norm().item(), v_B.norm().item(), ct_AB, ct_BA)

    def _dispatch_superposition(self, v_collective, seq, sid):
        """
        Process the holographic consensus vector.
        In production: inject v_collective directly into the target LLM's
        attention layer.  Here we log the norm as a proxy for signal strength.
        """
        norm = v_collective.norm().item()
        logger.info("[Superposition] seq=%d consensus_norm=%.4f session=%s",
                    seq, norm, sid)

    # ── Session bootstrap from Control Plane ─────────────────────────────────

    def _fetch_session(self, sid, pod_id, dim, mode, epoch):
        """Fetch the sender's key for (sid, epoch) so the target decrypts with
        the exact generation the payload was encrypted under."""
        try:
            bundle = self._cp.GetSessionKey(
                fabric_pb2.KeyRequest(pod_id=f"target-{pod_id}",
                                      session_id=sid, key_epoch=epoch)
            )
            if not bundle.session_id:
                logger.error("Control Plane has no session %s", sid)
                return None
            W_A, W_B = None, None
            if mode == "isolation" and self._cache.get(sid) is None:
                proj = self._cp.RequestOrthogonalProjections(
                    fabric_pb2.KeyRequest(pod_id=f"target-{pod_id}", session_id=sid)
                )
                W_A = ce.bytes_to_matrix(proj.W_A, proj.dim).double()
                W_B = ce.bytes_to_matrix(proj.W_B, proj.dim).double()
            return self._cache.ensure(sid, bundle, W_A, W_B)
        except grpc.RpcError as e:
            logger.error("Control Plane unreachable: %s", e)
            return None


def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fabric_pb2_grpc.add_TargetPodServicer_to_server(TargetPodServicer(), server)
    addr = f"[::]:{TARGET_PORT}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("Target Pod listening on %s", addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
