"""
CHORUS Protocol – control_plane.py
=====================================
Control Plane Service  (Option B: dedicated key-issuance node)

Every pod that wants to communicate must first register here and receive:
  • Ephemeral session key bundle  (K, K_inv, watermark_seed, TTL, key_epoch)
  • Orthogonal projection pair    (W_A, W_B) on request  [Mode A]
  • Relay routing instruction     on request             [Relay nodes]
  • Rotated key generations       on request             [forward secrecy]

Dynamic key rotation (v0.2.0)
-----------------------------
Each session owns a SessionKeyManager that tracks key *epochs*. RotateKey mints
a fresh QR key for an existing session and bumps the epoch; GetSessionKey lets
the receiving pod fetch the exact key generation a payload was tagged with, so
both ends follow the rotation in lock-step and retired keys decrypt nothing.

Phase 1 storage: in-memory dict (restarts clear all sessions).
Phase 2 upgrade: swap _sessions for Redis / etcd for HA multi-instance.
"""

import logging
import os
import time
import uuid
from concurrent import futures

import grpc

from chorus_fabric import crypto_engine as ce
from chorus_fabric import fabric_pb2
from chorus_fabric import fabric_pb2_grpc

logger = logging.getLogger("chorus.control_plane")

SESSION_TTL         = int(os.getenv("CHORUS_SESSION_TTL", "3600"))
DIM                 = int(os.getenv("CHORUS_DIM", str(ce.DEFAULT_DIM)))
CONTROL_PLANE_PORT  = int(os.getenv("CONTROL_PLANE_PORT", "50051"))
TARGET_HOST         = os.getenv("CHORUS_TARGET_HOST", "target-pod")
TARGET_PORT_ENV     = os.getenv("CHORUS_TARGET_PORT", "50053")
AMPLIFY_FACTOR      = float(os.getenv("CHORUS_AMPLIFY_FACTOR", "1.0"))
KEY_GRACE           = int(os.getenv("CHORUS_KEY_GRACE", str(ce.DEFAULT_KEY_GRACE)))
# Advisory rekey cadence broadcast to relays (0 = rotation driven by the client).
REKEY_EVERY         = int(os.getenv("CHORUS_REKEY_EVERY", "0"))


class ControlPlaneServicer(fabric_pb2_grpc.ControlPlaneServicer):

    def __init__(self):
        self._sessions: dict = {}   # session_id -> session dict
        logger.info("Control Plane initialised  dim=%d  ttl=%ds  key_grace=%d",
                    DIM, SESSION_TTL, KEY_GRACE)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bundle(self, sid: str, sess: dict, epoch: int) -> fabric_pb2.SessionKeyBundle:
        """Serialise the key pair at ``epoch`` of a session into a wire bundle."""
        K, K_inv = sess["manager"].get(epoch)
        return fabric_pb2.SessionKeyBundle(
            session_id=sid,
            key_matrix_K=ce.matrix_to_bytes(K),
            key_matrix_K_inv=ce.matrix_to_bytes(K_inv),
            watermark_seed=sess["watermark_seed"],
            issued_at=sess["issued_at"],
            expires_at=sess["expires_at"],
            dim=DIM,
            key_epoch=epoch,
        )

    # ── RegisterAndRequestKey ─────────────────────────────────────────────────

    def RegisterAndRequestKey(
        self,
        request: fabric_pb2.PodRegistration,
        context: grpc.ServicerContext,
    ) -> fabric_pb2.SessionKeyBundle:
        """
        Pod registers -> receives a fresh session at key epoch 0.
        A new session is always created (idempotent registration is Phase 2).
        """
        now = int(time.time())
        sid = str(uuid.uuid4())
        self._sessions[sid] = {
            "manager":        ce.SessionKeyManager(DIM, grace=KEY_GRACE),
            "watermark_seed": ce.generate_watermark_seed(DIM),
            "pod_id":         request.pod_id,
            "role":           request.pod_role,
            "issued_at":      now,
            "expires_at":     now + SESSION_TTL,
        }
        logger.info("Session created  pod=%s  role=%s  sid=%s  epoch=0  expires=%d",
                    request.pod_id, request.pod_role, sid, now + SESSION_TTL)
        return self._bundle(sid, self._sessions[sid], 0)

    # ── RotateKey ─────────────────────────────────────────────────────────────

    def RotateKey(
        self,
        request: fabric_pb2.KeyRequest,
        context: grpc.ServicerContext,
    ) -> fabric_pb2.SessionKeyBundle:
        """
        Mint a fresh key generation for an existing session (forward secrecy).
        Returns the bundle for the new epoch; old epochs beyond the grace window
        are retired by the SessionKeyManager and can no longer decrypt traffic.
        """
        sess = self.get_session(request.session_id)
        if sess is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Session not found or expired")
            return fabric_pb2.SessionKeyBundle()
        new_epoch = sess["manager"].rotate()
        logger.info("Key rotated  sid=%s  new_epoch=%d  live=%s",
                    request.session_id, new_epoch, sess["manager"].live_epochs())
        return self._bundle(request.session_id, sess, new_epoch)

    # ── GetSessionKey ─────────────────────────────────────────────────────────

    def GetSessionKey(
        self,
        request: fabric_pb2.KeyRequest,
        context: grpc.ServicerContext,
    ) -> fabric_pb2.SessionKeyBundle:
        """
        Return the key bundle for a specific (session_id, key_epoch). If the
        requested epoch has been retired or never existed, the current epoch is
        returned. Lets the receiving pod follow the sender's rotations exactly.
        """
        sess = self.get_session(request.session_id)
        if sess is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Session not found or expired")
            return fabric_pb2.SessionKeyBundle()
        mgr = sess["manager"]
        epoch = request.key_epoch if mgr.has_epoch(request.key_epoch) else mgr.current_epoch
        return self._bundle(request.session_id, sess, epoch)

    # ── RequestOrthogonalProjections ─────────────────────────────────────────

    def RequestOrthogonalProjections(
        self,
        request: fabric_pb2.KeyRequest,
        context: grpc.ServicerContext,
    ) -> fabric_pb2.ProjectionBundle:
        """Return a fresh orthogonal projection pair (W_A, W_B) for Mode A."""
        W_A, W_B = ce.generate_orthogonal_projections(DIM)
        logger.info("Projections issued  pod=%s  sid=%s", request.pod_id, request.session_id)
        return fabric_pb2.ProjectionBundle(
            session_id=request.session_id,
            W_A=ce.matrix_to_bytes(W_A),
            W_B=ce.matrix_to_bytes(W_B),
            dim=DIM,
        )

    # ── GetRelayInstruction ───────────────────────────────────────────────────

    def GetRelayInstruction(
        self,
        request: fabric_pb2.KeyRequest,
        context: grpc.ServicerContext,
    ) -> fabric_pb2.RelayInstruction:
        """Return routing + amplification config for a relay node."""
        return fabric_pb2.RelayInstruction(
            session_id=request.session_id,
            forward_to=f"{TARGET_HOST}:{TARGET_PORT_ENV}",
            amplify_factor=AMPLIFY_FACTOR,
            log_fingerprint=True,
            rekey=REKEY_EVERY > 0,
        )

    # ── Session helper (used by server.py) ───────────────────────────────────

    def get_session(self, session_id: str):
        s = self._sessions.get(session_id)
        if s and time.time() < s["expires_at"]:
            return s
        if s:
            del self._sessions[session_id]
        return None


def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fabric_pb2_grpc.add_ControlPlaneServicer_to_server(ControlPlaneServicer(), server)
    addr = f"[::]:{CONTROL_PLANE_PORT}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("Control Plane listening on %s", addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
