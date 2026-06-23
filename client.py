"""
CHORUS Protocol – client.py
=============================
Source Pod  –  generates, encrypts, and streams tensor signals.

Supported modes:
  "isolation"    – Mode A: two signals projected into orthogonal subspaces,
                   mixed into one wire tensor, sent as one payload.
  "superposition"– Mode B: two signals added together holographically,
                   encrypted as a single consensus tensor.
  "direct"       – single signal, encrypted, sent with no multiplexing.

Downstream routing:
  CHORUS_USE_RELAY=true  -> sends to Relay Node (default)
  CHORUS_USE_RELAY=false -> sends directly to Target Pod
"""

import logging
import os
import time
from typing import Generator, Iterator

import grpc
import torch

import crypto_engine as ce
import fabric_pb2
import fabric_pb2_grpc

logger = logging.getLogger("chorus.client")

CONTROL_PLANE_HOST  = os.getenv("CONTROL_PLANE_HOST", "control-plane")
CONTROL_PLANE_PORT  = int(os.getenv("CONTROL_PLANE_PORT", "50051"))
RELAY_HOST          = os.getenv("RELAY_HOST", "relay-node")
RELAY_PORT          = int(os.getenv("RELAY_PORT", "50052"))
TARGET_HOST         = os.getenv("CHORUS_TARGET_HOST", "target-pod")
TARGET_PORT_ENV     = os.getenv("CHORUS_TARGET_PORT", "50053")
DIM                 = int(os.getenv("CHORUS_DIM", str(ce.DEFAULT_DIM)))
USE_RELAY           = os.getenv("CHORUS_USE_RELAY", "true").lower() == "true"


class ChorusClient:
    """
    Source Pod client.  One instance per pod identity.
    Must call handshake() before streaming.
    """

    def __init__(self, pod_id: str = "pod-A"):
        self.pod_id  = pod_id
        self.dim     = DIM

        # Control Plane connection
        cp_addr = f"{CONTROL_PLANE_HOST}:{CONTROL_PLANE_PORT}"
        self._cp = fabric_pb2_grpc.ControlPlaneStub(grpc.insecure_channel(cp_addr))
        logger.info("[%s] -> Control Plane %s", pod_id, cp_addr)

        # Downstream connection (relay or target)
        if USE_RELAY:
            ds_addr = f"{RELAY_HOST}:{RELAY_PORT}"
            self._relay  = fabric_pb2_grpc.RelayNodeStub(grpc.insecure_channel(ds_addr))
            self._target = None
            logger.info("[%s] -> Relay %s", pod_id, ds_addr)
        else:
            ds_addr = f"{TARGET_HOST}:{TARGET_PORT_ENV}"
            self._relay  = None
            self._target = fabric_pb2_grpc.TargetPodStub(grpc.insecure_channel(ds_addr))
            logger.info("[%s] -> Target (direct) %s", pod_id, ds_addr)

        # Session state – populated by handshake()
        self.session_id:     str | None          = None
        self.K:              torch.Tensor | None = None
        self.K_inv:          torch.Tensor | None = None
        self.watermark_seed: bytes | None        = None
        self.W_A:            torch.Tensor | None = None
        self.W_B:            torch.Tensor | None = None

    # ── Handshake ──────────────────────────────────────────────────────────────

    def handshake(self, request_projections: bool = False) -> str:
        """
        Register with the Control Plane and receive the session key bundle.
        Set request_projections=True for Mode A (Isolation).
        Returns session_id.
        """
        bundle = self._cp.RegisterAndRequestKey(
            fabric_pb2.PodRegistration(pod_id=self.pod_id, pod_role="source")
        )
        self.session_id     = bundle.session_id
        self.K              = ce.bytes_to_matrix(bundle.key_matrix_K,     bundle.dim).double()
        self.K_inv          = ce.bytes_to_matrix(bundle.key_matrix_K_inv, bundle.dim).double()
        self.watermark_seed = bundle.watermark_seed
        logger.info("[%s] Handshake OK  sid=%s  dim=%d  expires=%d",
                    self.pod_id, self.session_id, bundle.dim, bundle.expires_at)

        if request_projections:
            proj = self._cp.RequestOrthogonalProjections(
                fabric_pb2.KeyRequest(pod_id=self.pod_id, session_id=self.session_id)
            )
            self.W_A = ce.bytes_to_matrix(proj.W_A, proj.dim).double()
            self.W_B = ce.bytes_to_matrix(proj.W_B, proj.dim).double()
            logger.info("[%s] Orthogonal projections received.", self.pod_id)

        return self.session_id

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signal(self, concept_seed: int | None = None) -> torch.Tensor:
        """
        Simulate an LLM hidden-state vector (unit-normalised float32).
        In production: tap the actual last hidden state from the model.
        concept_seed makes the vector reproducible for testing.
        """
        if concept_seed is not None:
            torch.manual_seed(concept_seed)
        v = torch.randn(self.dim, dtype=torch.float32)
        return v / (v.norm() + 1e-8)

    # ── Payload builders ───────────────────────────────────────────────────────

    def _wrap(self, wire_tensor: torch.Tensor, mode: str, seq: int) -> fabric_pb2.TensorPayload:
        return fabric_pb2.TensorPayload(
            data=ce.tensor_to_bytes(wire_tensor),
            dim=self.dim, seq_len=1,
            pod_id=self.pod_id, session_id=self.session_id,
            mode=mode, watermark=b"", seq_num=seq,
        )

    def build_isolation_payload(self, v_A: torch.Tensor, v_B: torch.Tensor,
                                 seq: int) -> fabric_pb2.TensorPayload:
        """
        Mode A: Space-Division Orthogonal Multiplexing.
          watermark(V_A) -> encrypt -> project into W_A subspace
          watermark(V_B) -> encrypt -> project into W_B subspace
          V_tunnel = W_A·V_A_enc + W_B·V_B_enc  (single wire tensor)
        """
        assert self.W_A is not None, "Call handshake(request_projections=True) first"
        vA = ce.inject_watermark(v_A, self.watermark_seed, seq)
        vB = ce.inject_watermark(v_B, self.watermark_seed, seq + 10000)
        vA_enc  = ce.encrypt(vA, self.K)
        vB_enc  = ce.encrypt(vB, self.K)
        vA_proj = ce.project_signal(vA_enc, self.W_A)
        vB_proj = ce.project_signal(vB_enc, self.W_B)
        tunnel  = ce.mix_signals(vA_proj, vB_proj)
        return self._wrap(tunnel, "isolation", seq)

    def build_superposition_payload(self, v_A: torch.Tensor, v_B: torch.Tensor,
                                     seq: int) -> fabric_pb2.TensorPayload:
        """
        Mode B: Holographic Superposition.
          V_collective = V_A + V_B  (holographic blend)
          watermark(V_collective) -> encrypt -> send
        """
        collective = ce.superpose_signals(v_A, v_B)
        collective = ce.inject_watermark(collective, self.watermark_seed, seq)
        enc = ce.encrypt(collective, self.K)
        return self._wrap(enc, "superposition", seq)

    def build_direct_payload(self, v_raw: torch.Tensor,
                              seq: int) -> fabric_pb2.TensorPayload:
        """Direct single-signal encrypted stream."""
        vwm = ce.inject_watermark(v_raw, self.watermark_seed, seq)
        enc = ce.encrypt(vwm, self.K)
        return self._wrap(enc, "direct", seq)

    # ── Stream dispatch ────────────────────────────────────────────────────────

    def send_stream(self, payloads: Iterator[fabric_pb2.TensorPayload]) -> list:
        """
        Send a payload stream to relay or target, collect and return all acks.
        """
        acks = []
        if USE_RELAY and self._relay:
            for ack in self._relay.RelayStream(payloads):
                acks.append({"seq": ack.seq_num, "forwarded": ack.forwarded,
                              "status": ack.status})
                logger.info("[%s] RelayAck seq=%d fwd=%s status=%s",
                            self.pod_id, ack.seq_num, ack.forwarded, ack.status)
        elif self._target:
            for ack in self._target.StreamSignal(payloads):
                acks.append({"seq": ack.seq_num, "verified": ack.verified,
                              "norm": ack.signal_norm, "status": ack.status})
                logger.info("[%s] SignalAck seq=%d verified=%s norm=%.4f status=%s",
                            self.pod_id, ack.seq_num, ack.verified,
                            ack.signal_norm, ack.status)
        return acks


# ── Demo entrypoint (used by docker-compose source-pod container) ──────────────

def run_demo():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    time.sleep(5)  # Allow CP / relay / target to start

    pod = ChorusClient(pod_id=os.getenv("POD_ID", "pod-demo"))
    pod.handshake(request_projections=True)

    N = 5  # chunks per mode

    logger.info("═══ Mode A: Orthogonal Isolation ═══")

    def iso():
        for i in range(N):
            yield pod.build_isolation_payload(
                pod.generate_signal(100 + i),
                pod.generate_signal(200 + i),
                seq=i,
            )

    pod.send_stream(iso())

    logger.info("═══ Mode B: Holographic Superposition ═══")

    def sup():
        for i in range(N):
            yield pod.build_superposition_payload(
                pod.generate_signal(100 + i),
                pod.generate_signal(200 + i),
                seq=i,
            )

    pod.send_stream(sup())
    logger.info("═══ Demo complete ═══")


if __name__ == "__main__":
    run_demo()
