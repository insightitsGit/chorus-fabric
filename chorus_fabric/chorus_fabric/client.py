"""
chorus_fabric.client
====================
High-level ChorusClient — the main entry point for developers.

Usage:
    from chorus_fabric import ChorusClient
    import torch

    client = ChorusClient(pod_id="agent-1", control_plane="localhost:50051")
    client.handshake()

    tensor = torch.randn(128)
    acks = client.send_direct(tensor)
"""

import logging
import os
from typing import Iterator, List

import grpc
import torch

from chorus_fabric import crypto_engine as ce
from chorus_fabric import fabric_pb2, fabric_pb2_grpc

logger = logging.getLogger("chorus_fabric.client")


class ChorusClient:
    """
    Source Pod client for the CHORUS Fabric.

    Parameters
    ----------
    pod_id : str
        Unique identifier for this agent/node.
    control_plane : str
        Address of the Control Plane service, e.g. "localhost:50051".
    relay : str | None
        Address of the Relay Node. If None, sends directly to target.
    target : str | None
        Address of the Target Pod (used when relay is None).
    dim : int
        Embedding dimension. Must match the server configuration.
    """

    def __init__(
        self,
        pod_id: str = "pod-A",
        control_plane: str = "localhost:50051",
        relay: str | None = "localhost:50052",
        target: str | None = None,
        dim: int = ce.DEFAULT_DIM,
    ):
        self.pod_id = pod_id
        self.dim = dim

        self._cp = fabric_pb2_grpc.ControlPlaneStub(
            grpc.insecure_channel(control_plane)
        )

        if relay:
            self._relay = fabric_pb2_grpc.RelayNodeStub(
                grpc.insecure_channel(relay)
            )
            self._target_stub = None
            self._use_relay = True
        elif target:
            self._relay = None
            self._target_stub = fabric_pb2_grpc.TargetPodStub(
                grpc.insecure_channel(target)
            )
            self._use_relay = False
        else:
            raise ValueError("Provide either relay= or target= address.")

        # Populated after handshake()
        self.session_id: str | None = None
        self.K: torch.Tensor | None = None
        self.K_inv: torch.Tensor | None = None
        self.watermark_seed: bytes | None = None
        self.W_A: torch.Tensor | None = None
        self.W_B: torch.Tensor | None = None

    # ── Setup ──────────────────────────────────────────────────────────────────

    def handshake(self, isolation_mode: bool = False) -> str:
        """
        Register with the Control Plane and receive an ephemeral session key.

        Parameters
        ----------
        isolation_mode : bool
            Set True to also fetch orthogonal projection matrices (W_A, W_B)
            needed for Mode A (send_isolation).

        Returns
        -------
        str
            The session_id assigned by the Control Plane.
        """
        bundle = self._cp.RegisterAndRequestKey(
            fabric_pb2.PodRegistration(pod_id=self.pod_id, pod_role="source")
        )
        self.session_id = bundle.session_id
        self.K = ce.bytes_to_matrix(bundle.key_matrix_K, bundle.dim).double()
        self.K_inv = ce.bytes_to_matrix(bundle.key_matrix_K_inv, bundle.dim).double()
        self.watermark_seed = bundle.watermark_seed

        if isolation_mode:
            proj = self._cp.RequestOrthogonalProjections(
                fabric_pb2.KeyRequest(
                    pod_id=self.pod_id, session_id=self.session_id
                )
            )
            self.W_A = ce.bytes_to_matrix(proj.W_A, proj.dim).double()
            self.W_B = ce.bytes_to_matrix(proj.W_B, proj.dim).double()

        logger.info(
            "Handshake OK  pod=%s  session=%s  dim=%d  expires=%d",
            self.pod_id, self.session_id, bundle.dim, bundle.expires_at,
        )
        return self.session_id

    # ── Send helpers ───────────────────────────────────────────────────────────

    def send_direct(
        self,
        tensor: torch.Tensor,
        seq_start: int = 0,
    ) -> List[dict]:
        """
        Encrypt and stream a single tensor to the target.

        Parameters
        ----------
        tensor : torch.Tensor
            The signal vector to transmit (shape: [dim] or [N, dim]).
        seq_start : int
            Starting sequence number.

        Returns
        -------
        list[dict]
            List of acknowledgement dicts with keys: seq, verified, status.
        """
        self._require_handshake()
        tensors = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

        def _payloads():
            for i, t in enumerate(tensors):
                wm = ce.inject_watermark(t, self.watermark_seed, seq_start + i)
                enc = ce.encrypt(wm, self.K)
                yield fabric_pb2.TensorPayload(
                    data=ce.tensor_to_bytes(enc),
                    dim=self.dim, seq_len=1,
                    pod_id=self.pod_id, session_id=self.session_id,
                    mode="direct", watermark=b"", seq_num=seq_start + i,
                )

        return self._dispatch(_payloads())

    def send_isolation(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        seq_start: int = 0,
    ) -> List[dict]:
        """
        Mode A — Orthogonal Isolation.
        Send two independent signals over a single encrypted channel.
        The receiver recovers each signal independently with zero crosstalk.

        Requires handshake(isolation_mode=True).

        Parameters
        ----------
        tensor_a : torch.Tensor
            Signal from Agent A  (shape: [dim]).
        tensor_b : torch.Tensor
            Signal from Agent B  (shape: [dim]).
        """
        self._require_handshake()
        if self.W_A is None:
            raise RuntimeError("Call handshake(isolation_mode=True) before send_isolation().")

        def _payloads():
            vA = ce.inject_watermark(tensor_a, self.watermark_seed, seq_start)
            vB = ce.inject_watermark(tensor_b, self.watermark_seed, seq_start + 10000)
            vA_enc = ce.encrypt(vA, self.K)
            vB_enc = ce.encrypt(vB, self.K)
            vA_proj = ce.project_signal(vA_enc, self.W_A)
            vB_proj = ce.project_signal(vB_enc, self.W_B)
            tunnel = ce.mix_signals(vA_proj, vB_proj)
            yield fabric_pb2.TensorPayload(
                data=ce.tensor_to_bytes(tunnel),
                dim=self.dim, seq_len=1,
                pod_id=self.pod_id, session_id=self.session_id,
                mode="isolation", watermark=b"", seq_num=seq_start,
            )

        return self._dispatch(_payloads())

    def send_superposition(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        seq_start: int = 0,
    ) -> List[dict]:
        """
        Mode B — Holographic Superposition.
        Arithmetically blend two signals into one collective transmission.
        Useful for consensus voting, ensemble agreement, and sensor fusion.

        Parameters
        ----------
        tensor_a : torch.Tensor
            First contributing signal  (shape: [dim]).
        tensor_b : torch.Tensor
            Second contributing signal (shape: [dim]).
        """
        self._require_handshake()
        collective = ce.superpose_signals(tensor_a, tensor_b)
        return self.send_direct(collective, seq_start=seq_start)

    def send_stream(
        self,
        payloads: Iterator[fabric_pb2.TensorPayload],
    ) -> List[dict]:
        """
        Low-level: send a pre-built payload iterator and collect all acks.
        Use send_direct / send_isolation / send_superposition for normal use.
        """
        return self._dispatch(payloads)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _dispatch(self, payloads) -> List[dict]:
        acks = []
        if self._use_relay:
            for ack in self._relay.RelayStream(payloads):
                acks.append({
                    "seq": ack.seq_num,
                    "forwarded": ack.forwarded,
                    "status": ack.status,
                })
        else:
            for ack in self._target_stub.StreamSignal(payloads):
                acks.append({
                    "seq": ack.seq_num,
                    "verified": ack.verified,
                    "norm": ack.signal_norm,
                    "status": ack.status,
                })
        return acks

    def _require_handshake(self):
        if self.session_id is None:
            raise RuntimeError("Call handshake() before sending signals.")
