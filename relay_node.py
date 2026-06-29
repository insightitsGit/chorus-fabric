"""
CHORUS Protocol – relay_node.py
================================
Relay Node  –  Amplifier / Logger / Combiner

Position in topology:
    Source Pod  ->  [Relay Node]  ->  Target Pod

Key design rule: the Relay NEVER decrypts the signal.
  End-to-end confidentiality is preserved because only the Target Pod holds K_inv.
  The Relay operates entirely on the ciphertext, which is safe to amplify
  because encryption is linear: scale(V_enc) decrypts to scale(V_raw).

Responsibilities:
  • Accept encrypted TensorPayload streams from source pods
  • Log a SHA-256 fingerprint of each payload (audit without plaintext exposure)
  • Optionally amplify signal norm (multiply ciphertext vector by scalar)
  • Forward to Target Pod via its own bidirectional gRPC stream
  • Return RelayAck to the upstream source
"""

import hashlib
import logging
import os
import queue
import threading
import time
from concurrent import futures

import grpc
import torch

import crypto_engine as ce
import fabric_pb2
import fabric_pb2_grpc

logger = logging.getLogger("chorus.relay")

RELAY_PORT          = int(os.getenv("RELAY_PORT", "50052"))
CONTROL_PLANE_HOST  = os.getenv("CONTROL_PLANE_HOST", "control-plane")
CONTROL_PLANE_PORT  = int(os.getenv("CONTROL_PLANE_PORT", "50051"))
TARGET_HOST         = os.getenv("CHORUS_TARGET_HOST", "target-pod")
TARGET_PORT_ENV     = os.getenv("CHORUS_TARGET_PORT", "50053")
DIM                 = int(os.getenv("CHORUS_DIM", str(ce.DEFAULT_DIM)))


# ── Signal fingerprint log ─────────────────────────────────────────────────────

class SignalLog:
    """
    Append-only in-memory audit log.
    Stores SHA-256 fingerprints, never raw vectors.
    Phase 2: replace body with OpenTelemetry / Loki sink.
    """
    def __init__(self):
        self._entries = []
        self._lock = threading.Lock()

    def record(self, session_id: str, seq_num: int,
               raw_bytes: bytes, norm_hint: float) -> str:
        fp = hashlib.sha256(raw_bytes).hexdigest()
        with self._lock:
            self._entries.append({
                "ts": time.time(), "session_id": session_id,
                "seq_num": seq_num, "fingerprint": fp, "norm": norm_hint,
            })
        logger.info("[SignalLog] session=%s seq=%d fp=%s…%s norm=%.4f",
                    session_id, seq_num, fp[:8], fp[-8:], norm_hint)
        return fp

    def dump(self) -> list:
        with self._lock:
            return list(self._entries)


SIGNAL_LOG = SignalLog()


# ── Relay Node servicer ────────────────────────────────────────────────────────

class RelayNodeServicer(fabric_pb2_grpc.RelayNodeServicer):

    def __init__(self):
        cp_addr = f"{CONTROL_PLANE_HOST}:{CONTROL_PLANE_PORT}"
        self._cp = fabric_pb2_grpc.ControlPlaneStub(grpc.insecure_channel(cp_addr))

        tgt_addr = f"{TARGET_HOST}:{TARGET_PORT_ENV}"
        self._target = fabric_pb2_grpc.TargetPodStub(grpc.insecure_channel(tgt_addr))

        logger.info("Relay Node initialised  CP=%s  Target=%s", cp_addr, tgt_addr)

    def RelayStream(self, request_iterator, context):
        """
        Bidirectional stream:
          Incoming  <- TensorPayload from source pod
          Outgoing  -> RelayAck     to source pod

        The relay opens its own bidirectional stream to the Target Pod and
        acts as a transparent-but-amplifying intermediary.
        """
        fwd_queue: queue.Queue   = queue.Queue()
        target_acks: dict        = {}
        done                     = threading.Event()

        def _payload_gen():
            while not done.is_set():
                try:
                    item = fwd_queue.get(timeout=0.1)
                    if item is None:
                        break
                    yield item
                except queue.Empty:
                    continue

        def _collect_acks(stream):
            try:
                for ack in stream:
                    target_acks[ack.seq_num] = ack
            except grpc.RpcError as e:
                logger.warning("Target ack stream error: %s", e)
            finally:
                done.set()

        tgt_stream = self._target.StreamSignal(_payload_gen())
        threading.Thread(target=_collect_acks, args=(tgt_stream,), daemon=True).start()

        for payload in request_iterator:
            sid = payload.session_id
            seq = payload.seq_num

            # Fetch routing instruction from Control Plane
            instr = self._get_instruction(sid, payload.pod_id)

            # Log fingerprint (operates on ciphertext only – no decryption)
            norm_hint = self._estimate_norm(payload.data, payload.dim or DIM)
            if instr.log_fingerprint:
                SIGNAL_LOG.record(sid, seq, payload.data, norm_hint)

            # Amplify (scalar multiply on ciphertext – mathematically safe)
            fwd = self._amplify(payload, instr.amplify_factor)
            fwd_queue.put(fwd)

            # Wait up to 50 ms for target ack
            t_ack = None
            for _ in range(50):
                if seq in target_acks:
                    t_ack = target_acks.pop(seq)
                    break
                time.sleep(0.001)

            forwarded = t_ack is not None and t_ack.status == "ok"
            yield fabric_pb2.RelayAck(
                session_id=sid, seq_num=seq, forwarded=forwarded,
                next_hop=instr.forward_to,
                status=t_ack.status if t_ack else "forwarded_no_ack",
            )

        fwd_queue.put(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_instruction(self, sid: str, pod_id: str) -> fabric_pb2.RelayInstruction:
        try:
            return self._cp.GetRelayInstruction(
                fabric_pb2.KeyRequest(pod_id=pod_id, session_id=sid)
            )
        except grpc.RpcError:
            return fabric_pb2.RelayInstruction(
                session_id=sid, forward_to=f"{TARGET_HOST}:{TARGET_PORT_ENV}",
                amplify_factor=1.0, log_fingerprint=True, rekey=False,
            )

    @staticmethod
    def _amplify(p: fabric_pb2.TensorPayload, factor: float) -> fabric_pb2.TensorPayload:
        if abs(factor - 1.0) < 1e-6:
            return p
        dim = p.dim or DIM
        v   = ce.bytes_to_tensor(p.data, dim) * factor
        return fabric_pb2.TensorPayload(
            data=ce.tensor_to_bytes(v), dim=p.dim, seq_len=p.seq_len,
            pod_id=p.pod_id, session_id=p.session_id, mode=p.mode,
            watermark=p.watermark, seq_num=p.seq_num, key_epoch=p.key_epoch,
        )

    @staticmethod
    def _estimate_norm(raw: bytes, dim: int) -> float:
        try:
            return ce.bytes_to_tensor(raw, dim).norm().item()
        except Exception:
            return 0.0


def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fabric_pb2_grpc.add_RelayNodeServicer_to_server(RelayNodeServicer(), server)
    addr = f"[::]:{RELAY_PORT}"
    server.add_insecure_port(addr)
    server.start()
    logger.info("Relay Node listening on %s", addr)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
