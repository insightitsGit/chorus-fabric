"""
CHORUS Protocol - benchmark_client.py
=======================================
Cross-region benchmark client.
Streams all three modes (isolation, superposition, direct) to the remote
Target Pod and records per-chunk RTT, payload size, norm, and watermark
verification result into the SQLite benchmark database.

Environment variables:
  CHORUS_TARGET_HOST        Remote target pod hostname/IP
  CHORUS_TARGET_PORT        (default 50053)
  CHORUS_CONTROL_PLANE_HOST Control plane host
  CHORUS_USE_RELAY          true|false
  CHORUS_DIM                Embedding dimension (default 128)
  BENCHMARK_DB              SQLite file path (default chorus_benchmark.db)
  SOURCE_REGION             Region label for this pod (e.g. eastus)
  TARGET_REGION             Region label for remote pod (e.g. germanywestcentral)
  BENCHMARK_CHUNKS          Number of chunks per mode (default 200)

Usage:
    python benchmark_client.py
"""

import logging
import os
import time
from datetime import datetime, timezone

import torch

import crypto_engine as ce
import fabric_pb2
import fabric_pb2_grpc
import grpc
from client import ChorusClient
from metrics_logger import MetricsLogger
from log_writer import ChorusLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger  = logging.getLogger("chorus.benchmark")
LOG_DIR = os.getenv("LOG_DIR", "/data")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH        = os.getenv("BENCHMARK_DB",    "chorus_benchmark.db")
SOURCE_REGION  = os.getenv("SOURCE_REGION",   "local")
TARGET_REGION  = os.getenv("TARGET_REGION",   "remote")
N_CHUNKS       = int(os.getenv("BENCHMARK_CHUNKS", "200"))
DIM            = int(os.getenv("CHORUS_DIM",  str(ce.DEFAULT_DIM)))
WARMUP_CHUNKS  = 5   # discarded from stats (TCP slow-start / TLS handshake warmup)


def run_benchmark():
    # Dual-sink logger: stdout + persistent file on Azure File Share
    clog = ChorusLogger(role="benchmark", log_dir=LOG_DIR, region=SOURCE_REGION)

    metrics = MetricsLogger(
        db_path=DB_PATH,
        source_region=SOURCE_REGION,
        target_region=TARGET_REGION,
    )

    clog.info("benchmark_started",
              source=SOURCE_REGION, target=TARGET_REGION,
              chunks_per_mode=N_CHUNKS, warmup=WARMUP_CHUNKS,
              db=DB_PATH, log_file=clog.log_path)
    logger.info("CHORUS Cross-Region Benchmark starting")
    logger.info("  Source:  %s", SOURCE_REGION)
    logger.info("  Target:  %s", TARGET_REGION)
    logger.info("  Chunks:  %d per mode (+ %d warmup)", N_CHUNKS, WARMUP_CHUNKS)
    logger.info("  DB:      %s", DB_PATH)
    logger.info("  Log:     %s", clog.log_path)

    # ── Wait for services to be ready ─────────────────────────────────────────
    _wait_for_target()

    # ── Run all three modes ───────────────────────────────────────────────────
    results = {}

    for mode in ["direct", "isolation", "superposition"]:
        clog.info("mode_started", mode=mode)
        logger.info("\n" + "=" * 50)
        logger.info("Starting mode: %s", mode)
        logger.info("=" * 50)
        summary = _run_mode(mode, metrics, clog, N_CHUNKS, WARMUP_CHUNKS)
        results[mode] = summary
        clog.info("mode_complete", mode=mode,
                  avg_rtt_ms=summary.get("avg_rtt_ms"),
                  p99_rtt_ms=summary.get("p99_rtt_ms"),
                  throughput_mbps=round(summary.get("throughput_bps", 0) / 1e6, 3),
                  verified_pct=summary.get("verified_pct"))

    # ── Final comparison table ────────────────────────────────────────────────
    _print_comparison(results)
    clog.info("benchmark_complete", modes=list(results.keys()),
              log_file=clog.log_path, db=DB_PATH)
    clog.close()
    metrics.close()


def _run_mode(mode: str, metrics: MetricsLogger, clog: ChorusLogger,
              n_chunks: int, warmup: int) -> dict:
    """Stream n_chunks payloads in the given mode, record metrics."""
    pod = ChorusClient(pod_id=f"bench-{SOURCE_REGION}")
    pod.handshake(request_projections=(mode == "isolation"))

    run_id = metrics.begin_run(mode=mode,
                               notes=f"cross_region {SOURCE_REGION}->{TARGET_REGION}")

    total_chunks = warmup + n_chunks

    def payload_generator():
        for i in range(total_chunks):
            vA = pod.generate_signal(concept_seed=100 + i)
            vB = pod.generate_signal(concept_seed=200 + i)

            if mode == "isolation":
                pld = pod.build_isolation_payload(vA, vB, seq=i)
            elif mode == "superposition":
                pld = pod.build_superposition_payload(vA, vB, seq=i)
            else:
                pld = pod.build_direct_payload(vA, seq=i)

            yield pld

    # Determine which stub to call
    use_relay = os.getenv("CHORUS_USE_RELAY", "true").lower() == "true"

    if use_relay and pod._relay:
        ack_iter = pod._relay.RelayStream(payload_generator())
    else:
        ack_iter = pod._target.StreamSignal(payload_generator())

    send_times: dict[int, float] = {}
    seq_counter = [0]

    def timed_payloads():
        for pld in payload_generator():
            send_times[pld.seq_num] = time.perf_counter()
            seq_counter[0] = pld.seq_num
            yield pld

    # Re-run with timing (can't instrument the generator after the fact cleanly)
    pod2 = ChorusClient(pod_id=f"bench2-{SOURCE_REGION}")
    pod2.handshake(request_projections=(mode == "isolation"))

    def timed_gen():
        for i in range(total_chunks):
            vA = pod2.generate_signal(concept_seed=100 + i)
            vB = pod2.generate_signal(concept_seed=200 + i)
            if mode == "isolation":
                pld = pod2.build_isolation_payload(vA, vB, seq=i)
            elif mode == "superposition":
                pld = pod2.build_superposition_payload(vA, vB, seq=i)
            else:
                pld = pod2.build_direct_payload(vA, seq=i)
            send_times[i] = time.perf_counter()
            yield pld

    if use_relay and pod2._relay:
        ack_stream = pod2._relay.RelayStream(timed_gen())
    else:
        ack_stream = pod2._target.StreamSignal(timed_gen())

    chunk_count = 0
    for ack in ack_stream:
        seq     = ack.seq_num
        recv_ts = time.perf_counter()
        send_t  = send_times.get(seq)
        rtt_ms  = (recv_ts - send_t) * 1000 if send_t else 0.0

        payload_bytes = DIM * 4  # float32 per dimension
        verified      = getattr(ack, "verified", True)
        status        = getattr(ack, "status",   "ok")
        signal_norm   = getattr(ack, "signal_norm", 0.0)

        # Skip warmup chunks from recorded stats
        if seq >= warmup:
            metrics.record_chunk(
                run_id=run_id,
                seq_num=seq,
                rtt_ms=rtt_ms,
                payload_bytes=payload_bytes,
                signal_norm=signal_norm,
                verified=verified,
                status=status,
                mode=mode,
                relay_hop=1 if use_relay else 0,
            )
            chunk_count += 1

        # Every chunk written to file log; every 50th also to stdout
        clog.metric("chunk_ack", seq_num=seq, rtt_ms=round(rtt_ms, 2),
                    verified=verified, status=status, mode=mode)
        if seq % 50 == 0:
            logger.info("[%s] seq=%d  rtt=%.1f ms  norm=%.4f  status=%s",
                        mode, seq, rtt_ms, signal_norm or 0, status)

    summary = metrics.finish_run(run_id)
    return summary


def _wait_for_target(timeout: int = 60):
    """Poll the Target Pod until it responds or timeout."""
    target_host = os.getenv("CHORUS_TARGET_HOST", "target-pod")
    target_port = int(os.getenv("CHORUS_TARGET_PORT", "50053"))
    addr        = f"{target_host}:{target_port}"
    deadline    = time.time() + timeout

    logger.info("Waiting for target pod at %s ...", addr)
    while time.time() < deadline:
        try:
            ch = grpc.insecure_channel(addr)
            grpc.channel_ready_future(ch).result(timeout=3)
            logger.info("Target pod ready at %s", addr)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Target pod at {addr} not ready after {timeout}s")


def _print_comparison(results: dict):
    modes = list(results.keys())
    print("\n" + "=" * 70)
    print("  CHORUS Cross-Region Benchmark  -  Mode Comparison")
    print(f"  {results[modes[0]].get('source_region','')} -> "
          f"{results[modes[0]].get('target_region','')}")
    print("=" * 70)
    print(f"  {'Metric':<22} " + "  ".join(f"{m.upper():>14}" for m in modes))
    print("-" * 70)

    metrics_to_show = [
        ("avg_rtt_ms",    "RTT avg (ms)"),
        ("p50_rtt_ms",    "RTT p50 (ms)"),
        ("p95_rtt_ms",    "RTT p95 (ms)"),
        ("p99_rtt_ms",    "RTT p99 (ms)"),
        ("throughput_bps","Throughput (Mbps)"),
        ("n_chunks",      "Chunks sent"),
        ("total_bytes",   "Total bytes"),
        ("verified_pct",  "Verified (%)"),
        ("tamper_count",  "Tamper alerts"),
    ]
    for key, label in metrics_to_show:
        vals = []
        for m in modes:
            v = results.get(m, {}).get(key, 0)
            if key == "throughput_bps":
                vals.append(f"{v/1e6:>14.3f}")
            elif key == "total_bytes":
                vals.append(f"{v/1024:>12.1f}KB")
            elif isinstance(v, float):
                vals.append(f"{v:>14.3f}")
            else:
                vals.append(f"{v:>14,}")
        print(f"  {label:<22} " + "  ".join(vals))
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
