"""
CHORUS Protocol - metrics_logger.py
=====================================
Structured metrics sink for cross-region benchmark runs.
Writes per-chunk signal telemetry and aggregate run summaries to SQLite.
Also streams structured JSON log lines to stdout for container log capture.

Usage:
    logger = MetricsLogger(db_path="chorus_benchmark.db",
                           source_region="eastus",
                           target_region="germanywestcentral")
    run_id = logger.begin_run(mode="isolation")
    logger.record_chunk(run_id, seq_num=0, rtt_ms=42.3, ...)
    logger.finish_run(run_id)
"""

import json
import logging
import os
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

logger = logging.getLogger("chorus.metrics")


@dataclass
class ChunkMetric:
    seq_num:       int
    send_ts:       str
    ack_ts:        str
    rtt_ms:        float
    payload_bytes: int
    signal_norm:   float
    verified:      bool
    status:        str
    mode:          str
    relay_hop:     int   # 0=direct, 1=via relay


@dataclass
class RunAccumulator:
    run_id:    int
    mode:      str
    start_ts:  float = field(default_factory=time.time)
    chunks:    List[ChunkMetric] = field(default_factory=list)


class MetricsLogger:
    """
    Thread-safe metrics sink.
    One instance per benchmark process; one run per mode.
    """

    def __init__(self,
                 db_path:       str = "chorus_benchmark.db",
                 source_region: str = "unknown",
                 target_region: str = "unknown"):
        self.db_path       = db_path
        self.source_region = source_region
        self.target_region = target_region
        self._conn         = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._active_runs: dict[int, RunAccumulator] = {}
        logger.info("MetricsLogger ready  db=%s  %s->%s",
                    db_path, source_region, target_region)

    # ── Run lifecycle ──────────────────────────────────────────────────────────

    def begin_run(self, mode: str, notes: str = "") -> int:
        """
        Insert a placeholder benchmark_run row and return its run_id.
        Values are finalised when finish_run() is called.
        """
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO benchmark_runs
               (run_ts, source_region, target_region, mode,
                n_chunks, total_bytes, duration_ms, throughput_bps,
                avg_rtt_ms, min_rtt_ms, max_rtt_ms,
                p50_rtt_ms, p95_rtt_ms, p99_rtt_ms,
                verified_pct, tamper_count, notes)
               VALUES (?,?,?,?, 0,0,0,0, 0,0,0, 0,0,0, 0,0,?)""",
            (ts, self.source_region, self.target_region, mode, notes),
        )
        self._conn.commit()
        run_id = cur.lastrowid
        self._active_runs[run_id] = RunAccumulator(run_id=run_id, mode=mode)
        self._emit("run_started", run_id=run_id, mode=mode)
        return run_id

    def record_chunk(self, run_id: int, seq_num: int, rtt_ms: float,
                     payload_bytes: int, signal_norm: float,
                     verified: bool, status: str, mode: str,
                     relay_hop: int = 1):
        """Record telemetry for a single transmitted chunk."""
        now = datetime.now(timezone.utc).isoformat()
        send_ts = now   # approximation (exact send_ts passed by benchmark_client)
        ack_ts  = now

        chunk = ChunkMetric(
            seq_num=seq_num, send_ts=send_ts, ack_ts=ack_ts,
            rtt_ms=rtt_ms, payload_bytes=payload_bytes,
            signal_norm=signal_norm, verified=verified,
            status=status, mode=mode, relay_hop=relay_hop,
        )
        if run_id in self._active_runs:
            self._active_runs[run_id].chunks.append(chunk)

        self._conn.execute(
            """INSERT INTO signal_metrics
               (run_id, seq_num, send_ts, ack_ts, rtt_ms, payload_bytes,
                signal_norm, verified, status, mode, relay_hop)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, seq_num, send_ts, ack_ts, rtt_ms, payload_bytes,
             signal_norm, int(verified), status, mode, relay_hop),
        )
        self._conn.commit()

        self._emit("chunk_ack",
                   run_id=run_id, seq=seq_num, rtt_ms=round(rtt_ms, 3),
                   bytes=payload_bytes, norm=round(signal_norm or 0, 4),
                   verified=verified, status=status)

    def finish_run(self, run_id: int) -> dict:
        """
        Compute aggregate statistics and update the benchmark_runs row.
        Returns the summary dict.
        """
        acc = self._active_runs.pop(run_id, None)
        if acc is None or not acc.chunks:
            return {}

        chunks   = acc.chunks
        rtts     = sorted(c.rtt_ms for c in chunks)
        n        = len(chunks)
        total_b  = sum(c.payload_bytes for c in chunks)
        elapsed  = (time.time() - acc.start_ts) * 1000  # ms
        tampers  = sum(1 for c in chunks if not c.verified)
        verified_pct = (n - tampers) / n * 100

        def pct(data, p):
            idx = max(0, int(len(data) * p / 100) - 1)
            return data[idx]

        summary = {
            "run_id":        run_id,
            "mode":          acc.mode,
            "source_region": self.source_region,
            "target_region": self.target_region,
            "n_chunks":      n,
            "total_bytes":   total_b,
            "duration_ms":   round(elapsed, 2),
            "throughput_bps": round(total_b * 8 / (elapsed / 1000), 2) if elapsed > 0 else 0,
            "avg_rtt_ms":    round(statistics.mean(rtts), 3),
            "min_rtt_ms":    round(rtts[0], 3),
            "max_rtt_ms":    round(rtts[-1], 3),
            "p50_rtt_ms":    round(pct(rtts, 50), 3),
            "p95_rtt_ms":    round(pct(rtts, 95), 3),
            "p99_rtt_ms":    round(pct(rtts, 99), 3),
            "verified_pct":  round(verified_pct, 4),
            "tamper_count":  tampers,
        }

        self._conn.execute(
            """UPDATE benchmark_runs SET
               n_chunks=?, total_bytes=?, duration_ms=?, throughput_bps=?,
               avg_rtt_ms=?, min_rtt_ms=?, max_rtt_ms=?,
               p50_rtt_ms=?, p95_rtt_ms=?, p99_rtt_ms=?,
               verified_pct=?, tamper_count=?
               WHERE run_id=?""",
            (summary["n_chunks"], summary["total_bytes"], summary["duration_ms"],
             summary["throughput_bps"], summary["avg_rtt_ms"], summary["min_rtt_ms"],
             summary["max_rtt_ms"], summary["p50_rtt_ms"], summary["p95_rtt_ms"],
             summary["p99_rtt_ms"], summary["verified_pct"], summary["tamper_count"],
             run_id),
        )
        self._conn.commit()
        self._emit("run_complete", **summary)
        self._print_summary(summary)
        return summary

    # ── Query helpers ──────────────────────────────────────────────────────────

    def get_run_summary(self, run_id: int) -> dict:
        row = self._conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return {}
        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM benchmark_runs LIMIT 0").description]
        return dict(zip(cols, row))

    def get_all_runs(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY run_id DESC"
        ).fetchall()
        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM benchmark_runs LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        self._conn.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, event: str, **kwargs):
        """Emit a structured JSON log line for container log capture."""
        record = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "event":  event,
            "source": self.source_region,
            "target": self.target_region,
            **kwargs,
        }
        print(json.dumps(record), flush=True)

    @staticmethod
    def _print_summary(s: dict):
        print("\n" + "=" * 60)
        print(f"  CHORUS Benchmark Run #{s['run_id']}  mode={s['mode']}")
        print(f"  {s['source_region']}  -->  {s['target_region']}")
        print("=" * 60)
        print(f"  Chunks:        {s['n_chunks']:>8,}")
        print(f"  Total bytes:   {s['total_bytes']:>8,}  ({s['total_bytes']/1024/1024:.2f} MB)")
        print(f"  Duration:      {s['duration_ms']:>8.1f} ms")
        print(f"  Throughput:    {s['throughput_bps']/1e6:>8.2f} Mbps")
        print(f"  RTT avg:       {s['avg_rtt_ms']:>8.1f} ms")
        print(f"  RTT min/max:   {s['min_rtt_ms']:>6.1f} / {s['max_rtt_ms']:.1f} ms")
        print(f"  RTT p50/p95:   {s['p50_rtt_ms']:>6.1f} / {s['p95_rtt_ms']:.1f} ms")
        print(f"  RTT p99:       {s['p99_rtt_ms']:>8.1f} ms")
        print(f"  Verified:      {s['verified_pct']:>7.3f}%")
        print(f"  Tamper alerts: {s['tamper_count']:>8,}")
        print("=" * 60)
