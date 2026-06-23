"""
CHORUS Protocol - log_writer.py
================================
Dual-sink logger: writes every log line to both stdout AND a file
on the Azure File Share (/data/chorus_<role>_<timestamp>.log).

Since ACI containers are ephemeral, stdout vanishes with the container.
The file share mount at /data persists after the container stops,
so this is the durable record of everything that happened.

Log format: one JSON object per line (JSON Lines / ndjson).
This makes the file trivially parseable by jq, Python, or Excel.

Usage:
    from log_writer import ChorusLogger
    log = ChorusLogger(role="benchmark", log_dir="/data")
    log.info("benchmark_started", mode="isolation", chunks=200)
    log.error("watermark_failed", seq=42, session_id="abc")
    log.close()   # flushes and writes a final summary line
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ChorusLogger:
    """
    Thread-safe dual-sink logger.
    Every call to info/warning/error writes:
      1. A JSON line to stdout  (captured by az container logs)
      2. The same JSON line to  /data/chorus_<role>_<ts>.log
         (persisted on the Azure File Share)
    """

    def __init__(self,
                 role:    str = "node",
                 log_dir: str = "/data",
                 region:  str | None = None):
        self.role    = role
        self.region  = region or os.getenv("SOURCE_REGION", "unknown")
        self._start  = time.time()
        self._count  = {"info": 0, "warning": 0, "error": 0}

        # Build log file path
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir_path = Path(log_dir)

        try:
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_path = log_dir_path / f"chorus_{role}_{ts_str}.log"
            self._file = open(log_path, "a", encoding="utf-8", buffering=1)
            self._log_path = str(log_path)
            self._file_available = True
            self._emit_raw("log_opened", level="info",
                           path=self._log_path, role=role, region=self.region)
        except OSError as e:
            # If /data is not mounted (e.g. local dev), fall back to stdout only
            self._file = None
            self._log_path = None
            self._file_available = False
            print(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": "warning",
                "event": "log_file_unavailable",
                "reason": str(e),
                "fallback": "stdout_only",
            }), flush=True)

    # ── Public log methods ────────────────────────────────────────────────────

    def info(self, event: str, **kwargs):
        self._count["info"] += 1
        self._emit_raw(event, level="info", **kwargs)

    def warning(self, event: str, **kwargs):
        self._count["warning"] += 1
        self._emit_raw(event, level="warning", **kwargs)

    def error(self, event: str, **kwargs):
        self._count["error"] += 1
        self._emit_raw(event, level="error", **kwargs)

    def metric(self, event: str, **kwargs):
        """High-frequency metric line — written to file but not stdout to reduce noise."""
        record = self._build_record(event, level="metric", **kwargs)
        line   = json.dumps(record)
        if self._file_available and self._file:
            self._file.write(line + "\n")
        # Still emit to stdout for az container logs capture (concise format)
        print(json.dumps({
            "ts":  record["ts"],
            "ev":  event,
            **{k: v for k, v in kwargs.items()
               if k in ("seq_num", "rtt_ms", "verified", "status", "mode")},
        }), flush=True)

    def close(self):
        """Write a final summary line and flush/close the log file."""
        elapsed = time.time() - self._start
        self._emit_raw(
            "log_closed",
            level="info",
            elapsed_s=round(elapsed, 2),
            total_info=self._count["info"],
            total_warnings=self._count["warning"],
            total_errors=self._count["error"],
            log_path=self._log_path,
        )
        if self._file_available and self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    @property
    def log_path(self) -> str | None:
        return self._log_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_record(self, event: str, level: str, **kwargs) -> dict:
        return {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  level,
            "event":  event,
            "role":   self.role,
            "region": self.region,
            **kwargs,
        }

    def _emit_raw(self, event: str, level: str, **kwargs):
        record = self._build_record(event, level, **kwargs)
        line   = json.dumps(record)
        # Always write to stdout
        print(line, flush=True)
        # Write to file if available
        if self._file_available and self._file:
            try:
                self._file.write(line + "\n")
            except OSError:
                pass   # don't crash the benchmark if disk write fails
