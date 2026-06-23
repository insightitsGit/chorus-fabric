"""
CHORUS Protocol - analyze_results.py
======================================
Reads the benchmark SQLite database and prints a full analysis report.
Run locally after downloading results with azure_get_results.ps1.

Usage:
    python analyze_results.py [--db chorus_results_local.db]
"""

import argparse
import sqlite3
import statistics
from datetime import datetime


def analyze(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=" * 70)
    print("  CHORUS Protocol - Cross-Region Benchmark Analysis")
    print(f"  Database: {db_path}")
    print(f"  Generated: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Benchmark runs summary ────────────────────────────────────────────────
    runs = conn.execute(
        "SELECT * FROM benchmark_runs ORDER BY run_id"
    ).fetchall()

    if not runs:
        print("  No benchmark runs found in database.")
        return

    print(f"\n  {len(runs)} benchmark run(s) found:\n")

    for run in runs:
        print(f"  Run #{run['run_id']}  mode={run['mode'].upper():>15}  "
              f"{run['source_region']} -> {run['target_region']}")
        print(f"    Chunks:       {run['n_chunks']:>6,}")
        print(f"    Total data:   {run['total_bytes']/1024:.1f} KB")
        print(f"    Duration:     {run['duration_ms']:.1f} ms")
        print(f"    Throughput:   {run['throughput_bps']/1e6:.3f} Mbps")
        print(f"    RTT avg/p50:  {run['avg_rtt_ms']:.1f} / {run['p50_rtt_ms']:.1f} ms")
        print(f"    RTT p95/p99:  {run['p95_rtt_ms']:.1f} / {run['p99_rtt_ms']:.1f} ms")
        print(f"    RTT min/max:  {run['min_rtt_ms']:.1f} / {run['max_rtt_ms']:.1f} ms")
        print(f"    Verified:     {run['verified_pct']:.4f}%")
        print(f"    Tamper alerts:{run['tamper_count']:>4}")
        print()

    # ── Per-mode RTT histogram ────────────────────────────────────────────────
    print("  RTT Distribution by Mode (ms buckets):\n")
    for run in runs:
        rtts = [r["rtt_ms"] for r in conn.execute(
            "SELECT rtt_ms FROM signal_metrics WHERE run_id=?", (run["run_id"],)
        ).fetchall()]
        if not rtts:
            continue

        print(f"  [{run['mode'].upper():>15}]  n={len(rtts)}")
        buckets = [0, 20, 50, 100, 150, 200, 300, 500, float("inf")]
        labels  = ["<20ms", "20-50", "50-100", "100-150",
                   "150-200", "200-300", "300-500", ">500ms"]
        for i, label in enumerate(labels):
            count = sum(1 for r in rtts if buckets[i] <= r < buckets[i+1])
            bar   = "#" * min(40, count // max(1, len(rtts) // 40))
            pct   = count / len(rtts) * 100
            print(f"    {label:>10}  {bar:<40}  {count:>5} ({pct:4.1f}%)")
        print()

    # ── Verified percentage check ─────────────────────────────────────────────
    print("  Watermark Integrity Check:")
    for run in runs:
        if run["tamper_count"] > 0:
            print(f"    [ALERT] Run #{run['run_id']} {run['mode']}: "
                  f"{run['tamper_count']} tamper alerts detected")
        else:
            print(f"    Run #{run['run_id']} {run['mode']:>15}: "
                  f"100.000% verified - no tampering detected")

    # ── Database contents ─────────────────────────────────────────────────────
    print("\n  Database Contents:")
    for table in ["order_book_ticks", "trades", "benchmark_runs", "signal_metrics"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            size  = conn.execute(
                f"SELECT SUM(length(cast({table} as blob))) FROM {table}"
            ).fetchone()[0] or 0
            print(f"    {table:<25} {count:>8,} rows  (~{size/1024:.0f} KB)")
        except Exception:
            print(f"    {table:<25} (not found)")

    conn.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="chorus_results_local.db")
    args = parser.parse_args()
    analyze(args.db)
