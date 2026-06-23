"""
CHORUS Protocol - seed_db.py
==============================
Generates and seeds a SQLite database with realistic high-frequency
order book tick data (~200k rows, ~40MB) for use as a heavy payload
stress test in the cross-region benchmark.

Schema:
  order_book_ticks  - raw L2 order book updates (bid/ask/size/exchange)
  trades            - executed trade events
  benchmark_runs    - CHORUS protocol benchmark results (written during test)
  signal_metrics    - per-chunk telemetry from the gRPC stream

Run:
    python seed_db.py [--db PATH] [--rows N]
"""

import argparse
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

# Reproducible generation
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DB   = "chorus_benchmark.db"
TICKERS      = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                 "TSLA", "JPM",  "GS",    "BAC",  "BTC",  "ETH",
                 "EUR",  "GBP",  "XAU",   "WTI",  "SPY",  "QQQ",
                 "VIX",  "ES1"]  # 20 instruments across stocks/crypto/FX/futures
EXCHANGES    = ["NYSE", "NASDAQ", "LSE", "XETRA", "CBOE", "BINANCE", "CME"]
SIDES        = ["bid", "ask"]
ORDER_TYPES  = ["limit", "market", "iceberg", "stop"]

# Base prices per instrument (realistic starting point)
BASE_PRICES  = {
    "AAPL": 185.0, "MSFT": 420.0, "GOOGL": 175.0, "AMZN": 190.0,
    "NVDA": 900.0, "META": 500.0, "TSLA": 240.0,  "JPM":  195.0,
    "GS":   480.0, "BAC":   38.0, "BTC": 68000.0,  "ETH":  3500.0,
    "EUR":    1.09, "GBP":   1.27, "XAU": 2350.0,  "WTI":   82.0,
    "SPY":   525.0, "QQQ":  450.0, "VIX":   14.5,  "ES1":  5300.0,
}


# ── Helper generators ─────────────────────────────────────────────────────────

def _walk_price(price: float, volatility: float = 0.0005) -> float:
    """Random walk with mean reversion."""
    return max(0.01, price * (1 + random.gauss(0, volatility)))


def _random_ts(base: datetime, delta_ms: int) -> str:
    return (base + timedelta(milliseconds=delta_ms)).strftime("%Y-%m-%d %H:%M:%S.%f")


def generate_order_book_ticks(n: int, prices: dict):
    """
    Yield n order book update rows as tuples matching the DB schema.
    Simulates realistic bid/ask spread dynamics with occasional large orders.
    """
    base_ts  = datetime(2024, 1, 2, 9, 30, 0)  # US market open
    ts_ms    = 0
    prices   = dict(prices)

    for i in range(n):
        ticker   = random.choice(TICKERS)
        price    = prices[ticker]
        price    = _walk_price(price)
        prices[ticker] = price

        spread   = price * random.uniform(0.0001, 0.0005)
        side     = random.choice(SIDES)
        bid      = round(price - spread / 2, 4)
        ask      = round(price + spread / 2, 4)
        quote    = bid if side == "bid" else ask
        size     = round(random.lognormvariate(4, 1.5), 2)  # log-normal size
        depth    = random.randint(1, 10)                     # order book depth level
        exchange = random.choice(EXCHANGES)
        otype    = random.choices(ORDER_TYPES, weights=[60, 25, 10, 5])[0]

        # Advance timestamp by 0-50ms (burst) or 50-500ms (normal)
        ts_ms   += random.randint(0, 50) if random.random() < 0.3 else random.randint(50, 500)
        ts_str   = _random_ts(base_ts, ts_ms)

        # Raw vector blob: simulate a 128-dim embedding of the tick (32 bytes per dim)
        vector_blob = bytes([random.randint(0, 255) for _ in range(512)])  # 128 x float32

        yield (
            i + 1,         # tick_id
            ts_str,        # timestamp
            ticker,        # instrument
            exchange,      # exchange
            side,          # side (bid/ask)
            otype,         # order_type
            round(quote, 4),  # price
            round(size, 2),   # size
            depth,         # depth_level
            round(bid, 4), # best_bid
            round(ask, 4), # best_ask
            round(ask - bid, 6),  # spread
            vector_blob,   # embedding_blob (128-dim float32)
        )


def generate_trades(n: int, prices: dict):
    """Yield n executed trade rows."""
    base_ts = datetime(2024, 1, 2, 9, 30, 5)
    ts_ms   = 0
    prices  = dict(prices)

    for i in range(n):
        ticker   = random.choice(TICKERS)
        price    = _walk_price(prices[ticker])
        prices[ticker] = price
        ts_ms   += random.randint(100, 2000)
        ts_str   = _random_ts(base_ts, ts_ms)
        aggressor = random.choice(["buy", "sell"])
        size     = round(random.lognormvariate(3.5, 1.2), 2)

        yield (
            i + 1,
            ts_str,
            ticker,
            random.choice(EXCHANGES),
            aggressor,
            round(price, 4),
            round(size, 2),
            round(price * size, 2),  # notional
        )


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS order_book_ticks (
    tick_id        INTEGER PRIMARY KEY,
    timestamp      TEXT    NOT NULL,
    instrument     TEXT    NOT NULL,
    exchange       TEXT    NOT NULL,
    side           TEXT    NOT NULL,
    order_type     TEXT    NOT NULL,
    price          REAL    NOT NULL,
    size           REAL    NOT NULL,
    depth_level    INTEGER NOT NULL,
    best_bid       REAL    NOT NULL,
    best_ask       REAL    NOT NULL,
    spread         REAL    NOT NULL,
    embedding_blob BLOB                -- 128-dim float32 vector (512 bytes)
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id   INTEGER PRIMARY KEY,
    timestamp  TEXT    NOT NULL,
    instrument TEXT    NOT NULL,
    exchange   TEXT    NOT NULL,
    aggressor  TEXT    NOT NULL,
    price      REAL    NOT NULL,
    size       REAL    NOT NULL,
    notional   REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_instrument ON order_book_ticks(instrument);
CREATE INDEX IF NOT EXISTS idx_ticks_ts         ON order_book_ticks(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument);

-- CHORUS benchmark results (written during cross-region test)
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts         TEXT    NOT NULL,
    source_region  TEXT    NOT NULL,
    target_region  TEXT    NOT NULL,
    mode           TEXT    NOT NULL,   -- isolation | superposition | direct
    n_chunks       INTEGER NOT NULL,
    total_bytes    INTEGER NOT NULL,
    duration_ms    REAL    NOT NULL,
    throughput_bps REAL    NOT NULL,
    avg_rtt_ms     REAL    NOT NULL,
    min_rtt_ms     REAL    NOT NULL,
    max_rtt_ms     REAL    NOT NULL,
    p50_rtt_ms     REAL    NOT NULL,
    p95_rtt_ms     REAL    NOT NULL,
    p99_rtt_ms     REAL    NOT NULL,
    verified_pct   REAL    NOT NULL,
    tamper_count   INTEGER NOT NULL,
    notes          TEXT
);

-- Per-chunk signal telemetry
CREATE TABLE IF NOT EXISTS signal_metrics (
    metric_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES benchmark_runs(run_id),
    seq_num        INTEGER NOT NULL,
    send_ts        TEXT    NOT NULL,
    ack_ts         TEXT    NOT NULL,
    rtt_ms         REAL    NOT NULL,
    payload_bytes  INTEGER NOT NULL,
    signal_norm    REAL,
    verified       INTEGER NOT NULL,   -- 0/1
    status         TEXT    NOT NULL,
    mode           TEXT    NOT NULL,
    relay_hop      INTEGER NOT NULL    -- 0=direct, 1=via relay
);
"""


# ── Seeder ────────────────────────────────────────────────────────────────────

def seed(db_path: str, tick_rows: int = 200_000, trade_rows: int = 20_000):
    print(f"Seeding CHORUS benchmark database: {db_path}")
    start = time.time()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB cache
    conn.executescript(SCHEMA)

    # ── Order book ticks ──────────────────────────────────────────────────────
    print(f"  Generating {tick_rows:,} order book ticks...")
    batch, batch_size = [], 5000
    t0 = time.time()
    for row in generate_order_book_ticks(tick_rows, dict(BASE_PRICES)):
        batch.append(row)
        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO order_book_ticks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            batch.clear()
            progress = row[0] / tick_rows * 100
            print(f"    {row[0]:>7,} / {tick_rows:,}  ({progress:.0f}%)", end="\r")
    if batch:
        conn.executemany(
            "INSERT INTO order_book_ticks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()
    print(f"    {tick_rows:,} ticks written in {time.time()-t0:.1f}s        ")

    # ── Trades ────────────────────────────────────────────────────────────────
    print(f"  Generating {trade_rows:,} trades...")
    batch = []
    for row in generate_trades(trade_rows, dict(BASE_PRICES)):
        batch.append(row)
    conn.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
        batch
    )
    conn.commit()
    print(f"    {trade_rows:,} trades written")

    # ── Stats ─────────────────────────────────────────────────────────────────
    db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
    elapsed    = time.time() - start

    counts = {
        "ticks":  conn.execute("SELECT COUNT(*) FROM order_book_ticks").fetchone()[0],
        "trades": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
    }
    conn.close()

    print(f"\n  Database ready:")
    print(f"    Path:        {db_path}")
    print(f"    Size:        {db_size_mb:.1f} MB")
    print(f"    Ticks:       {counts['ticks']:,}")
    print(f"    Trades:      {counts['trades']:,}")
    print(f"    Total time:  {elapsed:.1f}s")
    return db_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed CHORUS benchmark SQLite database")
    parser.add_argument("--db",   default=DEFAULT_DB, help="SQLite file path")
    parser.add_argument("--rows", type=int, default=200_000, help="Order book tick rows")
    args = parser.parse_args()
    seed(args.db, tick_rows=args.rows)
