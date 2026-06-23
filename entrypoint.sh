#!/bin/sh
# CHORUS Protocol - entrypoint.sh
# Selects the correct Python module based on CHORUS_ROLE.
# For the benchmark role, seeds the SQLite DB first if it doesn't exist.

set -e

ROLE="${CHORUS_ROLE:-client}"
DB="${BENCHMARK_DB:-/data/chorus_benchmark.db}"
LOG_DIR="${LOG_DIR:-/data}"

echo "[CHORUS] Starting role=${ROLE} log_dir=${LOG_DIR}"

# Ensure /data exists (may not be mounted in local dev)
mkdir -p "${LOG_DIR}" 2>/dev/null || true

case "${ROLE}" in
    control_plane)
        exec python control_plane.py
        ;;
    relay)
        exec python relay_node.py
        ;;
    target)
        exec python server.py
        ;;
    client)
        exec python client.py
        ;;
    test)
        exec python test_suite.py
        ;;
    benchmark)
        # Seed the database if it doesn't exist on the file share yet
        if [ ! -f "${DB}" ]; then
            echo "[CHORUS] Seeding benchmark database at ${DB} ..."
            python seed_db.py --db "${DB}" --rows 200000
            echo "[CHORUS] Database seeded."
        else
            echo "[CHORUS] Database already exists at ${DB}, skipping seed."
        fi
        # Short pause to ensure CP / relay / target are fully ready
        sleep 8
        exec python benchmark_client.py
        ;;
    seed)
        exec python seed_db.py --db "${DB}" --rows 200000
        ;;
    *)
        echo "[CHORUS] Unknown role: ${ROLE}" >&2
        exit 1
        ;;
esac
