# ─────────────────────────────────────────────────────────────────────────────
# CHORUS Protocol – Unified Container Image
#
# Single image for all four roles.
# Role is selected at runtime by the CHORUS_ROLE environment variable:
#
#   control_plane  →  python control_plane.py
#   relay          →  python relay_node.py
#   target         →  python server.py
#   client         →  python client.py
#   test           →  python test_suite.py
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (large layer, cache separately)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ ./proto/
COPY *.py ./

# Generate gRPC stubs from fabric.proto → fabric_pb2.py + fabric_pb2_grpc.py
RUN python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=. \
    --grpc_python_out=. \
    ./proto/fabric.proto

ENV CHORUS_ROLE=client
ENV LOG_DIR=/data

# entrypoint.sh handles:
#   1. Seed the SQLite DB if role=benchmark and DB doesn't exist yet
#   2. Launch the appropriate Python module for the role
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
