"""
chorus_fabric.servers
=====================
Convenience launchers for running CHORUS services.

    from chorus_fabric.servers import start_control_plane, start_target, start_relay

Or via CLI:
    python -m chorus_fabric.servers control_plane
    python -m chorus_fabric.servers target
    python -m chorus_fabric.servers relay
"""

import sys


def start_control_plane():
    from chorus_fabric.control_plane import serve
    serve()


def start_target():
    from chorus_fabric.server import serve
    serve()


def start_relay():
    from chorus_fabric.relay_node import serve
    serve()


if __name__ == "__main__":
    cmds = {"control_plane": start_control_plane, "target": start_target, "relay": start_relay}
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in cmds:
        print(f"Usage: python -m chorus_fabric.servers [{'|'.join(cmds)}]")
        sys.exit(1)
    cmds[cmd]()
