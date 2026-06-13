from __future__ import annotations

import socket


def find_free_port(start: int = 7860, attempts: int = 32) -> int:
    """Return the first available TCP port starting at `start`."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port found in range {start}-{start + attempts - 1}")