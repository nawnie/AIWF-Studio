from __future__ import annotations

import socket


def find_free_port(start: int = 7860, attempts: int = 32) -> int:
    """Return the first available TCP port starting at `start`."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port found in range {start}-{start + attempts - 1}")
