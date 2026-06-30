import socket

from aiwf.core.util.network import find_free_port


def test_find_free_port_skips_listener_bound_to_all_interfaces():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", 0))
        busy_port = probe.getsockname()[1]
        probe.listen(1)

        selected = find_free_port(busy_port, attempts=16)

    assert selected != busy_port
    assert selected > busy_port
