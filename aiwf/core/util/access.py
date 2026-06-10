from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TailscaleInfo:
    installed: bool = False
    connected: bool = False
    ipv4: str | None = None
    ipv6: str | None = None
    dns_name: str | None = None
    backend_state: str | None = None
    tailnet: str | None = None


@dataclass
class NetworkAccessInfo:
    hostname: str
    listen_enabled: bool
    port: int
    localhost_url: str
    lan_urls: list[str] = field(default_factory=list)
    tailscale: TailscaleInfo = field(default_factory=TailscaleInfo)
    recommended_phone_url: str | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _run_command(args: list[str], *, timeout: float = 4.0) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("Command %s failed: %s", args, exc)
        return None
    if result.returncode != 0:
        logger.debug("Command %s exited %s: %s", args, result.returncode, result.stderr.strip())
        return None
    return result.stdout.strip()


def get_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def get_lan_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()

    try:
        for info in socket.getaddrinfo(get_hostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass

    return sorted(addresses)


def _parse_tailscale_json(payload: str) -> TailscaleInfo:
    info = TailscaleInfo(installed=True)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return info

    info.backend_state = data.get("BackendState")
    info.connected = info.backend_state in {"Running", "NeedsLogin"} or bool(data.get("Self"))
    self_info = data.get("Self") or {}
    dns_name = (self_info.get("DNSName") or "").rstrip(".")
    if dns_name:
        info.dns_name = dns_name
    for ip in self_info.get("TailscaleIPs") or []:
        if ":" in ip and not info.ipv6:
            info.ipv6 = ip
        elif "." in ip and not info.ipv4:
            info.ipv4 = ip
    if info.dns_name and "." in info.dns_name:
        parts = info.dns_name.split(".")
        if len(parts) >= 3 and parts[-2] == "ts" and parts[-1] == "net":
            info.tailnet = parts[-3]
    info.connected = info.connected and bool(info.ipv4 or info.dns_name)
    return info


def get_tailscale_info() -> TailscaleInfo:
    if shutil.which("tailscale") is None:
        return TailscaleInfo(installed=False)

    status_json = _run_command(["tailscale", "status", "--json"])
    if status_json:
        return _parse_tailscale_json(status_json)

    info = TailscaleInfo(installed=True)
    ipv4 = _run_command(["tailscale", "ip", "-4"])
    if ipv4:
        info.ipv4 = ipv4.splitlines()[0].strip()
        info.connected = True
    ipv6 = _run_command(["tailscale", "ip", "-6"])
    if ipv6:
        info.ipv6 = ipv6.splitlines()[0].strip()
    return info


def build_network_access_info(*, listen: bool, port: int) -> NetworkAccessInfo:
    hostname = get_hostname()
    lan_ips = get_lan_ipv4_addresses()
    tailscale = get_tailscale_info()
    localhost_url = f"http://127.0.0.1:{port}"
    lan_urls = [f"http://{ip}:{port}" for ip in lan_ips]

    warnings: list[str] = []
    notes: list[str] = []
    recommended: str | None = None

    if not listen:
        warnings.append(
            "Remote devices cannot reach this PC yet. Restart AIWF Studio with `--listen` so it binds to your network interfaces."
        )
    else:
        notes.append("Network listening is enabled (`--listen`). The server accepts connections from other devices.")

    if tailscale.installed:
        if tailscale.connected and tailscale.dns_name:
            recommended = f"http://{tailscale.dns_name}:{port}"
            notes.append("Tailscale is connected. Use the MagicDNS URL on your phone or tablet (Tailscale app must be on).")
        elif tailscale.ipv4 and listen:
            recommended = f"http://{tailscale.ipv4}:{port}"
            notes.append("Tailscale is connected. You can open the Tailscale IP URL from any device on your tailnet.")
        elif not tailscale.connected:
            warnings.append("Tailscale is installed but not connected. Open Tailscale on this PC and sign in.")
    else:
        notes.append("Tailscale CLI not found. Install Tailscale on this PC for easy phone/tablet access away from home.")

    if listen and lan_urls and recommended is None:
        recommended = lan_urls[0]
        notes.append("On the same Wi‑Fi, you can use a LAN address below.")

    return NetworkAccessInfo(
        hostname=hostname,
        listen_enabled=listen,
        port=port,
        localhost_url=localhost_url,
        lan_urls=lan_urls,
        tailscale=tailscale,
        recommended_phone_url=recommended,
        warnings=warnings,
        notes=notes,
    )


def _url_line(label: str, url: str | None, *, highlight: bool = False) -> str:
    if not url:
        return ""
    prefix = "**" if highlight else ""
    suffix = "**" if highlight else ""
    return f"- {prefix}{label}{suffix}: `{url}`"


def format_remote_access_markdown(info: NetworkAccessInfo) -> str:
    lines = [
        "### Remote access",
        "",
        f"**PC name** `{info.hostname}` · **Port** `{info.port}` · **Listen** "
        f"`{'on' if info.listen_enabled else 'off'}`",
        "",
    ]

    if info.recommended_phone_url:
        lines.extend(
            [
                "#### Open on phone or tablet",
                f"`{info.recommended_phone_url}`",
                "",
                "_Copy this into Safari/Chrome on your device (with Tailscale connected)._",
                "",
            ]
        )

    lines.append("#### All URLs")
    lines.append(_url_line("This PC only", info.localhost_url))
    for index, url in enumerate(info.lan_urls, start=1):
        lines.append(_url_line(f"LAN {index}", url))
    if info.tailscale.ipv4:
        lines.append(_url_line("Tailscale IPv4", f"http://{info.tailscale.ipv4}:{info.port}"))
    if info.tailscale.dns_name:
        lines.append(_url_line("Tailscale MagicDNS", f"http://{info.tailscale.dns_name}:{info.port}", highlight=True))

    lines.extend(["", "#### Tailscale"])
    if not info.tailscale.installed:
        lines.append("- **Status**: not installed")
    else:
        state = info.tailscale.backend_state or ("connected" if info.tailscale.connected else "offline")
        lines.append(f"- **Status**: `{state}`")
        if info.tailscale.ipv4:
            lines.append(f"- **Tailscale IP**: `{info.tailscale.ipv4}`")
        if info.tailscale.dns_name:
            lines.append(f"- **MagicDNS**: `{info.tailscale.dns_name}`")
        if info.tailscale.tailnet:
            lines.append(f"- **Tailnet**: `{info.tailscale.tailnet}`")

    if info.warnings:
        lines.extend(["", "#### Action needed"])
        for warning in info.warnings:
            lines.append(f"- ⚠ {warning}")

    if info.notes:
        lines.extend(["", "#### Tips"])
        for note in info.notes:
            lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "#### Quick setup",
            "1. Start AIWF Studio with `--listen` (and keep your chosen port, default `7860`).",
            "2. Install **Tailscale** on this PC and on your phone/tablet; stay signed in on both.",
            "3. On your phone, open the **recommended URL** above in the browser.",
            "",
            "Restart command example:",
            f"`python -m aiwf.app --listen --port {info.port}`",
        ]
    )

    return "\n".join(line for line in lines if line is not None)