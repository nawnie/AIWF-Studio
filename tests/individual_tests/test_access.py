from aiwf.core.util.access import (
    NetworkAccessInfo,
    TailscaleInfo,
    build_network_access_info,
    format_remote_access_markdown,
)


def test_build_network_access_warns_without_listen(monkeypatch):
    monkeypatch.setattr(
        "aiwf.core.util.access.get_tailscale_info",
        lambda: TailscaleInfo(installed=False),
    )
    monkeypatch.setattr("aiwf.core.util.access.get_lan_ipv4_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr("aiwf.core.util.access.get_hostname", lambda: "studio-pc")

    info = build_network_access_info(listen=False, port=7860)

    assert info.localhost_url == "http://127.0.0.1:7860"
    assert info.lan_urls == ["http://192.168.1.20:7860"]
    assert any("--listen" in warning for warning in info.warnings)
    assert info.recommended_phone_url is None


def test_build_network_access_prefers_tailscale_dns(monkeypatch):
    monkeypatch.setattr(
        "aiwf.core.util.access.get_tailscale_info",
        lambda: TailscaleInfo(
            installed=True,
            connected=True,
            ipv4="100.64.0.8",
            dns_name="studio-pc.tailnet.ts.net",
            tailnet="tailnet",
            backend_state="Running",
        ),
    )
    monkeypatch.setattr("aiwf.core.util.access.get_lan_ipv4_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr("aiwf.core.util.access.get_hostname", lambda: "studio-pc")

    info = build_network_access_info(listen=True, port=7860)

    assert info.recommended_phone_url == "http://studio-pc.tailnet.ts.net:7860"
    assert not info.warnings


def test_format_remote_access_markdown_includes_phone_url():
    info = NetworkAccessInfo(
        hostname="studio-pc",
        listen_enabled=True,
        port=7860,
        localhost_url="http://127.0.0.1:7860",
        lan_urls=["http://192.168.1.20:7860"],
        tailscale=TailscaleInfo(
            installed=True,
            connected=True,
            ipv4="100.64.0.8",
            dns_name="studio-pc.tailnet.ts.net",
        ),
        recommended_phone_url="http://studio-pc.tailnet.ts.net:7860",
    )

    text = format_remote_access_markdown(info)

    assert "Open on phone or tablet" in text
    assert "http://studio-pc.tailnet.ts.net:7860" in text
    assert "Tailscale MagicDNS" in text
    assert "--listen" in text