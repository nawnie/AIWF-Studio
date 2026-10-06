from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_chat.backends.base import BackendError
from simple_ai_chat.backends.llama_server import LlamaServerBackend, build_command

from .conftest import CONFIG_DIR, spec


def args_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_brain_command_from_shipped_config():
    from simple_ai_chat.registry import Registry

    registry = Registry.from_yaml(CONFIG_DIR / "models.yaml", CONFIG_DIR / "hardware.yaml")
    cmd = build_command(registry.get("bonsai2-27b"), binary="llama-server", port=8100, models_dir="models")
    assert cmd[0] == "llama-server"
    assert Path(args_after(cmd, "-m")) == Path("models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    assert args_after(cmd, "-c") == "32768"
    assert args_after(cmd, "--cache-ram") == "24576"
    assert args_after(cmd, "--flash-attn") == "on"
    assert "--jinja" in cmd
    assert "-ctk" not in cmd  # f16 KV is the default
    assert "--no-mmproj-offload" not in cmd


def test_kv_type_mmproj_placeholders_and_flags():
    model = spec(
        "q38",
        ctx=32768,
        kv_type="q4_0",
        mmproj_on_gpu=False,
        files={"model": "m.gguf", "mmproj": "p.gguf", "draft": "d.gguf"},
        launch={
            "spec_type": "draft-mtp",
            "spec_draft_model": "{draft}",
            "parallel": 1,
            "verbose": False,
            "metrics": True,
            "port": 9999,
            "extra": ["--seed", "42"],
        },
    )
    cmd = build_command(model, binary="llama-server", port=8123, models_dir="/m")
    assert args_after(cmd, "--port") == "8123"
    assert args_after(cmd, "-ctk") == "q4_0" and args_after(cmd, "-ctv") == "q4_0"
    assert "--no-mmproj-offload" in cmd
    assert Path(args_after(cmd, "--spec-draft-model")) == Path("/m/q38/d.gguf")
    assert "--metrics" in cmd and "--verbose" not in cmd
    assert cmd[-2:] == ["--seed", "42"]
    assert "9999" not in cmd  # launch.port is consumed by the backend, not passed through


def test_command_errors():
    with pytest.raises(BackendError, match="files.model"):
        build_command(spec("x"), binary="b", port=1, models_dir="m")
    bad = spec("x", files={"model": "m.gguf"}, launch={"spec_draft_model": "{draft}"})
    with pytest.raises(BackendError, match="draft"):
        build_command(bad, binary="b", port=1, models_dir="m")


def test_ports_are_stable_and_respect_explicit(tmp_path):
    backend = LlamaServerBackend(tmp_path / "llama-server", tmp_path, base_port=8200)
    a, b = spec("a"), spec("b")
    pinned = spec("c", launch={"port": 9000})
    assert backend.port_for(a) == 8200
    assert backend.port_for(b) == 8201
    assert backend.port_for(a) == 8200
    assert backend.port_for(pinned) == 9000


def test_load_fails_cleanly_without_binary_or_model(tmp_path):
    backend = LlamaServerBackend(tmp_path / "missing-server", tmp_path)
    with pytest.raises(BackendError, match="binary"):
        backend.load(spec("a", files={"model": "m.gguf"}))

    binary = tmp_path / "llama-server"
    binary.write_text("")
    backend = LlamaServerBackend(binary, tmp_path)
    with pytest.raises(BackendError, match="model file"):
        backend.load(spec("a", files={"model": "m.gguf"}))
    assert backend.save_state("a") is False
