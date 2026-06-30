import pytest
import json

from aiwf import app_pro
from aiwf.app import (
    _ConsoleNoiseFilter,
    _auth_pairs,
    _friendly_device_name,
    _friendly_library_message,
    _gradio_allowed_paths,
)
from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.user_messages import attention_display_label
import logging


def test_friendly_device_name_trims_cuda_details():
    assert (
        _friendly_device_name("CUDA (NVIDIA GeForce RTX 4070 Ti SUPER, 16.0 GB VRAM, torch cuda 12.4)")
        == "NVIDIA GeForce RTX 4070 Ti SUPER"
    )


def test_friendly_device_name_maps_cpu_mode():
    assert _friendly_device_name("CPU (slow - install CUDA PyTorch for GPU acceleration)") == "CPU mode"


def test_friendly_library_message_when_models_exist():
    assert _friendly_library_message(2, 0) == "Library ready with 2 base models."
    assert _friendly_library_message(1, 3) == "Library ready with 1 base model and 3 LoRAs."


def test_friendly_library_message_when_no_models_exist():
    assert _friendly_library_message(0, 0) == (
        "No base models were found yet. Add one in Models or import another library in Settings."
    )


def test_attention_label_prefers_sage_when_available(monkeypatch):
    monkeypatch.setattr("aiwf.core.user_messages.sageattention_2_available", lambda: True)

    assert attention_display_label(RuntimeFlags()) == "Sage"


def test_attention_label_reports_sdpa_when_explicit():
    assert attention_display_label(RuntimeFlags(attention_backend="sdpa")) == "SDPA"


def test_console_filter_keeps_aiwf_and_hides_third_party_warning():
    filt = _ConsoleNoiseFilter()

    assert filt.filter(logging.LogRecord("aiwf.app", logging.INFO, "", 1, "ready", (), None))
    assert not filt.filter(logging.LogRecord("diffusers", logging.WARNING, "", 1, "noise", (), None))
    assert filt.filter(logging.LogRecord("diffusers", logging.ERROR, "", 1, "failure", (), None))


def test_auth_pairs_parse_multiple_users():
    assert _auth_pairs("shawn:secret, admin:other ") == [("shawn", "secret"), ("admin", "other")]


def test_auth_pairs_reject_invalid_auth():
    with pytest.raises(ValueError, match="gradio_auth"):
        _auth_pairs("shawn")
    with pytest.raises(ValueError, match="gradio_auth"):
        _auth_pairs("shawn:")


def test_gradio_allowed_paths_only_exposes_outputs(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, output_dir=tmp_path / "outputs", models_dir=tmp_path / "models")

    assert _gradio_allowed_paths(flags) == [str((tmp_path / "outputs").resolve())]


def test_pro_app_window_prefers_edge_app_mode(monkeypatch, tmp_path):
    edge = tmp_path / "msedge.exe"

    def fake_which(name: str):
        return str(edge) if name == "msedge" else None

    monkeypatch.setattr(app_pro.shutil, "which", fake_which)

    assert app_pro._browser_app_command("http://127.0.0.1:7860") == [
        str(edge),
        "--new-window",
        "--start-fullscreen",
        "--app=http://127.0.0.1:7860",
    ]


def test_pro_autolaunch_is_default_with_opt_out_flag():
    assert app_pro._pro_autolaunch_enabled([])
    assert app_pro._pro_autolaunch_enabled(["--port", "7891"])
    assert not app_pro._pro_autolaunch_enabled(["--no-autolaunch"])


def test_existing_pro_runtime_detects_live_pro_payload(monkeypatch):
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"status": "idle", "backend": "diffusers"}).encode("utf-8")

    monkeypatch.setattr(app_pro.urllib.request, "urlopen", lambda *args, **kwargs: _Response())

    assert app_pro._existing_pro_runtime("http://127.0.0.1:7860")
