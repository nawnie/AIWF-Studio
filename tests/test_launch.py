from pathlib import Path
import sys

from aiwf.core.config.launch import (
    LaunchSettings,
    merge_launch_settings,
    save_launch_settings,
    write_webui_settings_bat,
)
from aiwf.core.config.settings import RuntimeFlags
import launch


def test_launch_settings_argv_includes_listen_and_port():
    settings = LaunchSettings(listen=True, port=8188, xformers=True)
    assert "--listen" in settings.argv()
    assert "--port" in settings.argv()
    assert "8188" in settings.argv()
    assert "--xformers" in settings.argv()


def test_launch_settings_argv_includes_cpu_and_api_flags():
    settings = LaunchSettings(cpu=True, api=True, nowebui=True)
    argv = settings.argv()
    assert "--cpu" in argv
    assert "--api" in argv
    assert "--nowebui" in argv


def test_merge_launch_settings_respects_explicit_cli(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, listen=False, port=7860, xformers=False)
    saved = LaunchSettings(listen=True, port=9000, xformers=True)
    explicit = {"--port"}

    merged = merge_launch_settings(cli, saved, explicit=explicit)

    assert merged.listen is True
    assert merged.port == 7860
    assert merged.xformers is True


def test_merge_launch_settings_applies_saved_cpu_flag(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, cpu=False)
    saved = LaunchSettings(cpu=True, api=True)
    merged = merge_launch_settings(cli, saved, explicit=set())
    assert merged.cpu is True
    assert merged.api is True


def test_write_webui_settings_bat(tmp_path: Path):
    settings = LaunchSettings(listen=True, port=7860, xformers=True)
    bat = write_webui_settings_bat(tmp_path, settings)
    text = bat.read_text(encoding="utf-8")
    assert "COMMANDLINE_ARGS=" in text
    assert "--listen" in text
    assert "--xformers" in text


def test_save_and_load_launch_settings_roundtrip(tmp_path: Path):
    path = tmp_path / "launch.json"
    original = LaunchSettings(listen=True, gradio_auth="user:secret", models_dir=str(tmp_path / "models"))
    save_launch_settings(path, original)
    from aiwf.core.config.launch import load_launch_settings

    loaded = load_launch_settings(path)
    assert loaded is not None
    assert loaded.listen is True
    assert loaded.gradio_auth == "user:secret"
    assert loaded.models_dir.endswith("models")


def test_requirements_satisfied_detects_installed_and_missing(monkeypatch, tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pip>=1\n", encoding="utf-8")
    monkeypatch.setattr(launch, "REQUIREMENTS", requirements)
    assert launch.requirements_satisfied(sys.executable) is True

    requirements.write_text("definitely-missing-aiwf-package==999\n", encoding="utf-8")
    assert launch.requirements_satisfied(sys.executable) is False
