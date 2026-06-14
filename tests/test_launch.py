from pathlib import Path
import sys

from aiwf.core.config.launch import LaunchSettings, merge_launch_settings, save_launch_settings
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


def test_launch_settings_argv_includes_api_security_flags():
    settings = LaunchSettings(
        api_cors_origins="http://127.0.0.1:3000, https://studio.example",
        api_rate_limit_per_minute=120,
        block_private_download_urls=False,
    )
    argv = settings.argv()

    assert "--api-cors-origins" in argv
    assert "http://127.0.0.1:3000, https://studio.example" in argv
    assert "--api-rate-limit-per-minute" in argv
    assert "120" in argv
    assert "--allow-private-download-urls" in argv


def test_launch_settings_argv_includes_extra_model_search_dirs():
    settings = LaunchSettings(
        extra_model_dirs="D:\\Models\\Shared\nE:\\Archive\\Models",
        extra_ckpt_dirs="F:\\Checkpoints",
    )
    argv = settings.argv()

    assert argv.count("--extra-model-dir") == 2
    assert "D:\\Models\\Shared" in argv
    assert "E:\\Archive\\Models" in argv
    assert "--extra-ckpt-dir" in argv
    assert "F:\\Checkpoints" in argv


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


def test_merge_launch_settings_applies_api_security(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(
        api_cors_origins="https://studio.example",
        api_rate_limit_per_minute=60,
        block_private_download_urls=False,
    )

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.api_cors_origins == "https://studio.example"
    assert merged.api_rate_limit_per_minute == 60
    assert merged.block_private_download_urls is False


def test_merge_launch_settings_applies_extra_model_dirs(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(
        extra_model_dirs=str(tmp_path / "shared-models"),
        extra_ckpt_dirs=str(tmp_path / "shared-checkpoints"),
    )

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert [str(path) for path in merged.resolved_extra_model_dirs()] == [str((tmp_path / "shared-models").resolve())]
    assert [str(path) for path in merged.resolved_extra_ckpt_dirs()] == [str((tmp_path / "shared-checkpoints").resolve())]


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
