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
    assert "--vram-profile" in argv
    assert "--cpu" in argv
    assert "--api" in argv
    assert "--nowebui" in argv


def test_launch_settings_argv_includes_vram_profile_aliases():
    high = LaunchSettings(vram_profile="high")
    high_argv = high.argv()
    assert "--vram-profile" in high_argv
    assert "high" in high_argv
    assert "--highvram" in high_argv

    mid = LaunchSettings(vram_profile="medium")
    mid_argv = mid.argv()
    assert "mid" in mid_argv
    assert "--medvram" in mid_argv


def test_launch_settings_argv_includes_attention_backend():
    settings = LaunchSettings(attention_backend="xformers")
    argv = settings.argv()
    assert "--attention-backend" in argv
    assert "xformers" in argv


def test_launch_settings_argv_includes_api_security_flags():
    settings = LaunchSettings(
        api_cors_origins="http://127.0.0.1:3000, https://studio.example",
        api_rate_limit_per_minute=120,
        block_private_download_urls=False,
        gerror=True,
        genlog=True,
    )
    argv = settings.argv()

    assert "--api-cors-origins" in argv
    assert "http://127.0.0.1:3000, https://studio.example" in argv
    assert "--api-rate-limit-per-minute" in argv
    assert "120" in argv
    assert "--allow-private-download-urls" in argv
    assert "--gerror" in argv
    assert "--genlog" in argv


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


def test_launch_settings_argv_includes_pipeline_and_engine_flags():
    settings = LaunchSettings(
        inference_backend="onnx",
        onnx_provider="cuda",
        cuda_malloc=True,
        cuda_graphs=True,
        torchao=True,
        fp8_quant=True,
        torch_compile=True,
        channels_last=True,
        nvenc=True,
        hevc=True,
    )
    argv = settings.argv()

    assert "--inference-backend" in argv
    assert "onnx" in argv
    assert "--onnx-provider" in argv
    assert "cuda" in argv
    assert "--cuda-malloc" in argv
    assert "--cuda-graphs" in argv
    assert "--torchao" in argv
    assert "--fp8-quant" in argv
    assert "--torch-compile" in argv
    assert "--channels-last" in argv
    assert "--nvenc" in argv
    assert "--hevc" in argv


def test_launch_settings_accepts_dual_backend_profile():
    settings = LaunchSettings(inference_backend="dual")
    argv = settings.argv()

    assert "--inference-backend" in argv
    assert "dual" in argv


def test_launch_settings_argv_includes_external_tool_paths():
    settings = LaunchSettings(
        nvidia_vfx_sdk_root="D:\\SDKs\\NVIDIA\\VFX",
        vsr_video_effects_app="D:\\SDKs\\NVIDIA\\VideoEffectsApp.exe",
        vsr_upscale_app="D:\\SDKs\\NVIDIA\\UpscalePipelineApp.exe",
        videofx_denoise_app="D:\\SDKs\\NVIDIA\\DenoiseEffectApp.exe",
        videofx_aigs_app="D:\\SDKs\\NVIDIA\\AigsEffectApp.exe",
        videofx_relight_app="D:\\SDKs\\NVIDIA\\RelightingEffectApp.exe",
        vsr_model_dir="D:\\SDKs\\NVIDIA\\models",
    )
    argv = settings.argv()

    assert "--nvidia-vfx-sdk-root" in argv
    assert "D:\\SDKs\\NVIDIA\\VFX" in argv
    assert "--vsr-video-effects-app" in argv
    assert "D:\\SDKs\\NVIDIA\\VideoEffectsApp.exe" in argv
    assert "--vsr-upscale-app" in argv
    assert "--videofx-denoise-app" in argv
    assert "--videofx-aigs-app" in argv
    assert "--videofx-relight-app" in argv
    assert "--vsr-model-dir" in argv


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
    assert merged.effective_vram_profile() == "cpu"
    assert merged.api is True


def test_merge_launch_settings_applies_saved_high_vram_profile(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(vram_profile="high")

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.effective_vram_profile() == "high"
    assert merged.highvram is True
    assert merged.lowvram is False
    assert merged.medvram is False


def test_merge_launch_settings_respects_explicit_normal_vram_cli(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, vram_profile="normal", lowvram=False, medvram=False, highvram=False)
    saved = LaunchSettings(vram_profile="low")

    merged = merge_launch_settings(cli, saved, explicit={"--normalvram"})

    assert merged.effective_vram_profile() == "normal"
    assert merged.lowvram is False


def test_merge_launch_settings_applies_saved_attention_backend(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, attention_backend="sdpa")
    saved = LaunchSettings(attention_backend="xformers", xformers=True)
    merged = merge_launch_settings(cli, saved, explicit=set())
    assert merged.attention_backend == "xformers"
    assert merged.xformers is True


def test_merge_launch_settings_applies_api_security(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(
        api_cors_origins="https://studio.example",
        api_rate_limit_per_minute=60,
        block_private_download_urls=False,
        genlog=True,
    )

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.api_cors_origins == "https://studio.example"
    assert merged.api_rate_limit_per_minute == 60
    assert merged.block_private_download_urls is False
    assert merged.genlog is True


def test_merge_launch_settings_respects_explicit_genlog_cli(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, genlog=False)
    saved = LaunchSettings(genlog=True)

    merged = merge_launch_settings(cli, saved, explicit={"--genlog"})

    assert merged.genlog is False


def test_merge_launch_settings_applies_gerror(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(gerror=True)

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.gerror is True


def test_merge_launch_settings_respects_explicit_no_autolaunch_cli(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, autolaunch=False)
    saved = LaunchSettings(autolaunch=True)

    merged = merge_launch_settings(cli, saved, explicit={"--no-autolaunch"})

    assert merged.autolaunch is False


def test_merge_launch_settings_respects_explicit_cuda_malloc_cli(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path, cuda_malloc=True)
    saved = LaunchSettings(cuda_malloc=False)

    merged = merge_launch_settings(cli, saved, explicit={"--cuda-malloc"})

    assert merged.cuda_malloc is True


def test_merge_launch_settings_applies_extra_model_dirs(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(
        extra_model_dirs=str(tmp_path / "shared-models"),
        extra_ckpt_dirs=str(tmp_path / "shared-checkpoints"),
    )

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert [str(path) for path in merged.resolved_extra_model_dirs()] == [str((tmp_path / "shared-models").resolve())]
    assert [str(path) for path in merged.resolved_extra_ckpt_dirs()] == [str((tmp_path / "shared-checkpoints").resolve())]


def test_merge_launch_settings_applies_pipeline_and_engine_flags(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(
        inference_backend="onnx",
        onnx_provider="cuda",
        cuda_graphs=True,
        torchao=True,
        fp8_quant=True,
        torch_compile=True,
        channels_last=True,
        nvenc=True,
        hevc=True,
    )

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.inference_backend == "onnx"
    assert merged.onnx_provider == "cuda"
    assert merged.cuda_graphs is True
    assert merged.torchao is True
    assert merged.fp8_quant is True
    assert merged.torch_compile is True
    assert merged.channels_last is True
    assert merged.nvenc is True
    assert merged.hevc is True


def test_merge_launch_settings_applies_dual_backend(tmp_path: Path):
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(inference_backend="dual")

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.inference_backend == "dual"


def test_merge_launch_settings_applies_external_tool_paths(tmp_path: Path):
    sdk = tmp_path / "sdk"
    app = tmp_path / "VideoEffectsApp.exe"
    cli = RuntimeFlags(data_dir=tmp_path)
    saved = LaunchSettings(
        nvidia_vfx_sdk_root=str(sdk),
        vsr_video_effects_app=str(app),
        vsr_model_dir=str(tmp_path / "models"),
    )

    merged = merge_launch_settings(cli, saved, explicit=set())

    assert merged.nvidia_vfx_sdk_root == sdk.resolve()
    assert merged.vsr_video_effects_app == app.resolve()
    assert merged.vsr_model_dir == (tmp_path / "models").resolve()


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


def test_sageattention_install_is_opt_in():
    assert launch.should_install_sageattention([]) is False
    assert launch.should_install_sageattention(["--skip-sageattention"]) is False
    assert launch.should_install_sageattention(["--install-sageattention"]) is True
    assert launch.should_install_sageattention(["--sageattention"]) is True


def test_launch_only_flags_are_not_passed_to_webui():
    argv = ["--install-sageattention", "--listen", "--skip-sageattention", "--port", "7861"]

    assert launch.strip_launch_only_args(argv) == ["--listen", "--port", "7861"]


def test_llm_training_engine_is_optional_by_default():
    specs = {spec.name: spec for spec in launch._build_engine_registry()}

    assert "llm" in specs
    assert specs["llm"].enabled_by_default is False
    assert specs["llm"].skip_flag == "--skip-llm"
    assert specs["llm"].venv_dir.name == ".venv"


def test_ltx_video_engine_uses_specialized_bootstrap():
    specs = {spec.name: spec for spec in launch._build_engine_registry()}

    assert "ltx" in specs
    assert specs["ltx"].enabled_by_default is False
    assert specs["ltx"].skip_flag == "--skip-ltx"
    assert specs["ltx"].manual_bootstrap_script == "scripts/bootstrap_ltx.ps1"
    assert specs["ltx"].cuda_torch is False
