from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiwf.core.config.settings import RuntimeFlags, normalize_vram_profile

LAUNCH_FILENAME = "launch.json"


class LaunchSettings(BaseSettings):
    """User-editable options applied on the next app start.

    This is the persisted launch profile, not live process state. Explicit CLI
    flags win during merge so a one-off safe/local override is not overwritten
    by an older launch.json.
    """

    model_config = SettingsConfigDict(extra="ignore")

    listen: bool = False
    port: int = Field(default=7860, ge=1024, le=65535)
    autolaunch: bool = False
    theme: str = "dark"
    gradio_auth: str = ""
    api_cors_origins: str = ""
    api_rate_limit_per_minute: int = Field(default=0, ge=0, le=6000)
    # Security default for model downloads: block loopback/LAN/private targets
    # unless the user intentionally enables local private URL fetching.
    block_private_download_urls: bool = True
    gerror: bool = False
    genlog: bool = False
    share: bool = False
    medvram: bool = False
    lowvram: bool = False
    attention_backend: str = "sdpa"
    xformers: bool = False
    opt_sdp_attention: bool = False
    opt_split_attention: bool = False
    async_offload: bool = True
    pinned_memory: bool = True
    cuda_malloc: bool = False
    no_half: bool = False
    fp8: bool = False
    fluxfp8: bool = False
    directml: bool = False
    cpu: bool = False
    inference_backend: str = "diffusers"
    onnx_provider: str = "auto"
    vram_profile: str = "normal"
    highvram: bool = False
    cuda_graphs: bool = False
    torchao: bool = False
    fp8_quant: bool = False
    torch_compile: bool = False
    channels_last: bool = False
    nvenc: bool = False
    hevc: bool = False
    api: bool = False
    nowebui: bool = False
    models_dir: str = ""
    ckpt_dir: str = ""
    output_dir: str = ""
    # Newline-delimited text mirrors the Settings UI text boxes; conversion to
    # Path lists happens only when the profile becomes RuntimeFlags.
    extra_model_dirs: str = ""
    extra_ckpt_dirs: str = ""
    nvidia_vfx_sdk_root: str = ""
    vsr_video_effects_app: str = ""
    vsr_upscale_app: str = ""
    videofx_denoise_app: str = ""
    videofx_aigs_app: str = ""
    videofx_relight_app: str = ""
    vsr_model_dir: str = ""

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"dark", "light"}:
            raise ValueError("theme must be dark or light")
        return normalized

    @field_validator("gradio_auth")
    @classmethod
    def validate_gradio_auth(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return ""
        if ":" not in cleaned:
            raise ValueError("gradio_auth must be username:password")
        return cleaned

    @field_validator("inference_backend")
    @classmethod
    def validate_inference_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"diffusers", "onnx", "sdcpp", "dual"}:
            raise ValueError("inference_backend must be diffusers, onnx, sdcpp, or dual")
        return normalized

    @field_validator("onnx_provider")
    @classmethod
    def validate_onnx_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "cuda", "directml", "cpu"}:
            raise ValueError("onnx_provider must be auto, cuda, directml, or cpu")
        return normalized

    @field_validator("attention_backend")
    @classmethod
    def validate_attention_backend(cls, value: str) -> str:
        normalized = (value or "sdpa").strip().lower().replace("-", "_")
        if normalized in {"sage", "sageattention"}:
            normalized = "sage_sdpa"
        if normalized not in {"sage_sdpa", "sdpa", "xformers", "none"}:
            raise ValueError("attention_backend must be sage_sdpa, sdpa, xformers, or none")
        return normalized

    @field_validator("vram_profile")
    @classmethod
    def validate_vram_profile(cls, value: str) -> str:
        return normalize_vram_profile(value)

    def effective_vram_profile(self) -> str:
        if self.cpu:
            return "cpu"
        if self.lowvram:
            return "low"
        if self.medvram:
            return "mid"
        if self.highvram:
            return "high"
        return normalize_vram_profile(self.vram_profile)

    @classmethod
    def from_runtime_flags(cls, flags: RuntimeFlags) -> LaunchSettings:
        return cls(
            listen=flags.listen,
            port=flags.port,
            autolaunch=flags.autolaunch,
            theme=flags.theme,
            gradio_auth=flags.gradio_auth or "",
            api_cors_origins=flags.api_cors_origins,
            api_rate_limit_per_minute=flags.api_rate_limit_per_minute,
            block_private_download_urls=flags.block_private_download_urls,
            gerror=flags.gerror,
            genlog=flags.genlog,
            share=flags.share,
            medvram=flags.medvram,
            lowvram=flags.lowvram,
            attention_backend=flags.attention_backend,
            xformers=flags.xformers,
            opt_sdp_attention=flags.opt_sdp_attention,
            opt_split_attention=flags.opt_split_attention,
            async_offload=flags.async_offload,
            pinned_memory=flags.pinned_memory,
            cuda_malloc=flags.cuda_malloc,
            no_half=flags.no_half,
            fp8=flags.fp8,
            fluxfp8=flags.fluxfp8,
            directml=flags.directml,
            cpu=flags.cpu,
            inference_backend=flags.inference_backend,
            onnx_provider=flags.onnx_provider,
            vram_profile=flags.effective_vram_profile(),
            highvram=flags.highvram,
            cuda_graphs=flags.cuda_graphs,
            torchao=flags.torchao,
            fp8_quant=flags.fp8_quant,
            torch_compile=flags.torch_compile,
            channels_last=flags.channels_last,
            nvenc=flags.nvenc,
            hevc=flags.hevc,
            api=flags.api,
            nowebui=flags.nowebui,
            models_dir=str(flags.models_dir) if flags.models_dir else "",
            ckpt_dir=str(flags.ckpt_dir) if flags.ckpt_dir else "",
            output_dir=str(flags.output_dir) if flags.output_dir else "",
            extra_model_dirs="\n".join(str(path) for path in flags.resolved_extra_model_dirs()),
            extra_ckpt_dirs="\n".join(str(path) for path in flags.resolved_extra_ckpt_dirs()),
            nvidia_vfx_sdk_root=str(flags.nvidia_vfx_sdk_root) if flags.nvidia_vfx_sdk_root else "",
            vsr_video_effects_app=str(flags.vsr_video_effects_app) if flags.vsr_video_effects_app else "",
            vsr_upscale_app=str(flags.vsr_upscale_app) if flags.vsr_upscale_app else "",
            videofx_denoise_app=str(flags.videofx_denoise_app) if flags.videofx_denoise_app else "",
            videofx_aigs_app=str(flags.videofx_aigs_app) if flags.videofx_aigs_app else "",
            videofx_relight_app=str(flags.videofx_relight_app) if flags.videofx_relight_app else "",
            vsr_model_dir=str(flags.vsr_model_dir) if flags.vsr_model_dir else "",
        )

    def to_runtime_flags(self, base: RuntimeFlags) -> RuntimeFlags:
        payload = base.model_dump()
        vram_profile = self.effective_vram_profile()
        payload.update(
            {
                "listen": self.listen,
                "port": self.port,
                "autolaunch": self.autolaunch,
                "theme": self.theme,
                "gradio_auth": self.gradio_auth or None,
                "api_cors_origins": self.api_cors_origins,
                "api_rate_limit_per_minute": self.api_rate_limit_per_minute,
                "block_private_download_urls": self.block_private_download_urls,
                "gerror": self.gerror,
                "genlog": self.genlog,
                "share": self.share,
                "attention_backend": self.attention_backend,
                "xformers": self.xformers,
                "opt_sdp_attention": self.opt_sdp_attention,
                "opt_split_attention": self.opt_split_attention,
                "async_offload": self.async_offload,
                "pinned_memory": self.pinned_memory,
                "cuda_malloc": self.cuda_malloc,
                "no_half": self.no_half,
                "fp8": self.fp8,
                "fluxfp8": self.fluxfp8,
                "directml": self.directml,
                "cpu": vram_profile == "cpu",
                "inference_backend": self.inference_backend,
                "onnx_provider": self.onnx_provider,
                "vram_profile": vram_profile,
                "medvram": vram_profile == "mid",
                "lowvram": vram_profile == "low",
                "highvram": vram_profile == "high",
                "cuda_graphs": self.cuda_graphs,
                "torchao": self.torchao,
                "fp8_quant": self.fp8_quant,
                "torch_compile": self.torch_compile,
                "channels_last": self.channels_last,
                "nvenc": self.nvenc,
                "hevc": self.hevc,
                "api": self.api,
                "nowebui": self.nowebui,
                "models_dir": Path(self.models_dir).resolve() if self.models_dir.strip() else None,
                "ckpt_dir": Path(self.ckpt_dir).resolve() if self.ckpt_dir.strip() else None,
                "output_dir": Path(self.output_dir).resolve() if self.output_dir.strip() else None,
                "extra_model_dirs": [
                    Path(line).resolve()
                    for line in self.extra_model_dirs.splitlines()
                    if line.strip()
                ],
                "extra_ckpt_dirs": [
                    Path(line).resolve()
                    for line in self.extra_ckpt_dirs.splitlines()
                    if line.strip()
                ],
                "nvidia_vfx_sdk_root": Path(self.nvidia_vfx_sdk_root).resolve() if self.nvidia_vfx_sdk_root.strip() else None,
                "vsr_video_effects_app": Path(self.vsr_video_effects_app).resolve() if self.vsr_video_effects_app.strip() else None,
                "vsr_upscale_app": Path(self.vsr_upscale_app).resolve() if self.vsr_upscale_app.strip() else None,
                "videofx_denoise_app": Path(self.videofx_denoise_app).resolve() if self.videofx_denoise_app.strip() else None,
                "videofx_aigs_app": Path(self.videofx_aigs_app).resolve() if self.videofx_aigs_app.strip() else None,
                "videofx_relight_app": Path(self.videofx_relight_app).resolve() if self.videofx_relight_app.strip() else None,
                "vsr_model_dir": Path(self.vsr_model_dir).resolve() if self.vsr_model_dir.strip() else None,
            }
        )
        return RuntimeFlags.model_validate(payload)

    def argv(self) -> list[str]:
        args: list[str] = []
        vram_profile = self.effective_vram_profile()
        if self.listen:
            args.append("--listen")
        if self.port != 7860:
            args.extend(["--port", str(self.port)])
        if self.autolaunch:
            args.append("--autolaunch")
        if self.theme != "dark":
            args.extend(["--theme", self.theme])
        if self.gradio_auth:
            args.extend(["--gradio-auth", self.gradio_auth])
        if self.api_cors_origins.strip():
            args.extend(["--api-cors-origins", self.api_cors_origins.strip()])
        if self.api_rate_limit_per_minute:
            args.extend(["--api-rate-limit-per-minute", str(self.api_rate_limit_per_minute)])
        if not self.block_private_download_urls:
            args.append("--allow-private-download-urls")
        if self.gerror:
            args.append("--gerror")
        if self.genlog:
            args.append("--genlog")
        if self.share:
            args.append("--share")
        if vram_profile != "normal":
            args.extend(["--vram-profile", vram_profile])
        if vram_profile == "mid":
            args.append("--medvram")
        if vram_profile == "low":
            args.append("--lowvram")
        if vram_profile == "high":
            args.append("--highvram")
        if self.attention_backend != "sdpa":
            args.extend(["--attention-backend", self.attention_backend])
        if self.xformers:
            args.append("--xformers")
        if self.opt_sdp_attention:
            args.append("--opt-sdp-attention")
        if self.opt_split_attention:
            args.append("--opt-split-attention")
        if not self.async_offload:
            args.append("--no-async-offload")
        if not self.pinned_memory:
            args.append("--no-pinned-memory")
        if self.cuda_malloc:
            args.append("--cuda-malloc")
        if self.no_half:
            args.append("--no-half")
        if self.fp8:
            args.append("--fp8")
        if self.fluxfp8:
            args.append("--fluxfp8")
        if self.directml:
            args.append("--directml")
        if vram_profile == "cpu":
            args.append("--cpu")
        if self.inference_backend != "diffusers":
            args.extend(["--inference-backend", self.inference_backend])
        if self.onnx_provider != "auto":
            args.extend(["--onnx-provider", self.onnx_provider])
        if self.cuda_graphs:
            args.append("--cuda-graphs")
        if self.torchao:
            args.append("--torchao")
        if self.fp8_quant:
            args.append("--fp8-quant")
        if self.torch_compile:
            args.append("--torch-compile")
        if self.channels_last:
            args.append("--channels-last")
        if self.nvenc:
            args.append("--nvenc")
        if self.hevc:
            args.append("--hevc")
        if self.api:
            args.append("--api")
        if self.nowebui:
            args.append("--nowebui")
        if self.models_dir.strip():
            args.extend(["--models-dir", self.models_dir.strip()])
        if self.ckpt_dir.strip():
            args.extend(["--ckpt-dir", self.ckpt_dir.strip()])
        if self.output_dir.strip():
            args.extend(["--output-dir", self.output_dir.strip()])
        for path in [line.strip() for line in self.extra_model_dirs.splitlines() if line.strip()]:
            args.extend(["--extra-model-dir", path])
        for path in [line.strip() for line in self.extra_ckpt_dirs.splitlines() if line.strip()]:
            args.extend(["--extra-ckpt-dir", path])
        if self.nvidia_vfx_sdk_root.strip():
            args.extend(["--nvidia-vfx-sdk-root", self.nvidia_vfx_sdk_root.strip()])
        if self.vsr_video_effects_app.strip():
            args.extend(["--vsr-video-effects-app", self.vsr_video_effects_app.strip()])
        if self.vsr_upscale_app.strip():
            args.extend(["--vsr-upscale-app", self.vsr_upscale_app.strip()])
        if self.videofx_denoise_app.strip():
            args.extend(["--videofx-denoise-app", self.videofx_denoise_app.strip()])
        if self.videofx_aigs_app.strip():
            args.extend(["--videofx-aigs-app", self.videofx_aigs_app.strip()])
        if self.videofx_relight_app.strip():
            args.extend(["--videofx-relight-app", self.videofx_relight_app.strip()])
        if self.vsr_model_dir.strip():
            args.extend(["--vsr-model-dir", self.vsr_model_dir.strip()])
        return args

    def command_preview(self) -> str:
        argv = self.argv()
        if not argv:
            return "python -m aiwf.app"
        return f"python -m aiwf.app {shlex.join(argv)}"


def launch_settings_path(data_dir: Path) -> Path:
    return data_dir / LAUNCH_FILENAME


def explicit_cli_flags(argv: list[str] | None = None) -> set[str]:
    flags: set[str] = set()
    for arg in argv or sys.argv[1:]:
        if arg.startswith("--"):
            flags.add(arg.split("=", 1)[0])
    return flags


def load_launch_settings(path: Path) -> LaunchSettings | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LaunchSettings.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def save_launch_settings(path: Path, settings: LaunchSettings) -> None:
    path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")


def merge_launch_settings(
    cli_flags: RuntimeFlags,
    saved: LaunchSettings | None,
    *,
    explicit: set[str] | None = None,
) -> RuntimeFlags:
    """Merge saved launch.json with current CLI/env flags.

    Saved values provide the normal app-start defaults. Any explicitly supplied
    CLI flag takes precedence so maintainers can force a temporary runtime
    boundary without editing the user's persisted launch profile.
    """

    if saved is None:
        return cli_flags

    explicit = explicit or explicit_cli_flags()
    merged = saved.to_runtime_flags(cli_flags)

    vram_flags = ("--vram-profile", "--cpu", "--lowvram", "--medvram", "--normalvram", "--highvram")
    field_to_flag = {
        "listen": "--listen",
        "port": "--port",
        "autolaunch": ("--autolaunch", "--no-autolaunch"),
        "theme": "--theme",
        "gradio_auth": "--gradio-auth",
        "api_cors_origins": "--api-cors-origins",
        "api_rate_limit_per_minute": "--api-rate-limit-per-minute",
        "block_private_download_urls": "--allow-private-download-urls",
        "gerror": "--gerror",
        "genlog": "--genlog",
        "share": "--share",
        "vram_profile": vram_flags,
        "medvram": vram_flags,
        "lowvram": vram_flags,
        "highvram": vram_flags,
        "attention_backend": "--attention-backend",
        "xformers": "--xformers",
        "opt_sdp_attention": "--opt-sdp-attention",
        "opt_split_attention": "--opt-split-attention",
        "async_offload": "--no-async-offload",
        "pinned_memory": "--no-pinned-memory",
        "cuda_malloc": ("--cuda-malloc", "--no-cuda-malloc"),
        "no_half": "--no-half",
        "fp8": "--fp8",
        "fluxfp8": "--fluxfp8",
        "directml": "--directml",
        "cpu": vram_flags,
        "inference_backend": "--inference-backend",
        "onnx_provider": "--onnx-provider",
        "cuda_graphs": "--cuda-graphs",
        "torchao": "--torchao",
        "fp8_quant": "--fp8-quant",
        "torch_compile": "--torch-compile",
        "channels_last": "--channels-last",
        "nvenc": "--nvenc",
        "hevc": "--hevc",
        "api": "--api",
        "nowebui": "--nowebui",
        "models_dir": "--models-dir",
        "ckpt_dir": "--ckpt-dir",
        "output_dir": "--output-dir",
        "extra_model_dirs": "--extra-model-dir",
        "extra_ckpt_dirs": "--extra-ckpt-dir",
        "nvidia_vfx_sdk_root": "--nvidia-vfx-sdk-root",
        "vsr_video_effects_app": "--vsr-video-effects-app",
        "vsr_upscale_app": "--vsr-upscale-app",
        "videofx_denoise_app": "--videofx-denoise-app",
        "videofx_aigs_app": "--videofx-aigs-app",
        "videofx_relight_app": "--videofx-relight-app",
        "vsr_model_dir": "--vsr-model-dir",
    }

    cli_dump = cli_flags.model_dump()
    merged_dump = merged.model_dump()

    for field, flag in field_to_flag.items():
        flags = flag if isinstance(flag, tuple) else (flag,)
        if any(item in explicit for item in flags):
            merged_dump[field] = cli_dump[field]

    return RuntimeFlags.model_validate(merged_dump)


def format_launch_status(current: LaunchSettings, saved: LaunchSettings | None) -> str:
    if saved is None:
        return "_No saved launch profile yet. Adjust options below and click **Save launch options**._"
    if current.model_dump() == saved.model_dump():
        return "**Saved profile matches this session.** Restart anytime to re-apply the same options."
    return (
        "**Saved profile differs from this session.** "
        "Command-line overrides may be in effect, or you need to restart after saving."
    )
