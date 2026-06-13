from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiwf.core.config.settings import RuntimeFlags

LAUNCH_FILENAME = "launch.json"


class LaunchSettings(BaseSettings):
    """User-editable options applied on the next app start."""

    model_config = SettingsConfigDict(extra="ignore")

    listen: bool = False
    port: int = Field(default=7860, ge=1024, le=65535)
    autolaunch: bool = False
    theme: str = "dark"
    gradio_auth: str = ""
    api_cors_origins: str = ""
    api_rate_limit_per_minute: int = Field(default=0, ge=0, le=6000)
    block_private_download_urls: bool = True
    share: bool = False
    medvram: bool = False
    lowvram: bool = False
    xformers: bool = False
    opt_sdp_attention: bool = False
    opt_split_attention: bool = False
    no_half: bool = False
    fp8: bool = False
    directml: bool = False
    cpu: bool = False
    api: bool = False
    nowebui: bool = False
    models_dir: str = ""
    ckpt_dir: str = ""
    output_dir: str = ""
    extra_model_dirs: str = ""
    extra_ckpt_dirs: str = ""

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
            share=flags.share,
            medvram=flags.medvram,
            lowvram=flags.lowvram,
            xformers=flags.xformers,
            opt_sdp_attention=flags.opt_sdp_attention,
            opt_split_attention=flags.opt_split_attention,
            no_half=flags.no_half,
            fp8=flags.fp8,
            directml=flags.directml,
            cpu=flags.cpu,
            api=flags.api,
            nowebui=flags.nowebui,
            models_dir=str(flags.models_dir) if flags.models_dir else "",
            ckpt_dir=str(flags.ckpt_dir) if flags.ckpt_dir else "",
            output_dir=str(flags.output_dir) if flags.output_dir else "",
            extra_model_dirs="\n".join(str(path) for path in flags.resolved_extra_model_dirs()),
            extra_ckpt_dirs="\n".join(str(path) for path in flags.resolved_extra_ckpt_dirs()),
        )

    def to_runtime_flags(self, base: RuntimeFlags) -> RuntimeFlags:
        payload = base.model_dump()
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
                "share": self.share,
                "medvram": self.medvram,
                "lowvram": self.lowvram,
                "xformers": self.xformers,
                "opt_sdp_attention": self.opt_sdp_attention,
                "opt_split_attention": self.opt_split_attention,
                "no_half": self.no_half,
                "fp8": self.fp8,
                "directml": self.directml,
                "cpu": self.cpu,
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
            }
        )
        return RuntimeFlags.model_validate(payload)

    def argv(self) -> list[str]:
        args: list[str] = []
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
        if self.share:
            args.append("--share")
        if self.medvram:
            args.append("--medvram")
        if self.lowvram:
            args.append("--lowvram")
        if self.xformers:
            args.append("--xformers")
        if self.opt_sdp_attention:
            args.append("--opt-sdp-attention")
        if self.opt_split_attention:
            args.append("--opt-split-attention")
        if self.no_half:
            args.append("--no-half")
        if self.fp8:
            args.append("--fp8")
        if self.directml:
            args.append("--directml")
        if self.cpu:
            args.append("--cpu")
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
    if saved is None:
        return cli_flags

    explicit = explicit or explicit_cli_flags()
    merged = saved.to_runtime_flags(cli_flags)

    field_to_flag = {
        "listen": "--listen",
        "port": "--port",
        "autolaunch": "--autolaunch",
        "theme": "--theme",
        "gradio_auth": "--gradio-auth",
        "api_cors_origins": "--api-cors-origins",
        "api_rate_limit_per_minute": "--api-rate-limit-per-minute",
        "block_private_download_urls": "--allow-private-download-urls",
        "share": "--share",
        "medvram": "--medvram",
        "lowvram": "--lowvram",
        "xformers": "--xformers",
        "opt_sdp_attention": "--opt-sdp-attention",
        "opt_split_attention": "--opt-split-attention",
        "no_half": "--no-half",
        "cpu": "--cpu",
        "api": "--api",
        "nowebui": "--nowebui",
        "models_dir": "--models-dir",
        "ckpt_dir": "--ckpt-dir",
        "output_dir": "--output-dir",
        "extra_model_dirs": "--extra-model-dir",
        "extra_ckpt_dirs": "--extra-ckpt-dir",
    }

    cli_dump = cli_flags.model_dump()
    merged_dump = merged.model_dump()

    for field, flag in field_to_flag.items():
        if flag in explicit:
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
