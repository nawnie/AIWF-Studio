"""Persisted settings for the Qwen Image 2.1 Studio (JSON next to the engine, git-ignored)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ENGINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ENGINE_DIR.parents[1]
SETTINGS_FILE = ENGINE_DIR / "qwen21.settings.json"


def _default_output_dir() -> str:
    return str(REPO_ROOT / "outputs" / "qwen_image_2_1")


@dataclass
class Settings:
    comfy_url: str = "http://127.0.0.1:8188"
    # ComfyUI install to launch from the app. Leave empty to only connect.
    comfy_launch_command: str = ""  # e.g. F:\ComfyUI\venv\Scripts\python.exe main.py --listen 127.0.0.1 --port 8188
    comfy_launch_cwd: str = ""  # e.g. F:\ComfyUI
    comfy_models_dir: str = ""  # optional, lets the app check model files on disk
    output_dir: str = field(default_factory=_default_output_dir)
    # trainers
    ai_toolkit_dir: str = str(ENGINE_DIR / "ai-toolkit")
    ai_toolkit_python: str = ""  # defaults to <ai_toolkit_dir>/venv/Scripts/python.exe
    diffsynth_dir: str = str(ENGINE_DIR / "DiffSynth-Studio")
    diffsynth_python: str = ""
    training_output_dir: str = str(REPO_ROOT / "outputs" / "training" / "qwen_image_2_1")
    # Pro-tab status bridge (frontend QwenImageEditorLayout polls 127.0.0.1:7865/api/health)
    web_status_enabled: bool = True
    web_status_port: int = 7865
    # last-used generation defaults
    dit: str = "qwen_image_2.1_int8_convrot.safetensors"
    text_encoder: str = "qwen3vl_8b_int8_convrot.safetensors"
    vae: str = "qwen_image_2.1_vae_bf16.safetensors"
    pe_t2i: str = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
    pe_i2i: str = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
    controlnet_patch: str = "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"

    def ai_toolkit_python_exe(self) -> Path:
        if self.ai_toolkit_python:
            return Path(self.ai_toolkit_python)
        return venv_python(Path(self.ai_toolkit_dir) / "venv")

    def diffsynth_python_exe(self) -> Path:
        if self.diffsynth_python:
            return Path(self.diffsynth_python)
        return venv_python(Path(self.diffsynth_dir) / "venv")

    def comfy_launch_env(self) -> dict[str, str]:
        """UTF-8 environment for launching ComfyUI (Windows console encodings otherwise break emoji/CJK logs)."""
        env = dict(os.environ)
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def save(self, path: Path = SETTINGS_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        if not path.is_file():
            return cls()
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"
