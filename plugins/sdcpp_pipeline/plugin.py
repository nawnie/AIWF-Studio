from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

DEFAULT_PROFILE: dict[str, Any] = {
    "sdCli": "",
    "backend": "cuda0",
    "paramsBackend": "",
    "maxVram": "0",
    "offloadToCpu": False,
    "streamLayers": False,
    "diffusionFlashAttention": True,
    "vaeTiling": False,
    "mmap": True,
    "preview": "vae",
    "clipL": "",
    "clipG": "",
    "clipVision": "",
    "t5xxl": "",
    "llm": "",
    "llmVision": "",
    "diffusionModel": "",
    "highNoiseDiffusionModel": "",
    "uncondDiffusionModel": "",
    "vae": "",
    "taesd": "",
    "controlNet": "",
    "loraModelDir": "",
    "tensorTypeRules": "",
    "modelArgs": "",
    "extraSampleArgs": "",
    "extraArgs": "",
    "videoFrames": 25,
    "fps": 16,
}


class SdcppProfileUpdate(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _profile_path() -> Path:
    return _repo_root() / "_local" / "sdcpp_profile.json"


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _load_profile() -> dict[str, Any]:
    profile = dict(DEFAULT_PROFILE)
    try:
        data = json.loads(_profile_path().read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if isinstance(data, dict):
        profile.update({key: value for key, value in data.items() if key in DEFAULT_PROFILE})
    return profile


def _split_asset_args(profile: dict[str, Any]) -> list[str]:
    mapping = [
        ("clipL", "--clip_l"),
        ("clipG", "--clip_g"),
        ("clipVision", "--clip_vision"),
        ("t5xxl", "--t5xxl"),
        ("llm", "--llm"),
        ("llmVision", "--llm_vision"),
        ("diffusionModel", "--diffusion-model"),
        ("highNoiseDiffusionModel", "--high-noise-diffusion-model"),
        ("uncondDiffusionModel", "--uncond-diffusion-model"),
        ("vae", "--vae"),
        ("taesd", "--taesd"),
        ("controlNet", "--control-net"),
        ("loraModelDir", "--lora-model-dir"),
        ("tensorTypeRules", "--tensor-type-rules"),
        ("modelArgs", "--model-args"),
        ("extraSampleArgs", "--extra-sample-args"),
    ]
    args: list[str] = []
    for key, flag in mapping:
        value = _normalize_text(profile.get(key))
        if value:
            args.extend([flag, value])
    extra = _normalize_text(profile.get("extraArgs"))
    if extra:
        args.extend(shlex.split(extra, posix=os.name != "nt"))
    return args


def _apply_profile_env(profile: dict[str, Any]) -> None:
    env_map = {
        "AIWF_SDCPP_BINARY": profile.get("sdCli"),
        "AIWF_SDCPP_BACKEND": profile.get("backend"),
        "AIWF_SDCPP_PARAMS_BACKEND": profile.get("paramsBackend"),
        "AIWF_SDCPP_MAX_VRAM": profile.get("maxVram"),
        "AIWF_SDCPP_PREVIEW": profile.get("preview"),
    }
    for key, value in env_map.items():
        text = _normalize_text(value)
        if text:
            os.environ[key] = text
        else:
            os.environ.pop(key, None)
    os.environ["AIWF_SDCPP_OFFLOAD_TO_CPU"] = "1" if profile.get("offloadToCpu") else "0"
    os.environ["AIWF_SDCPP_STREAM_LAYERS"] = "1" if profile.get("streamLayers") else "0"
    os.environ["AIWF_SDCPP_DIFFUSION_FA"] = "1" if profile.get("diffusionFlashAttention", True) else "0"
    os.environ["AIWF_SDCPP_VAE_TILING"] = "1" if profile.get("vaeTiling") else "0"
    os.environ["AIWF_SDCPP_MMAP"] = "1" if profile.get("mmap", True) else "0"
    os.environ["AIWF_SDCPP_EXTRA_ARGS"] = " ".join(shlex.quote(str(item)) for item in _split_asset_args(profile))


def _save_profile(values: dict[str, Any]) -> dict[str, Any]:
    profile = dict(DEFAULT_PROFILE)
    profile.update(_load_profile())
    profile.update({key: value for key, value in values.items() if key in DEFAULT_PROFILE})
    _profile_path().parent.mkdir(parents=True, exist_ok=True)
    _profile_path().write_text(json.dumps(profile, indent=2), encoding="utf-8")
    _apply_profile_env(profile)
    return profile


def _field(key: str, label: str, value: Any, *, kind: str = "text") -> str:
    if kind == "checkbox":
        checked = "checked" if value else ""
        return f"<label class='toggle'><input type='checkbox' id='{key}' {checked}> <span>{label}</span></label>"
    safe_value = str(value or "").replace("&", "&amp;").replace("\"", "&quot;")
    return f"<label><span>{label}</span><input id='{key}' value=\"{safe_value}\"></label>"


def _html(profile: dict[str, Any]) -> str:
    fields = "".join(
        [
            _field("sdCli", "sd-cli path", profile["sdCli"]),
            _field("backend", "runtime backend", profile["backend"]),
            _field("paramsBackend", "params backend", profile["paramsBackend"]),
            _field("maxVram", "max VRAM GiB", profile["maxVram"]),
            _field("clipL", "CLIP-L", profile["clipL"]),
            _field("clipG", "CLIP-G", profile["clipG"]),
            _field("clipVision", "CLIP Vision", profile["clipVision"]),
            _field("t5xxl", "T5 XXL", profile["t5xxl"]),
            _field("llm", "LLM text encoder", profile["llm"]),
            _field("llmVision", "LLM vision encoder", profile["llmVision"]),
            _field("diffusionModel", "diffusion model", profile["diffusionModel"]),
            _field("highNoiseDiffusionModel", "high-noise diffusion model", profile["highNoiseDiffusionModel"]),
            _field("uncondDiffusionModel", "unconditional diffusion model", profile["uncondDiffusionModel"]),
            _field("vae", "VAE", profile["vae"]),
            _field("taesd", "TAESD", profile["taesd"]),
            _field("controlNet", "ControlNet", profile["controlNet"]),
            _field("loraModelDir", "LoRA model directory", profile["loraModelDir"]),
            _field("tensorTypeRules", "tensor type rules", profile["tensorTypeRules"]),
            _field("modelArgs", "model args", profile["modelArgs"]),
            _field("extraSampleArgs", "extra sample args", profile["extraSampleArgs"]),
            _field("extraArgs", "raw extra args", profile["extraArgs"]),
            _field("videoFrames", "video frames", profile["videoFrames"]),
            _field("fps", "video FPS", profile["fps"]),
            _field("offloadToCpu", "offload params to CPU", profile["offloadToCpu"], kind="checkbox"),
            _field("streamLayers", "stream layers", profile["streamLayers"], kind="checkbox"),
            _field("diffusionFlashAttention", "diffusion flash attention", profile["diffusionFlashAttention"], kind="checkbox"),
            _field("vaeTiling", "VAE tiling", profile["vaeTiling"], kind="checkbox"),
            _field("mmap", "memory map model", profile["mmap"], kind="checkbox"),
        ]
    )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>AIWF SDCPP Pipeline</title>
  <style>
    body {{ margin: 0; background: #191714; color: #f4ead9; font: 14px system-ui, sans-serif; }}
    main {{ max-width: 1100px; margin: 36px auto; padding: 24px; background: #211f1a; border: 1px solid #5c4a2f; border-radius: 18px; }}
    h1 {{ margin-top: 0; color: #f0b35a; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; }}
    label {{ display: grid; gap: 6px; }}
    input {{ background: #11100e; border: 1px solid #6b5636; color: #f4ead9; border-radius: 10px; padding: 9px; }}
    .toggle {{ display: flex; align-items: center; gap: 8px; background: #181613; border: 1px solid #4a3c28; padding: 10px; border-radius: 10px; }}
    button {{ margin-top: 16px; border: 1px solid #8b6a35; border-radius: 12px; background: #d8923b; color: #140f08; padding: 12px 16px; font-weight: 800; cursor: pointer; }}
    pre {{ white-space: pre-wrap; background: #11100e; border: 1px solid #3c3324; border-radius: 10px; padding: 12px; }}
  </style>
</head>
<body>
<main>
  <h1>SDCPP pipeline profile</h1>
  <p>These settings feed the sd.cpp backend through AIWF_SDCPP_* environment values and raw sd-cli flags. Restart into the sdcpp profile after saving.</p>
  <div class='grid'>{fields}</div>
  <button id='save'>Save profile</button>
  <pre id='status'>Ready.</pre>
</main>
<script>
const keys = {json.dumps(list(DEFAULT_PROFILE.keys()))};
function readValue(key) {{
  const el = document.getElementById(key);
  if (!el) return '';
  if (el.type === 'checkbox') return el.checked;
  if (key === 'videoFrames' || key === 'fps') return Number(el.value || 0);
  return el.value;
}}
document.getElementById('save').addEventListener('click', async () => {{
  const values = Object.fromEntries(keys.map((key) => [key, readValue(key)]));
  const res = await fetch('/api/ext/sdcpp-pipeline/profile', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ values }}) }});
  const payload = await res.json();
  document.getElementById('status').textContent = JSON.stringify(payload, null, 2);
}});
</script>
</body>
</html>
"""


def setup(ctx: Any) -> None:
    router = APIRouter()
    _apply_profile_env(_load_profile())

    @router.get("/status")
    def status() -> dict[str, Any]:
        profile = _load_profile()
        return {
            "profile": profile,
            "extraArgs": _split_asset_args(profile),
            "note": "Restart into the sdcpp backend profile after saving split-asset paths.",
        }

    @router.get("/ui", response_class=HTMLResponse)
    def ui() -> str:
        return _html(_load_profile())

    @router.post("/profile")
    def update_profile(payload: SdcppProfileUpdate) -> dict[str, Any]:
        profile = _save_profile(payload.values)
        return {
            "status": "saved",
            "profile": profile,
            "extraArgs": _split_asset_args(profile),
        }

    ctx.plugins.register_api("sdcpp-pipeline", router)
