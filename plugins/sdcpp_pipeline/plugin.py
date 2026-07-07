from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
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


class SmokeRequest(BaseModel):
    model: str = ""
    prompt: str = "AIWF stable-diffusion.cpp smoke test, sharp details"
    negativePrompt: str = "blurry, low quality"
    width: int = 512
    height: int = 512
    steps: int = 6
    cfgScale: float = 7.0
    seed: int = 42
    outputName: str = "sdcpp-smoke"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _profile_path() -> Path:
    return _repo_root() / "_local" / "sdcpp_profile.json"


def _output_root(ctx: Any) -> Path:
    flags = getattr(ctx, "flags", None)
    if flags is not None and hasattr(flags, "resolved_output_dir"):
        return Path(flags.resolved_output_dir())
    return _repo_root() / "outputs"


def _normalize_path_arg(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("\\", "/")


def _load_profile() -> dict[str, Any]:
    profile = dict(DEFAULT_PROFILE)
    try:
        data = json.loads(_profile_path().read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if isinstance(data, dict):
        profile.update(data)
    return profile


def _save_profile(profile: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_PROFILE)
    merged.update({key: value for key, value in profile.items() if key in DEFAULT_PROFILE})
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _apply_profile_env(merged)
    return merged


def _extra_args_from_profile(profile: dict[str, Any]) -> list[str]:
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
        value = _normalize_path_arg(profile.get(key))
        if value:
            args.extend([flag, value])
    extra = str(profile.get("extraArgs") or "").strip()
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
        text = str(value or "").strip()
        if text:
            os.environ[key] = text
        else:
            os.environ.pop(key, None)
    os.environ["AIWF_SDCPP_OFFLOAD_TO_CPU"] = "1" if profile.get("offloadToCpu") else "0"
    os.environ["AIWF_SDCPP_STREAM_LAYERS"] = "1" if profile.get("streamLayers") else "0"
    os.environ["AIWF_SDCPP_DIFFUSION_FA"] = "1" if profile.get("diffusionFlashAttention", True) else "0"
    os.environ["AIWF_SDCPP_VAE_TILING"] = "1" if profile.get("vaeTiling") else "0"
    os.environ["AIWF_SDCPP_MMAP"] = "1" if profile.get("mmap", True) else "0"
    extra_args = _extra_args_from_profile(profile)
    os.environ["AIWF_SDCPP_EXTRA_ARGS"] = " ".join(shlex.quote(str(item)) for item in extra_args)


def _find_sd_cli(profile: dict[str, Any]) -> str:
    explicit = str(profile.get("sdCli") or os.environ.get("AIWF_SDCPP_BINARY") or "").strip()
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve())
    root = _repo_root()
    candidates = [
        root / "tools" / "stable-diffusion.cpp" / "bin" / "sd-cli.exe",
        root / "tools" / "stable-diffusion.cpp" / "build" / "bin" / "Release" / "sd-cli.exe",
        root / "tools" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli.exe",
    ]
    for item in candidates:
        if item.is_file():
            return str(item.resolve())
    raise HTTPException(status_code=404, detail="sd-cli was not found. Set sdCli or run scripts/install_sdcpp.ps1.")


def _run_smoke(ctx: Any, request: SmokeRequest, *, video: bool = False) -> dict[str, Any]:
    profile = _load_profile()
    _apply_profile_env(profile)
    sd_cli = _find_sd_cli(profile)
    model = request.model or str(profile.get("diffusionModel") or "")
    if not model:
        raise HTTPException(status_code=422, detail="A smoke test model path is required.")
    out_dir = _output_root(ctx) / "sdcpp-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "mp4" if video else "png"
    output = out_dir / f"{request.outputName}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.{suffix}"
    cmd = [
        sd_cli,
        "-m",
        model,
        "-p",
        request.prompt,
        "-n",
        request.negativePrompt,
        "-o",
        str(output),
        "-W",
        str(request.width),
        "-H",
        str(request.height),
        "--steps",
        str(request.steps),
        "--cfg-scale",
        str(request.cfgScale),
        "-s",
        str(request.seed),
        "--backend",
        str(profile.get("backend") or "cuda0"),
    ]
    if video:
        cmd.extend(["-M", "vid_gen", "--video-frames", str(int(profile.get("videoFrames") or 25)), "--fps", str(int(profile.get("fps") or 16))])
    extra = _extra_args_from_profile(profile)
    if extra:
        cmd.extend(extra)
    try:
        completed = subprocess.run(cmd, cwd=str(_repo_root()), text=True, capture_output=True, timeout=60 * 30)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"sd-cli smoke test timed out: {exc}") from exc
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returnCode": completed.returncode,
        "command": cmd,
        "output": str(output),
        "stdout": completed.stdout[-6000:],
        "stderr": completed.stderr[-6000:],
    }


def _html(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile)
    fields = [
        ("sdCli", "sd-cli path"),
        ("backend", "Runtime backend"),
        ("paramsBackend", "Params backend"),
        ("maxVram", "Max VRAM GiB"),
        ("clipL", "CLIP-L"),
        ("clipG", "CLIP-G"),
        ("clipVision", "CLIP vision"),
        ("t5xxl", "T5XXL"),
        ("llm", "LLM encoder"),
        ("llmVision", "LLM vision"),
        ("diffusionModel", "Diffusion model"),
        ("highNoiseDiffusionModel", "High-noise diffusion model"),
        ("uncondDiffusionModel", "Unconditional diffusion model"),
        ("vae", "VAE"),
        ("taesd", "TAESD"),
        ("controlNet", "ControlNet"),
        ("loraModelDir", "LoRA model directory"),
        ("tensorTypeRules", "Tensor type rules"),
        ("modelArgs", "Model args"),
        ("extraSampleArgs", "Extra sample args"),
        ("extraArgs", "Raw extra args"),
    ]
    rows = "".join(
        f"<label><span>{label}</span><input data-key='{key}' value='{str(profile.get(key, ''))}'></label>"
        for key, label in fields
    )
    checks = "".join(
        f"<label class='check'><input type='checkbox' data-key='{key}' {'checked' if profile.get(key) else ''}><span>{label}</span></label>"
        for key, label in [
            ("offloadToCpu", "CPU offload"),
            ("streamLayers", "Stream layers"),
            ("diffusionFlashAttention", "Diffusion flash attention"),
            ("vaeTiling", "VAE tiling"),
            ("mmap", "Memory map"),
        ]
    )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>AIWF sd.cpp pipeline QA</title>
  <style>
    :root {{ --bg:#15110e; --panel:#211a14; --panel2:#2b2118; --line:#60492f; --text:#f5ebdc; --muted:#bca98e; --hot:#d99a32; --hot2:#f0b35a; --bad:#d66b5d; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px system-ui, sans-serif; }}
    main {{ max-width:1180px; margin:28px auto; padding:22px; }}
    h1 {{ color:var(--hot); margin:0 0 8px; }}
    .grid {{ display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap:12px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 12px 34px rgba(0,0,0,.25); }}
    label {{ display:flex; flex-direction:column; gap:6px; color:var(--muted); }}
    input, textarea, select {{ background:var(--panel2); border:1px solid var(--line); border-radius:10px; color:var(--text); padding:10px; }}
    .check {{ flex-direction:row; align-items:center; color:var(--text); }}
    button {{ background:var(--hot); color:#17110b; border:0; border-radius:12px; padding:11px 14px; font-weight:700; cursor:pointer; }}
    button.secondary {{ background:var(--panel2); color:var(--text); border:1px solid var(--line); }}
    pre {{ white-space:pre-wrap; background:#0f0c0a; border:1px solid var(--line); border-radius:14px; padding:14px; max-height:360px; overflow:auto; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }}
    .note {{ color:var(--muted); }}
  </style>
</head>
<body>
<main>
  <h1>stable-diffusion.cpp pipeline QA</h1>
  <p class='note'>Profile settings feed the sd.cpp backend through environment arguments. Restart only when changing the backend profile; most sd.cpp args apply to the next job.</p>
  <section class='panel'>
    <div class='grid'>{rows}</div>
    <div class='grid' style='margin-top:12px'>{checks}</div>
    <div class='actions'>
      <button id='save'>Save profile</button>
      <button class='secondary' id='smoke'>Run image smoke</button>
      <button class='secondary' id='video'>Run video smoke</button>
      <a class='secondary' href='/api/ext/sdcpp-pipeline/requirements' style='padding:11px 14px;border-radius:12px;text-decoration:none;color:var(--text);border:1px solid var(--line)'>Requirements JSON</a>
    </div>
  </section>
  <pre id='out'>{payload}</pre>
</main>
<script>
function collect(){{
  const values={{}};
  document.querySelectorAll('input[data-key]').forEach((el)=>{{
    if(el.type==='checkbox') values[el.dataset.key]=el.checked;
    else values[el.dataset.key]=el.value;
  }});
  return values;
}}
async function post(url, body) {{
  const res = await fetch(url, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body)}});
  const payload = await res.json();
  document.getElementById('out').textContent = JSON.stringify(payload,null,2);
}}
document.getElementById('save').onclick=()=>post('/api/ext/sdcpp-pipeline/profile', {{values: collect()}});
document.getElementById('smoke').onclick=()=>post('/api/ext/sdcpp-pipeline/smoke/image', {{model: collect().diffusionModel || '', outputName:'sdcpp-image-smoke'}});
document.getElementById('video').onclick=()=>post('/api/ext/sdcpp-pipeline/smoke/video', {{model: collect().diffusionModel || '', outputName:'sdcpp-video-smoke'}});
</script>
</body>
</html>
"""


def setup(ctx: Any) -> None:
    _apply_profile_env(_load_profile())
    router = APIRouter()

    @router.get("/status")
    def status() -> dict[str, Any]:
        profile = _load_profile()
        _apply_profile_env(profile)
        return {
            "profile": profile,
            "extraArgs": _extra_args_from_profile(profile),
            "env": {key: os.environ.get(key, "") for key in sorted(os.environ) if key.startswith("AIWF_SDCPP_")},
        }

    @router.get("/profile")
    def get_profile() -> dict[str, Any]:
        return {"profile": _load_profile()}

    @router.post("/profile")
    def save_profile(payload: SdcppProfileUpdate) -> dict[str, Any]:
        profile = _save_profile(payload.values)
        return {"status": "saved", "profile": profile, "extraArgs": _extra_args_from_profile(profile)}

    @router.get("/requirements")
    def requirements() -> dict[str, Any]:
        return {
            "routes": [
                {"route": "sd15/sdxl single-file", "backend": "sdcpp", "status": "qa-ready"},
                {"route": "img2img/inpaint", "backend": "sdcpp", "status": "qa-ready"},
                {"route": "lora", "backend": "sdcpp", "status": "directory-wired"},
                {"route": "controlnet", "backend": "sdcpp", "status": "path-and-image-wired"},
                {"route": "flux/qwen split assets", "backend": "sdcpp", "status": "argument-mapped"},
                {"route": "video", "backend": "sdcpp", "status": "experimental-smoke"},
            ],
            "fallback": "Use Diffusers/AIWF native pipelines when sd.cpp lacks a model family route, split asset mapping is incomplete, or video output does not match expected format.",
        }

    @router.get("/ui", response_class=HTMLResponse)
    def ui() -> str:
        return _html(_load_profile())

    @router.post("/smoke/image")
    def smoke_image(payload: SmokeRequest) -> dict[str, Any]:
        return _run_smoke(ctx, payload, video=False)

    @router.post("/smoke/video")
    def smoke_video(payload: SmokeRequest) -> dict[str, Any]:
        return _run_smoke(ctx, payload, video=True)

    ctx.plugins.register_api("sdcpp-pipeline", router)
