from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

VALID_BACKENDS = {
    "diffusers": "Diffusers",
    "sdcpp": "stable-diffusion.cpp",
    "onnx": "ONNX",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_backend(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "diffusers": "diffusers",
        "onnx": "onnx",
        "sdcpp": "sdcpp",
        "sd-cpp": "sdcpp",
        "stable-diffusion.cpp": "sdcpp",
        "stable-diffusion-cpp": "sdcpp",
    }
    return aliases.get(normalized, "")


def _profile_path() -> Path:
    return _repo_root() / "_local" / "backend_profile.json"


def _write_profile(backend: str) -> None:
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"backend": backend}, indent=2), encoding="utf-8")


def _read_profile() -> str:
    try:
        data = json.loads(_profile_path().read_text(encoding="utf-8"))
    except Exception:
        return "diffusers"
    backend = _normalize_backend(str(data.get("backend") or "")) if isinstance(data, dict) else ""
    return backend if backend in VALID_BACKENDS else "diffusers"


def _current_backend(ctx: Any) -> str:
    flags = getattr(ctx, "flags", None)
    backend = _normalize_backend(str(getattr(flags, "inference_backend", "") or ""))
    return backend if backend in VALID_BACKENDS else "diffusers"


def _html(current: str, saved: str) -> str:
    buttons = "".join(
        f"<button data-backend='{backend}' class='backend-button {'active' if backend == saved else ''}'>{label}</button>"
        for backend, label in VALID_BACKENDS.items()
    )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>AIWF Backend Switch</title>
  <style>
    body {{ margin: 0; background: #161310; color: #f4ead6; font: 15px system-ui, sans-serif; }}
    main {{ max-width: 760px; margin: 48px auto; padding: 24px; background: #211b16; border: 1px solid #57432f; border-radius: 18px; }}
    h1 {{ margin-top: 0; color: #d99a32; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
    button {{ border: 1px solid #72583b; border-radius: 12px; background: #2b241d; color: #f4ead6; padding: 12px 16px; cursor: pointer; }}
    button.active {{ background: #d99a32; color: #161310; font-weight: 700; }}
    .status {{ padding: 12px; border-radius: 12px; background: #120f0d; border: 1px solid #57432f; }}
    code {{ color: #f0b04f; }}
  </style>
</head>
<body>
  <main>
    <h1>AIWF backend profile</h1>
    <p>Current runtime: <code>{current}</code></p>
    <p>Saved profile launcher default: <code>{saved}</code></p>
    <div class='row'>{buttons}</div>
    <p class='status' id='status'>Pick a backend, then restart with <code>scripts/launch_backend_profile.ps1</code>. Logs first. Tiny fireworks later.</p>
  </main>
  <script>
    document.querySelectorAll('.backend-button').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const backend = button.dataset.backend;
        const status = document.getElementById('status');
        status.textContent = `Saving backend profile ${{backend}}...`;
        const res = await fetch(`/api/ext/backend-switch/profile/${{backend}}`, {{ method: 'POST' }});
        const payload = await res.json();
        status.textContent = payload.message || JSON.stringify(payload);
      }});
    }});
  </script>
</body>
</html>
"""


def setup(ctx: Any) -> None:
    router = APIRouter()

    @router.get("/status")
    def status() -> dict[str, Any]:
        generation = getattr(ctx, "generation", None)
        active_backend = getattr(generation, "backend", None)
        return {
            "current": _current_backend(ctx),
            "saved": _read_profile(),
            "available": [{"id": key, "label": label} for key, label in VALID_BACKENDS.items()],
            "activeBackendClass": active_backend.__class__.__name__ if active_backend is not None else "",
            "message": "Changing backend requires launching a new Pro profile because the engine object is built at boot.",
        }

    @router.get("/ui", response_class=HTMLResponse)
    def ui() -> str:
        return _html(_current_backend(ctx), _read_profile())

    @router.post("/profile/{backend}")
    def save_profile(backend: str) -> dict[str, Any]:
        normalized = _normalize_backend(backend)
        if normalized not in VALID_BACKENDS:
            raise HTTPException(status_code=422, detail="Backend must be diffusers, sdcpp, or onnx.")
        _write_profile(normalized)
        return {
            "status": "saved",
            "backend": normalized,
            "message": f"Saved {VALID_BACKENDS[normalized]} as the profile launcher default. Restart with scripts/launch_backend_profile.ps1.",
        }

    ctx.plugins.register_api("backend-switch", router)
