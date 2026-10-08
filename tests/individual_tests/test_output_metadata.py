"""Torch-free output listing for the engine API (aiwf/services/output_metadata.py).

The unified routes /outputs and catalog-outputs use these helpers instead of the private ones in
aiwf/web/pro_api.py, so the standalone engine API never imports torch. The first tests compare
both implementations on the same files, so a change in Pro's helpers that the copy does not follow
fails here instead of drifting. The last test runs the real router in a clean interpreter and
checks that listing outputs leaves torch unloaded.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from aiwf.services import output_metadata
from aiwf.web import pro_api

REPO = Path(__file__).resolve().parents[2]

INFOTEXTS = [
    "a lighthouse at dusk\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 42, Size: 512x768, Model: sdxl-base",
    "only a prompt",
    "red fox in snow\nSteps: 30, Sampler: euler, CFG scale: 1, Seed: 0, Size: 800x800, Model: Qwen Image 2.1 (int8)",
    "upscaled\nSteps: 12, Seed: 9, Size: 512x512, Hires resize: 1024x1024",
    "bad numbers\nSteps: many, CFG scale: high, Seed: -5, Size: axb",
    "",
]


def _png(path: Path, parameters: str | None) -> Path:
    info = PngImagePlugin.PngInfo()
    if parameters is not None:
        info.add_text("parameters", parameters)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, pnginfo=info)
    return path


@pytest.mark.parametrize("text", INFOTEXTS)
def test_settings_match_pro(text: str) -> None:
    assert output_metadata.settings_from_infotext(text) == pro_api._settings_from_infotext(text)


def test_infotext_reading_matches_pro(tmp_path: Path) -> None:
    with_text = _png(tmp_path / "with.png", INFOTEXTS[0])
    without = _png(tmp_path / "without.png", None)
    sidecar_image = _png(tmp_path / "sidecar.png", None)
    (tmp_path / "sidecar.txt").write_text("from the sidecar\nSteps: 5, Seed: 3", encoding="utf-8")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    for path in (with_text, without, sidecar_image, broken):
        assert output_metadata.read_output_infotext(path) == pro_api._read_output_infotext(path)
    assert output_metadata.read_output_infotext(with_text, fallback="given") == pro_api._read_output_infotext(with_text, fallback="given")


def test_recent_listing_matches_pro(tmp_path: Path) -> None:
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    (tmp_path / ".hidden").mkdir()
    files = [
        _png(tmp_path / "a.png", INFOTEXTS[0]),
        _png(tmp_path / "nested" / "b.png", INFOTEXTS[1]),
        _png(tmp_path / "nested" / "deeper" / "c.png", INFOTEXTS[2]),
        _png(tmp_path / ".hidden" / "skipped.png", INFOTEXTS[3]),
    ]
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    # distinct modification times so the newest-first order is well defined
    for offset, path in enumerate(files):
        os.utime(path, (time.time() + offset, time.time() + offset))
    for limit in (1, 2, 10):
        ours = output_metadata.recent_paths_from_disk(tmp_path, limit=limit)
        assert ours == pro_api._recent_paths_from_disk(tmp_path, limit=limit)
    assert all(".hidden" not in path.parts for path in output_metadata.recent_paths_from_disk(tmp_path, limit=10))
    assert output_metadata.recent_paths_from_disk(tmp_path / "missing", limit=5) == []


def test_listing_outputs_does_not_load_torch(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _png(outputs / "shot.png", INFOTEXTS[2])
    script = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from aiwf.web.unified_api import build_unified_router

        out = Path({str(outputs)!r})
        ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=out.parent, resolved_output_dir=lambda: out))
        app = FastAPI()
        app.include_router(build_unified_router(ctx))
        body = TestClient(app, client=("127.0.0.1", 5000)).get("/api/pro/unified/outputs").json()
        assert body["outputs"][0]["relative_path"] == "shot.png", body
        assert body["outputs"][0]["seed"] == 0 and body["outputs"][0]["width"] == 800, body
        print("TORCH_LOADED=" + str("torch" in sys.modules))
        print("PRO_API_LOADED=" + str("aiwf.web.pro_api" in sys.modules))
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "TORCH_LOADED=False" in result.stdout and "PRO_API_LOADED=False" in result.stdout, result.stdout
