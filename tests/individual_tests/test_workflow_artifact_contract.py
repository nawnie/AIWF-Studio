from __future__ import annotations

import io
import struct
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import aiwf
import aiwf.services
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = str(_REPO_ROOT / "aiwf")
if _PACKAGE_ROOT not in aiwf.__path__:
    aiwf.__path__.insert(0, _PACKAGE_ROOT)
_SERVICE_ROOT = str(Path(_PACKAGE_ROOT) / "services")
if _SERVICE_ROOT not in aiwf.services.__path__:
    aiwf.services.__path__.insert(0, _SERVICE_ROOT)

from aiwf.core.domain.workflow import WorkflowRunResult, WorkflowStepResult, WorkflowStepType
from aiwf.services.image_artifacts import image_artifact_dimensions
from aiwf.web import pro_api


def _image_bytes(format_name: str, size: tuple[int, int] = (13, 11)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (25, 50, 75)).save(stream, format=format_name)
    return stream.getvalue()


def _record(path: Path) -> dict:
    return {
        "run_id": "a" * 32,
        "workflow": {"name": "artifact fixture"},
        "status": "completed",
        "steps": [
            {
                "step_id": "generate",
                "type": "txt2img",
                "label": "generate",
                "status": "completed",
                "params_sha256": "b" * 64,
                "started_at": "2026-10-04T00:00:00+00:00",
                "completed_at": "2026-10-04T00:00:01+00:00",
                "receipt": {"message": "fake", "image_path": str(path)},
            }
        ],
        "output_path": str(path),
    }


def _chunk(kind: bytes, content: bytes) -> bytes:
    return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF)


def _huge_header_png(width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)


def _invalid_idat_png() -> bytes:
    header = struct.pack(">IIBBBBB", 13, 11, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", b"invalid compressed pixel data")
        + _chunk(b"IEND", b"")
    )


def test_output_contract_rejects_corrupt_truncated_wrongformat_and_huge_images(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    valid = _image_bytes("PNG")
    variants = {
        "corrupt.png": b"not a png",
        "truncated.png": valid[:-12],
        "truncated.jpg": _image_bytes("JPEG")[:-32],
        "invalid-idat.png": _invalid_idat_png(),
        "wrong-format.png": _image_bytes("JPEG"),
        "too-large.png": _huge_header_png(9000, 9000),
    }
    for filename, content in variants.items():
        (outputs / filename).write_bytes(content)
    valid_path = outputs / "valid.png"
    valid_path.write_bytes(valid)

    class Flags:
        @staticmethod
        def resolved_output_dir():
            return outputs

    ctx = SimpleNamespace(flags=Flags())
    app = FastAPI()
    app.include_router(pro_api.build_router(ctx))
    with TestClient(app) as client:
        for filename in variants:
            path = outputs / filename
            assert image_artifact_dimensions(path) is None
            payload = pro_api._pro_workflow_run_payload(ctx, _record(path))
            assert payload["output"] is None
            assert payload["steps"][0]["receipt"]["image_url"] is None
            response = client.get(f"/api/pro/outputs/{filename}")
            assert response.status_code == 422

        valid_payload = pro_api._pro_workflow_run_payload(ctx, _record(valid_path))
        assert valid_payload["output"] == {
            "url": "/api/pro/outputs/valid.png",
            "width": 13,
            "height": 11,
        }
        assert valid_payload["steps"][0]["receipt"]["image_url"] == "/api/pro/outputs/valid.png"
        valid_response = client.get(valid_payload["output"]["url"])
        assert valid_response.status_code == 200
        assert valid_response.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(valid_response.content)) as image:
            image.verify()

        missing = outputs / "missing.png"
        missing_payload = pro_api._pro_workflow_run_payload(ctx, _record(missing))
        assert missing_payload["output"] is None
        assert missing_payload["steps"][0]["receipt"]["image_url"] is None
        assert client.get("/api/pro/outputs/missing.png").status_code == 404


def test_artifact_validation_rechecks_same_size_replacement_with_restored_mtime(tmp_path):
    import os

    path = tmp_path / "replace.png"
    valid = _image_bytes("PNG")
    path.write_bytes(valid)
    original_stat = path.stat()
    assert image_artifact_dimensions(path) == (13, 11)

    path.write_bytes(b"x" * len(valid))
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert image_artifact_dimensions(path) is None


def _fake_generation(tmp_path: Path):
    checkpoint_path = tmp_path / "fake-sdxl.safetensors"
    checkpoint_path.write_bytes(b"CPU fake checkpoint fixture")
    checkpoint = SimpleNamespace(
        id="fake-sdxl",
        title="fake-sdxl",
        filename=checkpoint_path.name,
        path=str(checkpoint_path),
        architecture="sdxl",
    )

    class FakeGeneration:
        @staticmethod
        def list_checkpoints():
            return [checkpoint]

        @staticmethod
        def resolve_checkpoint(checkpoint_id=None):
            return checkpoint if checkpoint_id in (None, checkpoint.id) else None

    return FakeGeneration()


def test_completed_workflow_saves_publishable_memory_image_instead_of_trusting_bad_path(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    corrupt = output_root / "corrupt-final.png"
    corrupt.write_bytes(b"corrupt image bytes")
    external = tmp_path / "external-valid.png"
    Image.new("RGB", (9, 7), (1, 2, 3)).save(external, format="PNG")
    save_calls: list[str] = []

    class FakeWorkflowService:
        def run(self, workflow, *, seed_image=None, on_progress=None, on_step_complete=None):
            on_progress(1, 1, "CPU fake image step")
            on_step_complete(
                1,
                WorkflowStepResult(
                    step_id="generate",
                    step_type=WorkflowStepType.TXT2IMG,
                    label="generate",
                    message="fake step finished",
                    image_path=str(corrupt),
                    seed=42,
                ),
            )
            return WorkflowRunResult(
                workflow_name=workflow.name,
                final_image_path=str(external),
                summary="CPU fake returned valid pixels and an image outside the output root",
            ), [Image.new("RGB", (8, 8), (10, 20, 30))]

    class Store:
        @staticmethod
        def save(image, _infotext, _subdir):
            saved = output_root / "saved-valid.png"
            image.save(saved, format="PNG")
            save_calls.append(str(saved))
            return SimpleNamespace(path=str(saved))

    class Flags:
        data_dir = tmp_path

        @staticmethod
        def resolved_output_dir():
            return output_root

    ctx = SimpleNamespace(
        flags=Flags(),
        workflows=FakeWorkflowService(),
        generation=_fake_generation(tmp_path),
        enhance=SimpleNamespace(store=Store()),
    )
    app = FastAPI()
    app.include_router(pro_api.build_router(ctx))
    client = TestClient(app)
    body = {
        "workflow": {
            "name": "corrupt output workflow fixture",
            "steps": [
                {
                    "id": "generate",
                    "type": "txt2img",
                    "params": {"prompt": "fixture only", "checkpoint_id": "fake-sdxl"},
                }
            ],
        },
        "idempotencyKey": "qa-corrupt-output-repaired",
    }
    try:
        submitted = client.post("/api/pro/workflows/runs", json=body)
        assert submitted.status_code == 202
        run_id = submitted.json()["runId"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/api/pro/workflows/runs/{run_id}")
            payload = response.json()
            if payload.get("status") in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)

        assert payload["status"] == "completed"
        assert payload["output"] == {
            "url": "/api/pro/outputs/saved-valid.png",
            "width": 8,
            "height": 8,
        }
        assert payload["steps"][0]["receipt"]["image_url"] is None
        assert save_calls == [str(output_root / "saved-valid.png")]
        assert client.get("/api/pro/outputs/corrupt-final.png").status_code == 422
        asset = client.get(payload["output"]["url"])
        assert asset.status_code == 200
        with Image.open(io.BytesIO(asset.content)) as image:
            image.verify()
    finally:
        client.close()
        service = getattr(ctx, "_pro_workflow_run_service", None)
        if service is not None:
            service._executor.shutdown(wait=True, cancel_futures=True)
