from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from aiwf.workers import pipeline_benchmark


def test_run_with_receipt_writes_completed_receipt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        pipeline_benchmark,
        "run_benchmark",
        lambda config: {
            "kind": config["kind"],
            "elapsed_seconds": 1.25,
            "units": 5,
            "units_label": "steps",
            "steps_per_second": 4.0,
            "iterations_per_second": 4.0,
            "load_seconds": 0.5,
            "preprocess_seconds": 0.25,
            "prompt_encode_seconds": 0.4,
            "image_encode_seconds": 0.2,
            "latent_prepare_seconds": 0.3,
            "denoise_seconds": 1.0,
            "vae_decode_seconds": 0.6,
            "video_postprocess_seconds": 0.15,
            "offload_cleanup_seconds": 0.05,
            "postprocess_seconds": 0.1,
            "video_write_seconds": 0.2,
            "fp8_linear_layers": 12,
            "fp8_fast_mm_calls": 96,
            "fp8_fallback_calls": 0,
            "fp8_fallback_layers": 0,
            "fp8_fallback_reasons": [],
            "fp8_strict_mode": True,
            "fp8_native_available": True,
            "cache_mode": "gpu_active_cpu_unpinned_standby",
        },
    )

    rc, receipt_path = pipeline_benchmark.run_with_receipt(
        {"kind": "img2img", "request": {"steps": 5}},
        tmp_path,
    )

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert data["status"] == "completed"
    assert data["result"]["steps_per_second"] == 4.0
    assert data["result"]["iterations_per_second"] == 4.0
    assert data["result"]["fp8_linear_layers"] == 12
    assert data["result"]["fp8_fast_mm_calls"] == 96
    assert data["result"]["fp8_fallback_calls"] == 0
    assert data["result"]["fp8_strict_mode"] is True
    assert data["result"]["cache_mode"] == "gpu_active_cpu_unpinned_standby"
    assert data["runtime"]["app_version"]
    assert data["runtime"]["packages"]["torch"]
    assert data["diagnostics_log"] == str(tmp_path / "dev-trace.log")
    assert data["optimization_profile"]["profile_id"] == "balanced_sdpa_fp16"
    assert data["capability_report"]["report_id"]
    typed = data["typed_receipt"]
    assert typed["receipt_id"] == data["benchmark_id"]
    assert typed["status"] == "completed"
    assert typed["timing"]["first_generation_time_s"] == 1.25
    assert typed["timing"]["load_time_s"] == 0.5
    assert typed["timing"]["prompt_encode_time_s"] == 0.4
    assert typed["timing"]["preprocess_time_s"] == 0.75
    assert typed["timing"]["denoise_time_s"] == 1.0
    assert typed["timing"]["vae_decode_time_s"] == 0.6
    assert typed["timing"]["postprocess_time_s"] == 0.5
    assert typed["optimization_profile"]["profile_id"] == "balanced_sdpa_fp16"
    assert typed["generation"]["steps"] == 5


def test_run_with_receipt_writes_failed_receipt(tmp_path: Path, monkeypatch):
    def fail(_config):
        raise RuntimeError("no model")

    monkeypatch.setattr(pipeline_benchmark, "run_benchmark", fail)

    rc, receipt_path = pipeline_benchmark.run_with_receipt({"kind": "img2img"}, tmp_path)

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert data["status"] == "failed"
    assert data["error"] == "no model"
    assert data["typed_receipt"]["status"] == "failed"
    assert data["typed_receipt"]["error"] == "no model"


def test_unknown_benchmark_kind_fails():
    try:
        pipeline_benchmark.run_benchmark({"kind": "nope"})
    except ValueError as exc:
        assert "txt2img" in str(exc)
        assert "controlnet" in str(exc)
    else:
        raise AssertionError("unknown kind should fail")


def test_image_benchmark_routes_wire_inputs(tmp_path: Path, monkeypatch):
    calls = []

    class FakeBackend:
        def __init__(self, _flags, _devices):
            pass

        def generate(
            self,
            request,
            init_images=None,
            mask_images=None,
            control_images=None,
            preview_every_n_steps=0,
        ):
            calls.append(
                {
                    "request": request,
                    "init_images": init_images,
                    "mask_images": mask_images,
                    "control_images": control_images,
                    "preview_every_n_steps": preview_every_n_steps,
                }
            )
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), "blue")])

    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    control = tmp_path / "control.png"
    Image.new("RGB", (8, 8), "white").save(source)
    Image.new("L", (8, 8), 255).save(mask)
    Image.new("RGB", (8, 8), "black").save(control)

    monkeypatch.setattr(pipeline_benchmark, "_flags_from_config", lambda _config: SimpleNamespace())
    monkeypatch.setattr("aiwf.infrastructure.torch.devices.DeviceManager", lambda _flags: SimpleNamespace())
    monkeypatch.setattr("aiwf.infrastructure.diffusers.backend.DiffusersBackend", FakeBackend)

    inpaint = pipeline_benchmark.run_benchmark(
        {
            "kind": "inpaint",
            "init_image": str(source),
            "mask_image": str(mask),
            "request": {"prompt": "repair", "steps": 3},
        }
    )
    controlnet = pipeline_benchmark.run_benchmark(
        {
            "kind": "controlnet",
            "control_image": str(control),
            "request": {"prompt": "edge", "steps": 4},
        }
    )
    hires = pipeline_benchmark.run_benchmark({"kind": "hires", "request": {"prompt": "large", "steps": 5}})

    assert inpaint["kind"] == "inpaint"
    assert inpaint["inputs"] == {"init_image": True, "mask_image": True, "control_image": False}
    assert controlnet["kind"] == "controlnet"
    assert controlnet["inputs"] == {"init_image": False, "mask_image": False, "control_image": True}
    assert hires["kind"] == "hires"
    assert hires["request"]["enable_hr"] is True
    assert calls[0]["request"].mode == "inpaint"
    assert calls[0]["init_images"] and calls[0]["mask_images"]
    assert calls[1]["request"].mode == "txt2img"
    assert calls[1]["control_images"]
    assert calls[2]["request"].enable_hr is True


def test_probe_benchmark_returns_capabilities(monkeypatch):
    monkeypatch.setattr(
        "aiwf.infrastructure.torch.wan_perf.describe_wan_acceleration_capabilities",
        lambda: {"gguf_runtime": {"available": True}},
    )
    monkeypatch.setattr(
        "aiwf.infrastructure.torch.wan_perf.describe_wan_hardware_fingerprint",
        lambda: {"gpu_name": "Synthetic GPU"},
    )

    result = pipeline_benchmark.run_benchmark({"kind": "probe", "label": "main"})

    assert result == {
        "kind": "probe",
        "label": "main",
        "wan_capabilities": {"gguf_runtime": {"available": True}},
        "hardware_fingerprint": {"gpu_name": "Synthetic GPU"},
        "transfer_probe": {},
    }


def test_probe_receipt_writes_capabilities(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "aiwf.infrastructure.torch.wan_perf.describe_wan_acceleration_capabilities",
        lambda: {"torchao": {"available": False}},
    )
    monkeypatch.setattr(
        "aiwf.infrastructure.torch.wan_perf.describe_wan_hardware_fingerprint",
        lambda: {"gpu_name": "Synthetic GPU"},
    )

    rc, receipt_path = pipeline_benchmark.run_with_receipt({"kind": "probe"}, tmp_path)

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert data["status"] == "completed"
    assert data["result"]["wan_capabilities"]["torchao"]["available"] is False
    assert data["optimization_profile"]["attention_backend"]["name"] == "sdpa"
    assert data["typed_receipt"]["pipeline"]["kind"] == "probe"


def test_run_with_receipt_honors_optimization_profile_config(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        pipeline_benchmark,
        "run_benchmark",
        lambda _config: {"kind": "probe"},
    )

    rc, receipt_path = pipeline_benchmark.run_with_receipt(
        {
            "kind": "probe",
            "optimization": {"profile_id": "safe_eager_cuda"},
        },
        tmp_path,
    )

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert data["optimization_profile"]["profile_id"] == "safe_eager_cuda"


def test_main_reads_bom_config(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"kind":"probe"}', encoding="utf-8-sig")
    out = tmp_path / "receipts"
    monkeypatch.setattr(
        pipeline_benchmark,
        "run_with_receipt",
        lambda loaded, out_dir: (0, out_dir / loaded["kind"]),
    )

    with patch("sys.argv", ["pipeline_benchmark", "--config", str(config), "--out", str(out)]):
        rc = pipeline_benchmark.main()

    assert rc == 0
