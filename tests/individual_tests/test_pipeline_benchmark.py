from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
    assert data["runtime"]["app_version"]
    assert data["runtime"]["packages"]["torch"]


def test_run_with_receipt_writes_failed_receipt(tmp_path: Path, monkeypatch):
    def fail(_config):
        raise RuntimeError("no model")

    monkeypatch.setattr(pipeline_benchmark, "run_benchmark", fail)

    rc, receipt_path = pipeline_benchmark.run_with_receipt({"kind": "img2img"}, tmp_path)

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert data["status"] == "failed"
    assert data["error"] == "no model"


def test_unknown_benchmark_kind_fails():
    try:
        pipeline_benchmark.run_benchmark({"kind": "txt2img"})
    except ValueError as exc:
        assert "probe" in str(exc)
    else:
        raise AssertionError("unknown kind should fail")


def test_probe_benchmark_returns_capabilities(monkeypatch):
    monkeypatch.setattr(
        "aiwf.infrastructure.torch.wan_perf.describe_wan_acceleration_capabilities",
        lambda: {"gguf_runtime": {"available": True}},
    )

    result = pipeline_benchmark.run_benchmark({"kind": "probe", "label": "main"})

    assert result == {
        "kind": "probe",
        "label": "main",
        "wan_capabilities": {"gguf_runtime": {"available": True}},
    }


def test_probe_receipt_writes_capabilities(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "aiwf.infrastructure.torch.wan_perf.describe_wan_acceleration_capabilities",
        lambda: {"torchao": {"available": False}},
    )

    rc, receipt_path = pipeline_benchmark.run_with_receipt({"kind": "probe"}, tmp_path)

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert data["status"] == "completed"
    assert data["result"]["wan_capabilities"]["torchao"]["available"] is False


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
