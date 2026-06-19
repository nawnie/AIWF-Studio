from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

from aiwf.services.training.ed2_runner import ED2Runner


class _CapturingSupervisor:
    def __init__(self) -> None:
        self.request: dict | None = None

    def is_running(self, worker_id: str) -> bool:
        return False

    def start(self, worker_id: str, command, *, check: bool = False):
        request_file = Path(command.args[2])
        self.request = json.loads(request_file.read_text(encoding="utf-8"))
        yield "captured"


def test_ed2_runner_passes_repo_dir_to_worker_request(tmp_path: Path):
    repo_dir = tmp_path / "engines" / "ed2" / "EveryDream2trainer"
    worker_script = tmp_path / "engines" / "ed2" / "worker.py"
    repo_dir.mkdir(parents=True)
    worker_script.parent.mkdir(parents=True, exist_ok=True)
    worker_script.write_text("print('worker')", encoding="utf-8")
    supervisor = _CapturingSupervisor()

    runner = ED2Runner(
        python_exe=Path(sys.executable),
        repo_root=tmp_path,
        supervisor=supervisor,  # type: ignore[arg-type]
    )

    events = list(runner.start({"job_name": "smoke"}, job_id="ed2_test"))

    assert events == ["captured"]
    assert supervisor.request is not None
    assert supervisor.request["_repo_dir"] == str(repo_dir)


def _load_worker_module():
    worker_path = Path(__file__).resolve().parents[2] / "engines" / "ed2" / "worker.py"
    spec = importlib.util.spec_from_file_location("aiwf_ed2_worker_for_test", worker_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ed2_worker_hf_id_heuristic_rejects_windows_paths():
    worker = _load_worker_module()

    assert worker._looks_like_hf_repo_id("stabilityai/stable-diffusion-xl-base-1.0")
    assert not worker._looks_like_hf_repo_id("C:/models/base.safetensors")
    assert not worker._looks_like_hf_repo_id("C:\\models\\base.safetensors")
    assert not worker._looks_like_hf_repo_id("/models/base")


def test_ed2_worker_preflight_blocks_missing_local_model(tmp_path: Path):
    worker = _load_worker_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "image.png").write_bytes(b"fake image")

    try:
        worker._validate_request_paths(
            {
                "dataset_dir": str(dataset),
                "base_model_path": "C:/does/not/exist/model.safetensors",
            }
        )
    except FileNotFoundError as exc:
        assert "Base model not found" in str(exc)
    else:
        raise AssertionError("missing local base model should fail preflight")


def test_ed2_worker_preflight_blocks_empty_dataset(tmp_path: Path):
    worker = _load_worker_module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"fake model")

    try:
        worker._validate_request_paths(
            {
                "dataset_dir": str(dataset),
                "base_model_path": str(model),
            }
        )
    except ValueError as exc:
        assert "no training images" in str(exc)
    else:
        raise AssertionError("empty dataset should fail preflight")
