from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.generation import GenerationRequest
from aiwf.services.qwen_nunchaku import QwenNunchakuService, QwenNunchakuUnavailable


def test_qwen_nunchaku_generation_defers_before_worker_when_vram_is_low(tmp_path: Path, monkeypatch):
    service = QwenNunchakuService(RuntimeFlags(data_dir=tmp_path, output_dir=tmp_path / "outputs"))
    monkeypatch.setattr(service, "status", lambda _path: SimpleNamespace(ready=True, messages=()))
    monkeypatch.setattr(
        service,
        "_headroom_issue",
        lambda: "Qwen Nunchaku launch deferred: 2.4 GB VRAM is free; the selected route needs at least 8.0 GB headroom.",
    )
    monkeypatch.setattr(
        "aiwf.services.qwen_nunchaku.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Qwen Nunchaku worker launched"),
    )
    checkpoint = SimpleNamespace(path=str(tmp_path / "transformer.safetensors"))

    with pytest.raises(QwenNunchakuUnavailable, match="2.4 GB VRAM is free"):
        service.generate(
            checkpoint,
            GenerationRequest(prompt="test"),
            prompt="test",
            width=512,
            height=512,
            steps=4,
            seed=1,
        )


def test_qwen_nunchaku_headroom_fails_closed_when_nvidia_smi_cannot_map_device(monkeypatch):
    monkeypatch.setattr("aiwf.services.gpu_memory.nvidia_smi_free_bytes", lambda: None)

    issue = QwenNunchakuService._headroom_issue()

    assert issue is not None
    assert "could not be verified" in issue


def test_qwen_nunchaku_status_rejects_a_python_exe_without_required_runtime_imports(tmp_path: Path):
    service = QwenNunchakuService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    python = service.python_exe()
    python.parent.mkdir(parents=True)
    python.write_bytes(b"not a Python executable")
    service.runner_script().write_text("print('runner')", encoding="utf-8")

    status = service.status()

    assert status.ready is False
    assert any("isolated runtime import check" in message.lower() for message in status.messages)
