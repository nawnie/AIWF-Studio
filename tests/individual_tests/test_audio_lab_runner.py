from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[2] / "engines" / "audio_lab" / "runner.py"


def _core_audio_deps_present() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "soundfile", "pyloudnorm"))


@pytest.mark.skipif(not _core_audio_deps_present(), reason="Audio Lab optional dependencies are not installed in this environment")
def test_audio_runner_self_test_is_machine_readable() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "self-test"], capture_output=True, text=True, check=False)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode == 0
    assert payload["ok"] is True
    # the signal chain is engines/audio_lab/dsp.py on NumPy/SciPy; Pedalboard (GPL-3.0) is gone
    assert "dsp" in payload["versions"] and "pedalboard" not in payload["versions"]


def test_audio_status_does_not_import_dsp_stack_during_studio_startup(tmp_path, monkeypatch) -> None:
    from aiwf.services.audio_lab import AudioLabService

    service = AudioLabService(tmp_path)
    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"stub")
    monkeypatch.setattr(service, "engine_python", lambda: fake_python)

    def fail_if_run(*_args, **_kwargs):
        pytest.fail("shallow startup status must not launch the isolated engine")

    monkeypatch.setattr(service, "_run", fail_if_run)
    status = service.status()
    assert status.installed is True
    assert status.details["deep_self_test"] == "not run during Studio startup"


def test_minimum_audio_summary_distinguishes_detected_from_runtime_ready() -> None:
    from aiwf.web.tabs.audio import _minimum_setup_markdown

    shallow = _minimum_setup_markdown({
        "runtimeChecksPerformed": False,
        "components": [{"id": "audio-lab", "label": "Audio Lab DSP", "ready": True}],
    })
    deep = _minimum_setup_markdown({
        "runtimeChecksPerformed": True,
        "components": [
            {"id": "audio-lab", "label": "Audio Lab DSP", "ready": True},
            {"id": "mmaudio", "label": "MMAudio", "ready": False},
        ],
    })

    assert "Detected: **Audio Lab DSP**" in shallow
    assert "Ready: **Audio Lab DSP**" in deep
    assert "Needs setup: **MMAudio**" in deep


@pytest.mark.skipif(not _core_audio_deps_present(), reason="Audio Lab optional dependencies are not installed in this environment")
def test_audio_runner_regional_pitch_preserves_timeline_for_later_envelopes(tmp_path) -> None:
    import numpy as np
    import soundfile as sf

    sample_rate = 48000
    seconds = 2.0
    timeline = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    source_audio = np.stack(
        [
            0.08 * np.sin(2.0 * np.pi * 220.0 * timeline),
            0.06 * np.sin(2.0 * np.pi * 330.0 * timeline),
        ],
        axis=1,
    ).astype(np.float32)
    source = tmp_path / "input.wav"
    output = tmp_path / "output.wav"
    manifest = tmp_path / "job.json"
    request = tmp_path / "request.json"
    sf.write(source, source_audio, sample_rate, subtype="PCM_24")

    payload = {
        "schema": 1,
        "job_id": "pitch_timeline_test",
        "input_path": str(source),
        "output_path": str(output),
        "manifest_path": str(manifest),
        "settings": {
            "stages": ["trim", "pitch", "envelope", "export"],
            "trim_start_seconds": 0.05,
            "trim_end_seconds": 1.80,
            "pitch_semitones": 1.0,
            "pitch_start_seconds": 0.25,
            "pitch_end_seconds": 0.75,
            "fade_in_seconds": 0.05,
            "fade_out_seconds": 0.10,
            "gain_envelope": "0:-3,0.5:0,1.5:-2",
            "export_format": "wav",
            "sample_rate": 0,
        },
    }
    request.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(RUNNER), "process", str(request)],
        capture_output=True,
        text=True,
        check=False,
    )
    response = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode == 0, response
    assert response["ok"] is True
    info = sf.info(output)
    assert info.subtype == "PCM_24"
    assert info.channels == 2
    assert info.frames == pytest.approx(round(1.75 * sample_rate), abs=2)
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert any(item.startswith("Pitch shift:") for item in written["stage_log"])
    assert "Automation / fades" in written["stage_log"]


def test_audio_lab_process_rejects_missing_empty_or_misdirected_export(tmp_path, monkeypatch):
    from aiwf.core.domain.audio_lab import AudioLabSettings
    from aiwf.services.audio_lab import AudioLabService

    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture source")
    service = AudioLabService(tmp_path)

    for behavior in ("missing", "empty", "wrong_path"):
        def fake_run(_args, *, behavior=behavior, **_kwargs):
            request_path = Path(_args[1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            output = Path(request["output_path"])
            if behavior != "missing":
                output.write_bytes(b"" if behavior == "empty" else b"processed fixture")
            reported = str(output) + ".wrong" if behavior == "wrong_path" else str(output)
            return {"ok": True, "output_path": reported}

        monkeypatch.setattr(service, "_run", fake_run)
        with pytest.raises(RuntimeError, match="Audio Lab engine"):
            service.process(source, AudioLabSettings())


def test_audio_lab_process_returns_verified_export(tmp_path, monkeypatch):
    from aiwf.core.domain.audio_lab import AudioLabSettings
    from aiwf.services.audio_lab import AudioLabService

    source = tmp_path / "input.wav"
    source.write_bytes(b"fixture source")
    service = AudioLabService(tmp_path)

    def fake_run(args, **_kwargs):
        request_path = Path(args[1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        output = Path(request["output_path"])
        output.write_bytes(b"processed fixture")
        return {
            "ok": True,
            "output_path": str(output),
            "manifest_path": request["manifest_path"],
        }

    monkeypatch.setattr(service, "_run", fake_run)
    result = service.process(source, AudioLabSettings())

    assert Path(result["output_path"]).read_bytes() == b"processed fixture"
    assert Path(result["request_path"]).is_file()
