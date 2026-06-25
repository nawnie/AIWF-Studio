from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[2] / "engines" / "audio_lab" / "runner.py"


def _core_audio_deps_present() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "soundfile", "pedalboard", "pyloudnorm"))


@pytest.mark.skipif(not _core_audio_deps_present(), reason="Audio Lab optional dependencies are not installed in this environment")
def test_audio_runner_self_test_is_machine_readable() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "self-test"], capture_output=True, text=True, check=False)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode == 0
    assert payload["ok"] is True
    assert "pedalboard" in payload["versions"]


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
