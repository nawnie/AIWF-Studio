from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from aiwf.core.domain.audio_lab import (
    AUDIO_LAB_ORDER,
    AUDIO_STAGE_LABELS,
    AudioCommandPlan,
    AudioEngineStatus,
    AudioLabPlan,
    AudioLabSettings,
)

AUDIO_LAB_PRESETS: dict[str, dict[str, object]] = {
    "podcast_cleanup": {
        "stages": ["trim", "gate", "filters", "eq", "compressor", "normalize", "limiter", "export"],
        "gate_threshold_db": -48.0,
        "highpass_hz": 70.0,
        "lowpass_hz": 15500.0,
        "low_shelf_gain_db": -1.0,
        "mid_hz": 2800.0,
        "mid_gain_db": 1.5,
        "high_shelf_gain_db": 1.0,
        "compressor_threshold_db": -20.0,
        "compressor_ratio": 3.0,
        "target_lufs": -16.0,
        "limiter_threshold_db": -1.0,
    },
    "music_sweeten": {
        "stages": ["filters", "eq", "compressor", "gain", "normalize", "limiter", "export"],
        "highpass_hz": 25.0,
        "lowpass_hz": 20500.0,
        "low_shelf_gain_db": 0.5,
        "mid_hz": 2500.0,
        "mid_gain_db": 0.5,
        "high_shelf_gain_db": 0.75,
        "compressor_threshold_db": -16.0,
        "compressor_ratio": 2.0,
        "gain_db": 0.0,
        "target_lufs": -14.0,
        "limiter_threshold_db": -0.8,
    },
    "old_recording": {
        "stages": ["gate", "filters", "eq", "compressor", "normalize", "limiter", "export"],
        "gate_threshold_db": -55.0,
        "gate_ratio": 3.0,
        "highpass_hz": 45.0,
        "lowpass_hz": 12500.0,
        "low_shelf_gain_db": -1.5,
        "mid_hz": 3200.0,
        "mid_gain_db": -1.0,
        "high_shelf_gain_db": -0.5,
        "compressor_threshold_db": -24.0,
        "compressor_ratio": 2.5,
        "target_lufs": -16.0,
    },
    "custom": {"stages": ["normalize", "limiter", "export"]},
}


def preset_audio_settings(name: str) -> AudioLabSettings:
    preset = name if name in AUDIO_LAB_PRESETS else "custom"
    return AudioLabSettings(preset=preset, **AUDIO_LAB_PRESETS[preset])


def resolve_audio_plan(settings: AudioLabSettings) -> AudioLabPlan:
    selected = set(settings.stages)
    stages = [stage for stage in AUDIO_LAB_ORDER if stage in selected]
    if "export" not in stages:
        stages.append("export")
    warnings: list[str] = []
    if "pitch" in stages and abs(settings.pitch_semitones) < 1e-6:
        warnings.append("Pitch shift is selected but set to 0 semitones.")
    if "envelope" in stages and not settings.gain_envelope and not settings.fade_in_seconds and not settings.fade_out_seconds:
        warnings.append("Automation is selected but no fade or gain-envelope points are defined.")
    return AudioLabPlan(stages=stages, labels=[AUDIO_STAGE_LABELS[item] for item in stages], warnings=warnings)


_NUMBER_WORDS: dict[str, float] = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
    "ten": 10.0, "eleven": 11.0, "twelve": 12.0,
}
_NUMBER_TOKEN = r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def _number_value(token: str) -> float:
    value = token.strip().lower()
    return _NUMBER_WORDS.get(value, float(value) if re.fullmatch(r"\d+(?:\.\d+)?", value) else 0.0)


def parse_audio_command(text: str) -> AudioCommandPlan:
    """Parse a small, deterministic subset of DAW-style natural-language edits.

    This is a planner, not a hidden promise that the multitrack edit has already
    happened. The structured result is intended for the future project engine.
    """

    source = (text or "").strip()
    lowered = source.lower()
    if not source:
        return AudioCommandPlan(understood=False, notes=["Enter a command to preview."])

    transpose = re.search(
        rf"(?:measure|bar)\s+({_NUMBER_TOKEN}).*?(up|down)\s+({_NUMBER_TOKEN})\s+semitone", lowered
    )
    if transpose:
        direction = 1 if transpose.group(2) == "up" else -1
        return AudioCommandPlan(
            understood=True,
            operation="transpose_region",
            parameters={
                "start_measure": int(_number_value(transpose.group(1))),
                "semitones": int(direction * _number_value(transpose.group(3))),
                "region_hint": "second chorus" if "second chorus" in lowered else None,
            },
            notes=[
                "Execution needs a project tempo map and region/marker metadata.",
                "The v5 planner preserves the intent; destructive multitrack execution is not enabled yet.",
            ],
        )

    add_track = re.search(r"add\s+(?:a\s+)?([a-z][a-z ]+?)\s+track", lowered)
    if add_track and "unison" in lowered:
        instrument = add_track.group(1).strip().title()
        source_track_match = re.search(r"(?:of|with)\s+([a-z][a-z ]*?track\s*\d+)", lowered)
        velocity_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*velocity", lowered)
        fade_match = re.search(rf"fade\s*out.*?({_NUMBER_TOKEN})\s*beats?", lowered)
        octave = -1 if "octave below" in lowered else 1 if "octave above" in lowered else 0
        pan = 1.0 if "right speaker" in lowered or "pan right" in lowered else -1.0 if "left speaker" in lowered or "pan left" in lowered else 0.0
        return AudioCommandPlan(
            understood=True,
            operation="duplicate_orchestrate_track",
            parameters={
                "instrument": instrument,
                "source_track": source_track_match.group(1).strip() if source_track_match else None,
                "interval_semitones": octave * 12,
                "velocity_scale": float(velocity_match.group(1)) / 100.0 if velocity_match else 1.0,
                "fade_out_beats": _number_value(fade_match.group(1)) if fade_match else 0.0,
                "pan": pan,
                "alignment": "unison",
            },
            notes=[
                "Execution requires MIDI or note-event metadata for the source track.",
                "The planner does not infer notes from a mixed waveform in v5.",
            ],
        )

    return AudioCommandPlan(
        understood=False,
        operation="unparsed",
        parameters={"source_text": source},
        notes=[
            "The initial grammar understands measure-based transposition and unison-track orchestration examples.",
            "Free-form DAW command execution is a later Audio Lab engine milestone.",
        ],
    )


class AudioLabService:
    def __init__(self, output_root: str | Path) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.engine_dir = self.repo_root / "engines" / "audio_lab"
        self.runner = self.engine_dir / "runner.py"
        self.output_root = Path(output_root).expanduser().resolve() / "audio-lab"

    def engine_python(self) -> Path:
        if os.name == "nt":
            return self.engine_dir / ".venv" / "Scripts" / "python.exe"
        return self.engine_dir / ".venv" / "bin" / "python"

    @staticmethod
    def _decode_json(stdout: str) -> dict:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"Audio Lab engine returned no JSON result. Output: {stdout[-1200:]}")

    def _run(self, args: list[str], *, timeout: int = 3600) -> dict:
        python = self.engine_python()
        if not python.is_file():
            raise RuntimeError("Audio Lab engine is not installed. Use the Engine tab to install it first.")
        result = subprocess.run(
            [str(python), str(self.runner), *args],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        payload = self._decode_json(result.stdout)
        if result.returncode != 0 or not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or result.stderr or "Audio Lab engine failed"))
        return payload

    def status(self, *, deep: bool = False) -> AudioEngineStatus:
        """Return engine status without slowing Studio startup.

        A deep check launches the isolated interpreter and imports the DSP stack.
        That belongs behind the explicit Refresh/Install actions, not in the
        Gradio tab construction path.
        """
        python = self.engine_python()
        if not python.is_file():
            return AudioEngineStatus(
                installed=False,
                message="Optional Audio Lab engine is not installed.",
                details={"expected_python": str(python)},
            )
        if not deep:
            return AudioEngineStatus(
                installed=True,
                python_path=str(python),
                message="Audio Lab environment detected. Click Refresh status for a full self-test.",
                details={"deep_self_test": "not run during Studio startup"},
            )
        try:
            payload = self._run(["self-test"], timeout=120)
        except Exception as exc:
            return AudioEngineStatus(
                installed=False,
                python_path=str(python),
                message=f"Audio Lab environment exists but failed self-test: {exc}",
            )
        return AudioEngineStatus(
            installed=True,
            python_path=str(python),
            message="Audio Lab engine is ready.",
            details=payload,
        )

    def install(self, *, upgrade: bool = False) -> str:
        script = self.repo_root / "scripts" / "bootstrap_audio_lab.py"
        command = [sys.executable, str(script), "--repo", str(self.repo_root), "--json"]
        if upgrade:
            command.append("--upgrade")
        result = subprocess.run(command, cwd=str(self.repo_root), capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Audio Lab installation failed").strip())
        payload = self._decode_json(result.stdout)
        return json.dumps(payload, indent=2)

    def inspect_audio(self, path: str | Path) -> dict:
        return self._run(["inspect", str(Path(path).expanduser().resolve())], timeout=120)["metadata"]

    def inspect_midi(self, path: str | Path) -> dict:
        return self._run(["midi-inspect", str(Path(path).expanduser().resolve())], timeout=120)["metadata"]

    def build_plan(self, settings: AudioLabSettings) -> AudioLabPlan:
        return resolve_audio_plan(settings)

    def process(self, source: str | Path, settings: AudioLabSettings) -> dict:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"Audio input does not exist: {source_path}")
        plan = self.build_plan(settings)
        job_id = f"alab_{uuid.uuid4().hex[:12]}"
        job_dir = self.output_root / datetime.now().strftime("%Y%m%d") / job_id
        extension = settings.export_format
        output_path = job_dir / f"{source_path.stem}_processed.{extension}"
        manifest_path = job_dir / "job.json"
        request_path = job_dir / "request.json"
        job_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "schema": 1,
            "job_id": job_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(source_path),
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "resolved_order": plan.stages,
            "warnings": plan.warnings,
            "settings": settings.model_dump(mode="json"),
        }
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        result = self._run(["process", str(request_path)], timeout=7200)
        result["request_path"] = str(request_path)
        result["warnings"] = plan.warnings
        return result
