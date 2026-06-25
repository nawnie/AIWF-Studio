#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def _json_dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=None, separators=(",", ":")), flush=True)


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def self_test() -> int:
    core = ["numpy", "soundfile", "pedalboard", "pyloudnorm", "mido", "pretty_midi"]
    optional = ["librosa", "music21"]
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in core:
        try:
            versions[name] = _module_version(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    optional_status: dict[str, str] = {}
    for name in optional:
        try:
            optional_status[name] = _module_version(name)
        except Exception as exc:
            optional_status[name] = f"optional-missing: {exc}"
    payload = {"ok": not missing, "versions": versions, "optional": optional_status, "missing": missing}
    _json_dump(payload)
    return 0 if not missing else 2


def inspect_audio(path: Path) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
    return {
        "path": str(path),
        "name": path.name,
        "duration_seconds": duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": str(info.format),
        "subtype": str(info.subtype),
        "size_bytes": path.stat().st_size,
    }


def inspect_midi(path: Path) -> dict[str, Any]:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    tempo_times, tempi = midi.get_tempo_changes()
    tracks: list[dict[str, Any]] = []
    for index, instrument in enumerate(midi.instruments):
        velocities = [int(note.velocity) for note in instrument.notes]
        tracks.append(
            {
                "index": index,
                "name": instrument.name or f"Track {index + 1}",
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
                "note_count": len(instrument.notes),
                "pitch_min": min((int(note.pitch) for note in instrument.notes), default=None),
                "pitch_max": max((int(note.pitch) for note in instrument.notes), default=None),
                "velocity_mean": (sum(velocities) / len(velocities)) if velocities else 0.0,
                "start_seconds": min((float(note.start) for note in instrument.notes), default=0.0),
                "end_seconds": max((float(note.end) for note in instrument.notes), default=0.0),
            }
        )
    time_signatures = [
        {
            "numerator": int(item.numerator),
            "denominator": int(item.denominator),
            "time_seconds": float(item.time),
        }
        for item in midi.time_signature_changes
    ]
    key_signatures = [
        {"key_number": int(item.key_number), "time_seconds": float(item.time)}
        for item in midi.key_signature_changes
    ]
    return {
        "path": str(path),
        "name": path.name,
        "duration_seconds": float(midi.get_end_time()),
        "resolution": int(midi.resolution),
        "tempo_changes": [
            {"time_seconds": float(time_value), "bpm": float(bpm)}
            for time_value, bpm in zip(tempo_times.tolist(), tempi.tolist())
        ],
        "time_signatures": time_signatures,
        "key_signatures": key_signatures,
        "tracks": tracks,
    }


def _apply_board(audio, sample_rate: int, effects):
    """Run one offline effect board while preserving the input frame count.

    Pedalboard's ``reset=False`` streaming mode can legitimately return no
    samples for latency-bearing effects such as PitchShift until a later
    block arrives. Audio Lab processes complete regions, so each board call
    must flush its internal latency and then normalize the result back to the
    region's exact length. This keeps later automation times and project
    metadata stable.
    """
    import numpy as np
    from pedalboard import Pedalboard

    if not effects:
        return audio
    expected_frames = int(audio.shape[1])
    rendered = np.asarray(Pedalboard(effects)(audio, sample_rate, reset=True), dtype=np.float32)
    if rendered.ndim == 1:
        rendered = rendered[None, :]
    if rendered.shape[1] > expected_frames:
        rendered = rendered[:, :expected_frames]
    elif rendered.shape[1] < expected_frames:
        rendered = np.pad(rendered, ((0, 0), (0, expected_frames - rendered.shape[1])))
    return rendered


def _parse_envelope(text: str, duration: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw in (text or "").replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Envelope point '{item}' must use seconds:dB")
        seconds, gain_db = item.split(":", 1)
        point = (float(seconds.strip()), float(gain_db.strip()))
        if point[0] < 0 or point[0] > duration:
            raise ValueError(f"Envelope time {point[0]} is outside the clip duration {duration:.3f}s")
        points.append(point)
    points.sort(key=lambda value: value[0])
    return points


def _apply_pan(audio, pan: float):
    import numpy as np

    pan = max(-1.0, min(1.0, float(pan)))
    if audio.shape[0] == 1:
        mono = audio[0]
        angle = (pan + 1.0) * math.pi / 4.0
        return np.vstack([mono * math.cos(angle), mono * math.sin(angle)]).astype(np.float32)
    result = audio.copy()
    if pan > 0:
        result[0] *= math.cos(pan * math.pi / 2.0)
    elif pan < 0:
        result[1] *= math.cos(abs(pan) * math.pi / 2.0)
    return result


def _apply_envelope(audio, sample_rate: int, *, fade_in: float, fade_out: float, points_text: str):
    import numpy as np

    frames = audio.shape[1]
    duration = frames / sample_rate if sample_rate else 0.0
    gain = np.ones(frames, dtype=np.float32)
    if fade_in > 0:
        count = min(frames, max(1, int(round(fade_in * sample_rate))))
        gain[:count] *= np.linspace(0.0, 1.0, count, dtype=np.float32)
    if fade_out > 0:
        count = min(frames, max(1, int(round(fade_out * sample_rate))))
        gain[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    points = _parse_envelope(points_text, duration)
    if points:
        if points[0][0] > 0:
            points.insert(0, (0.0, points[0][1]))
        if points[-1][0] < duration:
            points.append((duration, points[-1][1]))
        times = np.array([item[0] * sample_rate for item in points], dtype=np.float64)
        values = np.array([10.0 ** (item[1] / 20.0) for item in points], dtype=np.float64)
        gain *= np.interp(np.arange(frames), times, values).astype(np.float32)
    return audio * gain[None, :]


def _normalize_loudness(audio, sample_rate: int, target_lufs: float):
    import numpy as np
    import pyloudnorm as pyln

    frames_first = audio.T
    meter = pyln.Meter(sample_rate)
    try:
        loudness = float(meter.integrated_loudness(frames_first))
        normalized = pyln.normalize.loudness(frames_first, loudness, float(target_lufs))
        return np.asarray(normalized.T, dtype=np.float32), loudness
    except Exception:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            return np.asarray(audio * min(1.0, 0.95 / peak), dtype=np.float32), None
        return audio, None


def process_job(job_path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from pedalboard import (
        Compressor,
        Gain,
        HighShelfFilter,
        HighpassFilter,
        Limiter,
        LowShelfFilter,
        LowpassFilter,
        NoiseGate,
        PeakFilter,
        PitchShift,
    )

    payload = json.loads(job_path.read_text(encoding="utf-8"))
    source = Path(payload["input_path"]).expanduser().resolve()
    output = Path(payload["output_path"]).expanduser().resolve()
    manifest_path = Path(payload["manifest_path"]).expanduser().resolve()
    settings = payload["settings"]
    stages = list(settings.get("stages") or [])
    started = time.perf_counter()
    data, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    audio = np.asarray(data.T, dtype=np.float32)
    stage_log: list[str] = []

    if "trim" in stages:
        start = int(round(float(settings.get("trim_start_seconds", 0.0)) * sample_rate))
        end_value = settings.get("trim_end_seconds")
        end = int(round(float(end_value) * sample_rate)) if end_value else audio.shape[1]
        audio = audio[:, max(0, start) : min(audio.shape[1], end)]
        stage_log.append(f"Trim: {start / sample_rate:.3f}s to {end / sample_rate:.3f}s")

    effects = []
    if "gate" in stages:
        effects.append(
            NoiseGate(
                threshold_db=float(settings["gate_threshold_db"]),
                ratio=float(settings["gate_ratio"]),
                attack_ms=float(settings["gate_attack_ms"]),
                release_ms=float(settings["gate_release_ms"]),
            )
        )
        stage_log.append("Noise gate")
    if "filters" in stages:
        effects.extend(
            [
                HighpassFilter(cutoff_frequency_hz=float(settings["highpass_hz"])),
                LowpassFilter(cutoff_frequency_hz=float(settings["lowpass_hz"])),
            ]
        )
        stage_log.append("High-pass / low-pass")
    if "eq" in stages:
        effects.extend(
            [
                LowShelfFilter(
                    cutoff_frequency_hz=float(settings["low_shelf_hz"]),
                    gain_db=float(settings["low_shelf_gain_db"]),
                    q=0.707,
                ),
                PeakFilter(
                    cutoff_frequency_hz=float(settings["mid_hz"]),
                    gain_db=float(settings["mid_gain_db"]),
                    q=float(settings["mid_q"]),
                ),
                HighShelfFilter(
                    cutoff_frequency_hz=float(settings["high_shelf_hz"]),
                    gain_db=float(settings["high_shelf_gain_db"]),
                    q=0.707,
                ),
            ]
        )
        stage_log.append("Three-band parametric EQ")
    if "compressor" in stages:
        effects.append(
            Compressor(
                threshold_db=float(settings["compressor_threshold_db"]),
                ratio=float(settings["compressor_ratio"]),
                attack_ms=float(settings["compressor_attack_ms"]),
                release_ms=float(settings["compressor_release_ms"]),
            )
        )
        stage_log.append("Compressor")
    audio = _apply_board(audio, sample_rate, effects)

    if "pitch" in stages and abs(float(settings.get("pitch_semitones", 0.0))) > 1e-6:
        start = int(round(float(settings.get("pitch_start_seconds", 0.0)) * sample_rate))
        end_value = settings.get("pitch_end_seconds")
        end = int(round(float(end_value) * sample_rate)) if end_value else audio.shape[1]
        start = max(0, min(audio.shape[1], start))
        end = max(start, min(audio.shape[1], end))
        shifted = _apply_board(
            audio[:, start:end], sample_rate, [PitchShift(semitones=float(settings["pitch_semitones"]))]
        )
        audio = np.concatenate([audio[:, :start], shifted, audio[:, end:]], axis=1)
        stage_log.append(
            f"Pitch shift: {float(settings['pitch_semitones']):+g} semitone(s), "
            f"{start / sample_rate:.3f}s–{end / sample_rate:.3f}s"
        )
    if "gain" in stages and abs(float(settings.get("gain_db", 0.0))) > 1e-6:
        audio = _apply_board(audio, sample_rate, [Gain(gain_db=float(settings["gain_db"]))])
        stage_log.append(f"Gain: {float(settings['gain_db']):+g} dB")
    if "pan" in stages:
        audio = _apply_pan(audio, float(settings.get("pan", 0.0)))
        stage_log.append(f"Pan: {float(settings.get('pan', 0.0)):+.2f}")
    if "envelope" in stages:
        audio = _apply_envelope(
            audio,
            sample_rate,
            fade_in=float(settings.get("fade_in_seconds", 0.0)),
            fade_out=float(settings.get("fade_out_seconds", 0.0)),
            points_text=str(settings.get("gain_envelope") or ""),
        )
        stage_log.append("Automation / fades")
    if "normalize" in stages:
        audio, measured = _normalize_loudness(audio, sample_rate, float(settings["target_lufs"]))
        stage_log.append(
            f"Loudness normalize: {measured:.2f} LUFS → {float(settings['target_lufs']):.2f} LUFS"
            if measured is not None
            else "Loudness normalize: peak fallback"
        )
    if "limiter" in stages:
        audio = _apply_board(
            audio,
            sample_rate,
            [
                Limiter(
                    threshold_db=float(settings["limiter_threshold_db"]),
                    release_ms=float(settings["limiter_release_ms"]),
                )
            ],
        )
        stage_log.append("Limiter")

    target_rate = int(settings.get("sample_rate") or 0)
    if target_rate > 0 and target_rate != sample_rate:
        try:
            import librosa

            audio = np.vstack(
                [librosa.resample(channel, orig_sr=sample_rate, target_sr=target_rate) for channel in audio]
            ).astype(np.float32)
            stage_log.append(f"Sample-rate conversion: {sample_rate} → {target_rate} Hz")
            sample_rate = target_rate
        except Exception as exc:
            stage_log.append(f"Sample-rate conversion skipped: {exc}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.stem}.partial{output.suffix}")
    subtype = "PCM_24" if output.suffix.lower() in {".wav", ".flac"} else None
    sf.write(str(temp), audio.T, sample_rate, subtype=subtype)
    os.replace(temp, output)
    elapsed = time.perf_counter() - started
    result = {
        "ok": True,
        "output_path": str(output),
        "manifest_path": str(manifest_path),
        "elapsed_seconds": elapsed,
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "duration_seconds": float(audio.shape[1] / sample_rate) if sample_rate else 0.0,
        "stage_log": stage_log,
    }
    manifest = dict(payload)
    manifest.update(result)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("path")
    midi_parser = sub.add_parser("midi-inspect")
    midi_parser.add_argument("path")
    process_parser = sub.add_parser("process")
    process_parser.add_argument("job")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            return self_test()
        if args.command == "inspect":
            _json_dump({"ok": True, "metadata": inspect_audio(Path(args.path).expanduser().resolve())})
            return 0
        if args.command == "midi-inspect":
            _json_dump({"ok": True, "metadata": inspect_midi(Path(args.path).expanduser().resolve())})
            return 0
        if args.command == "process":
            _json_dump(process_job(Path(args.job).expanduser().resolve()))
            return 0
    except Exception as exc:
        _json_dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
