#!/usr/bin/env python3
"""AIWF Studio's sound-effect worker for MOSS-SoundEffect v2.0 (Apache-2.0), run in its own env.

Studio starts one worker per job, so the GPU is free again when the job ends. The worker reads
a JSON job, loads the pipeline from Studio's model library only (HF_HUB_OFFLINE), renders every
item with the one loaded model (the two-step video soundtrack asks for several sounds at once),
and prints one JSON line.

Job file:
    {"model_dir": "<models>/audio/MOSS-SoundEffect-v2.0", "device": "cuda",
     "items": [{"prompt": "door creaks open", "seconds": 4.0, "seed": 7, "steps": 50,
                "cfg_scale": 4.0, "negative_prompt": "", "output": "D:/out/door.wav"}]}

torch.compile is switched off (TORCHDYNAMO_DISABLE): it needs Triton, which Windows does not
ship, and the upstream README recommends the same switch when compilation fails.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "MOSS-TTS"
_REQUIRED = ("model_index.json", "transformer", "text_encoder", "tokenizer", "vae", "scheduler")


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def self_test(model_dir: Path) -> int:
    """Import the pipeline and check that the weights Studio installed are complete."""
    missing = [name for name in _REQUIRED if not (model_dir / name).exists()]
    try:
        import torch  # noqa: F401
        from moss_soundeffect_v2 import MossSoundEffectPipeline  # noqa: F401
        import_error = ""
    except Exception as exc:  # reported to Studio, not raised
        import_error = f"{type(exc).__name__}: {exc}"
    ok = not missing and not import_error
    _emit({"ok": ok, "missing": missing, "import_error": import_error})
    return 0 if ok else 2


def render(job: dict) -> dict:
    model_dir = Path(job["model_dir"]).resolve()
    missing = [name for name in _REQUIRED if not (model_dir / name).exists()]
    if missing:
        raise RuntimeError("MOSS-SoundEffect weights are incomplete: " + ", ".join(missing))

    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    started = time.perf_counter()
    pipe = MossSoundEffectPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16,
                                                   device=str(job.get("device") or "cuda"))
    loaded_seconds = time.perf_counter() - started
    results = []
    # this loop renders each requested sound with the already-loaded pipeline
    for item in job["items"]:
        seconds = max(0.5, min(float(item.get("seconds", 5.0)), float(pipe.max_inference_seconds)))
        render_started = time.perf_counter()
        audio = pipe(
            prompt=str(item["prompt"]),
            seconds=seconds,
            num_inference_steps=int(item.get("steps", 50)),
            cfg_scale=float(item.get("cfg_scale", 4.0)),
            seed=int(item.get("seed", 0)) if int(item.get("seed", 0)) >= 0 else int(time.time()) % 2_000_000_000,
            negative_prompt=str(item.get("negative_prompt") or ""),
            progress_bar_cmd=lambda iterable, **_kwargs: iterable,
        )
        output = Path(item["output"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # written with soundfile rather than pipe.save_audio: torchaudio 2.9 saves through
        # torchcodec, which needs FFmpeg DLLs that Windows installs often lack
        import soundfile

        waveform = audio.detach().to(torch.float32).cpu().numpy()[0]       # (channels, samples)
        # The model emits very quiet audio (measured peak 0.01, about -40 dBFS). Sound effects are
        # delivered peak-normalized, so scale to -3 dBFS; the shape of the sound is unchanged.
        peak = float(abs(waveform).max())
        if peak > 1e-6:
            waveform = waveform * (0.708 / peak)
        soundfile.write(str(output), waveform.T, int(pipe.sample_rate), subtype="PCM_24")
        results.append({"output": str(output), "sample_rate": int(pipe.sample_rate), "seconds_rendered": seconds,
                        "seconds": round(time.perf_counter() - render_started, 2)})
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {"ok": True, "results": results, "load_seconds": round(loaded_seconds, 2), "peak_vram_bytes": int(peak_vram)}


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # never download at render time
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # no torch.compile/Triton on Windows
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    if len(sys.argv) >= 3 and sys.argv[1] == "self-test":
        return self_test(Path(sys.argv[2]).resolve())
    if len(sys.argv) != 3 or sys.argv[1] != "render":
        _emit({"ok": False, "error": "usage: aiwf_worker.py render <job.json> | self-test <model_dir>"})
        return 2
    try:
        _emit(render(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))))
        return 0
    except Exception as exc:
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
