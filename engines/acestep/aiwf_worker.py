#!/usr/bin/env python3
"""AIWF Studio's music worker for ACE-Step 1.5 (MIT), run inside the ACE-Step environment.

Studio starts one worker per render job (like the MMAudio worker), so the GPU is free again
as soon as the job ends. The worker reads a JSON job, loads ACE-Step from Studio's model
library only (no network: HF_HUB_OFFLINE), renders every item, and prints one JSON line.

Job file:
    {"checkpoints": "<models>/audio/ACE-Step-1.5", "device": "cuda", "offload": false,
     "items": [{"caption": "...", "lyrics": "", "duration": 30, "seed": 1234,
                "steps": 8, "output": "D:/out/song.wav"}]}

The planning language model is not loaded (thinking off): the DiT alone renders the music,
which keeps a render within a 16 GB card shared with other engines.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "ACE-Step-1.5"
CONFIG = "acestep-v15-turbo"


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _required_folders(checkpoints: Path) -> list[Path]:
    return [checkpoints / CONFIG, checkpoints / "vae", checkpoints / "Qwen3-Embedding-0.6B"]


def self_test(checkpoints: Path) -> int:
    """Import ACE-Step and check that the weights Studio installed are where ACE-Step looks."""
    missing = [str(path) for path in _required_folders(checkpoints) if not path.is_dir()]
    try:
        import torch  # noqa: F401
        from acestep.handler import AceStepHandler  # noqa: F401
        from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: F401
        import_error = ""
    except Exception as exc:  # report, do not crash: Studio shows this sentence
        import_error = f"{type(exc).__name__}: {exc}"
    ok = not missing and not import_error
    _emit({"ok": ok, "missing": missing, "import_error": import_error})
    return 0 if ok else 2


def render(job: dict) -> dict:
    checkpoints = Path(job["checkpoints"]).resolve()
    missing = [str(path) for path in _required_folders(checkpoints) if not path.is_dir()]
    if missing:
        raise RuntimeError("ACE-Step weights are incomplete: " + ", ".join(missing))

    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    started = time.perf_counter()
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(UPSTREAM),
        config_path=CONFIG,
        device=str(job.get("device") or "cuda"),
        offload_to_cpu=bool(job.get("offload", False)),
        offload_dit_to_cpu=bool(job.get("offload", False)),
    )
    if not ok:
        raise RuntimeError(f"ACE-Step could not start: {status}")
    loaded_seconds = time.perf_counter() - started

    results = []
    # this loop renders each requested piece with the already-loaded model
    for item in job["items"]:
        lyrics = str(item.get("lyrics") or "").strip()
        seed = int(item.get("seed", -1))
        params = GenerationParams(
            caption=str(item["caption"])[:512],
            lyrics=lyrics or "[Instrumental]",
            instrumental=not lyrics,
            duration=float(item.get("duration", 30)),
            inference_steps=int(item.get("steps", 8)),
            seed=seed,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=seed < 0)
        render_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="aiwf-acestep-") as scratch:
            result = generate_music(handler, None, params, config, save_dir=scratch)
            if not result.success or not result.audios:
                raise RuntimeError(f"ACE-Step did not produce audio: {result.error}")
            audio = result.audios[0]
            output = Path(item["output"]).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(audio["path"], output)
        results.append({
            "output": str(output),
            "sample_rate": int(audio.get("sample_rate") or 48000),
            "seed": int((audio.get("params") or {}).get("seed", seed)),
            "seconds": round(time.perf_counter() - render_started, 2),
        })
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {"ok": True, "results": results, "load_seconds": round(loaded_seconds, 2), "peak_vram_bytes": int(peak_vram)}


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never download at render time
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    if len(sys.argv) >= 3 and sys.argv[1] == "self-test":
        checkpoints = Path(sys.argv[2]).resolve()
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
        return self_test(checkpoints)
    if len(sys.argv) != 3 or sys.argv[1] != "render":
        _emit({"ok": False, "error": "usage: aiwf_worker.py render <job.json> | self-test <checkpoints>"})
        return 2
    try:
        job = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        # ACE-Step reads its weights from here (acestep/model_downloader.get_checkpoints_dir)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(Path(job["checkpoints"]).resolve())
        _emit(render(job))
        return 0
    except Exception as exc:
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
