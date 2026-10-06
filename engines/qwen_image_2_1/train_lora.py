"""Train a Qwen-Image 2.1 LoRA from the command line (no GUI).

Examples (Windows paths shown; forward slashes work too):

  # text-to-image / style / character LoRA on a 16 GB card with ai-toolkit
  python engines/qwen_image_2_1/train_lora.py --name shawn_v1 --dataset D:/data/shawn --trigger ohwx_shawn

  # reference-guided edit LoRA (targets in --dataset, references in --control-dir; same basenames)
  python engines/qwen_image_2_1/train_lora.py --name outfit_swap --dataset D:/data/targets --control-dir D:/data/refs

  # only write the YAML, do not start anything
  python engines/qwen_image_2_1/train_lora.py --name test --dataset D:/data/x --write-config-only

  # official DiffSynth-Studio recipe instead of ai-toolkit
  python engines/qwen_image_2_1/train_lora.py --trainer diffsynth --name test --dataset D:/data/x

Emits JSONL progress events (kind=status/progress/artifact/complete/error) when --jsonl is set,
matching docs/WORKER_PROTOCOL.md so AIWF's ProcessSupervisor can drive it later.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

from qwen21.settings import Settings  # noqa: E402
from qwen21.training import aitoolkit, dataset as ds, diffsynth  # noqa: E402
from qwen21.training.runner import Progress, TrainingRunner  # noqa: E402


def _emit(jsonl: bool, kind: str, **fields) -> None:
    if jsonl:
        print(json.dumps({"kind": kind, "job_id": fields.pop("job_id", "qwen21_train"), **fields}), flush=True)
    else:
        msg = fields.get("message") or fields.get("detail") or ""
        print(f"[{kind}] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen-Image 2.1 LoRA trainer wrapper (ai-toolkit / DiffSynth-Studio)")
    p.add_argument("--trainer", choices=("aitoolkit", "diffsynth"), default="aitoolkit")
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", required=True, help="folder with images and .txt captions")
    p.add_argument("--control-dir", action="append", default=[], help="reference image folder(s) for edit LoRAs (max 3)")
    p.add_argument("--trigger", default="", help="trigger word; written into captions with --prepend-trigger")
    p.add_argument("--prepend-trigger", action="store_true", help="prefix every caption with the trigger word")
    p.add_argument("--default-caption", default="", help="caption for images without a .txt")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=None, help="defaults to rank")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--resolution", default="1024", help="comma list of bucket sizes, e.g. 768,1024")
    p.add_argument("--timestep-type", default="shift", help="ai-toolkit timestep_type: shift|sigmoid|weighted|linear")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--rgba", action="store_true", help="dataset has transparent PNGs; train RGBA output")
    p.add_argument("--vram", choices=("16", "24"), default="16", help="memory preset")
    p.add_argument("--no-offload", action="store_true", help="disable layer offloading (needs ~24 GB)")
    p.add_argument("--qtype", default="convrot8", help="ai-toolkit DiT quantization: convrot8|float8|uint4|none")
    p.add_argument("--qtype-te", default="convrot8", help="ai-toolkit text-encoder quantization")
    p.add_argument("--sample", action="store_true", help="sample during training (costs VRAM/time)")
    p.add_argument("--sample-prompt", action="append", default=[])
    p.add_argument("--name-or-path", default=aitoolkit.DEFAULT_NAME_OR_PATH,
                   help="ai-toolkit base weights: HF repo, local folder, or single .safetensors DiT")
    p.add_argument("--output-root", default=None, help="training output folder (default from settings)")
    p.add_argument("--ai-toolkit-dir", default=None)
    p.add_argument("--python", default=None, help="trainer venv python")
    p.add_argument("--diffsynth-dir", default=None)
    p.add_argument("--write-config-only", action="store_true")
    p.add_argument("--jsonl", action="store_true", help="emit AIWF worker-protocol JSONL events")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()
    jsonl = args.jsonl
    out_root = Path(args.output_root or settings.training_output_dir)
    job_dir = out_root / args.name
    job_dir.mkdir(parents=True, exist_ok=True)

    report = ds.scan_dataset(args.dataset, args.control_dir)
    print(report.summary(), file=sys.stderr)
    if not report.ok:
        _emit(jsonl, "error", detail="\n".join(report.errors), message="dataset problems")
        return 2
    if args.trigger and args.prepend_trigger:
        n = ds.prepend_trigger(args.dataset, args.trigger)
        _emit(jsonl, "status", message=f"prepended trigger to {n} captions")
    if args.default_caption and report.captions_missing:
        n = ds.ensure_captions(args.dataset, args.default_caption, args.trigger)
        _emit(jsonl, "status", message=f"wrote {n} default captions")

    if args.trainer == "aitoolkit":
        spec = aitoolkit.TrainSpec(name=args.name, dataset_dir=str(Path(args.dataset).resolve()),
                                   control_dirs=[str(Path(c).resolve()) for c in args.control_dir],
                                   training_folder=str(out_root), name_or_path=args.name_or_path,
                                   default_caption=args.default_caption, rank=args.rank,
                                   alpha=args.alpha or args.rank, learning_rate=args.lr, steps=args.steps,
                                   resolutions=[int(r) for r in args.resolution.split(",")],
                                   timestep_type=args.timestep_type, save_every=args.save_every,
                                   rgba=args.rgba, sampling=args.sample, sample_prompts=args.sample_prompt)
        aitoolkit.apply_preset(spec, aitoolkit.PRESET_24GB if args.vram == "24" else aitoolkit.PRESET_16GB)
        spec.rank, spec.alpha = args.rank, args.alpha or args.rank
        spec.resolutions = [int(r) for r in args.resolution.split(",")]
        spec.qtype, spec.qtype_te = args.qtype, args.qtype_te
        spec.quantize, spec.quantize_te = args.qtype != "none", args.qtype_te != "none"
        if args.no_offload:
            spec.layer_offloading = False
        problems = aitoolkit.validate_spec(spec)
        if problems:
            _emit(jsonl, "error", detail="\n".join(problems), message="config problems")
            return 2
        config_path = aitoolkit.write_config(spec, job_dir / f"{args.name}.aitoolkit.yaml")
        _emit(jsonl, "status", message=f"config written: {config_path}")
        if args.write_config_only:
            print(config_path)
            return 0
        tk_dir = Path(args.ai_toolkit_dir or settings.ai_toolkit_dir)
        python = Path(args.python) if args.python else settings.ai_toolkit_python_exe()
        if not (tk_dir / "run.py").is_file() or not python.is_file():
            _emit(jsonl, "error", detail=f"ai-toolkit not found: {tk_dir / 'run.py'} / {python}",
                  message="run scripts/bootstrap_qwen21.ps1 -WithAiToolkit first")
            return 3
        cmd, cwd = aitoolkit.build_command(python, tk_dir, config_path), tk_dir
        artifact = aitoolkit.expected_output_lora(spec, tk_dir)
    else:
        edit = bool(args.control_dir)
        meta = ds.write_diffsynth_metadata(args.dataset, args.control_dir or None, args.default_caption)
        spec_d = diffsynth.DiffSynthSpec(name=args.name, dataset_dir=str(Path(args.dataset).resolve()),
                                         metadata_path=str(meta), edit_mode=edit, output_path=str(job_dir / "diffsynth"),
                                         learning_rate=args.lr, lora_rank=args.rank,
                                         num_epochs=max(1, args.steps // 400), save_steps=args.save_every)
        ds_dir = Path(args.diffsynth_dir or settings.diffsynth_dir)
        python = Path(args.python) if args.python else settings.diffsynth_python_exe()
        cmd = diffsynth.build_command(spec_d, python, ds_dir)
        (job_dir / f"{args.name}.diffsynth.cmd.txt").write_text(diffsynth.command_to_shell(cmd), encoding="utf-8")
        _emit(jsonl, "status", message=f"metadata: {meta}; command saved next to it")
        if args.write_config_only:
            print(diffsynth.command_to_shell(cmd))
            return 0
        if not python.is_file():
            _emit(jsonl, "error", detail=f"DiffSynth python not found: {python}", message="install DiffSynth-Studio")
            return 3
        cwd = ds_dir
        artifact = job_dir / "diffsynth"

    log_path = job_dir / f"{args.name}.log"
    started = time.time()

    def on_progress(prog: Progress) -> None:
        _emit(jsonl, "progress", step=prog.step, total=prog.total,
              message=f"step {prog.step}/{prog.total}" + (f" loss {prog.loss:.4f}" if prog.loss is not None else ""))

    def on_line(line: str) -> None:
        if not jsonl:
            print(line, flush=True)

    runner = TrainingRunner(cmd, cwd, on_line=on_line, on_progress=on_progress, log_path=log_path)
    _emit(jsonl, "status", message="launching: " + " ".join(cmd))
    runner.start()
    try:
        while runner.running:
            time.sleep(1.0)
            if jsonl and int(time.time() - started) % 30 == 0:
                _emit(jsonl, "heartbeat")
    except KeyboardInterrupt:
        runner.stop()
        _emit(jsonl, "error", detail="interrupted", message="cancelled")
        return 130
    rc = runner.returncode or 0
    if rc == 0:
        if Path(artifact).exists():
            _emit(jsonl, "artifact", path=str(artifact))
        _emit(jsonl, "complete", message=f"training finished in {time.time() - started:.0f} s; LoRA at {artifact}")
        return 0
    _emit(jsonl, "error", detail=f"trainer exit code {rc}; see {log_path}", message="training failed")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
