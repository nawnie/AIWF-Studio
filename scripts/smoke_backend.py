"""Backend-only smoke test for the image and video generation lanes.

No Gradio, no UI. Exercises the real checkpoint-loading + generation code
paths directly so regressions like the torch_dtype/dtype kwarg mismatch or
the Wan VAE channel-detection bug get caught before manual QA in the GUI.

By default, reads one prompt line each, at its own runtime, from
``_local/prompt_image.txt`` and ``_local/prompt_video.txt`` (repo-root
fallback). Both files are local-only and never committed.
The image prompt can be overridden with --prompt for validation runs.

Restricted to Q4/Q5 GGUF checkpoints only -- Q8 (and Q3/Q6 for Wan) are
known to exhaust VRAM or crash natively and are deliberately skipped here.

Each engine/checkpoint can also be run as its own isolated process (see the
--enumerate-* / single-target flags below). smoke_test.bat uses this so a
hard native crash in one engine (segfault, Rust panic, OOM abort -- the kind
of failure no try/except can catch) only kills that one subprocess; the loop
moves on and still tests everything else.

Usage:
    python scripts/smoke_backend.py                    # both lanes, one process
    python scripts/smoke_backend.py --image             # image lane only
    python scripts/smoke_backend.py --video             # video lane only
    python scripts/smoke_backend.py --list               # show what would run, no GPU work

    python scripts/smoke_backend.py --enumerate-checkpoints   # one checkpoint id per line
    python scripts/smoke_backend.py --checkpoint <id>         # run just that one checkpoint
    python scripts/smoke_backend.py --checkpoint <id> --steps 2 --width 512 --height 512
    python scripts/smoke_backend.py --enumerate-vae           # one VAE path per line
    python scripts/smoke_backend.py --vae <path>               # run just that one VAE load check
    python scripts/smoke_backend.py --video-gen                # video real-generation pass only
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _prompt_file(name: str) -> Path:
    """Prompt files are local-only (never committed); prefer the _local bucket."""
    local = ROOT / "_local" / name
    return local if local.is_file() else ROOT / name


PROMPT_IMAGE_FILE = _prompt_file("prompt_image.txt")
PROMPT_VIDEO_FILE = _prompt_file("prompt_video.txt")

# Only these quant tiers are considered safe to smoke-test. Q8 native-crashes
# on VRAM exhaustion (confirmed via aiwf-crash.log); Q3/Q6 are untested.
ALLOWED_QUANT_RE = re.compile(r"(?<![0-9])q[45](?:_k|k)?(?:_[ms])?(?=\b|high|low)", re.IGNORECASE)
BLOCKED_QUANT_RE = re.compile(r"(?<![0-9])q[3680](?:_k|k)?(?:_[ms])?(?=\b|high|low)", re.IGNORECASE)


def _read_first_line(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"{path} has no non-empty lines")


def _is_allowed_quant(filename: str) -> bool:
    name = filename.lower()
    if BLOCKED_QUANT_RE.search(name) and not ALLOWED_QUANT_RE.search(name):
        return False
    return bool(ALLOWED_QUANT_RE.search(name)) or "gguf" not in name


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _load_runtime_flags():
    """Build the same RuntimeFlags the real app would start with.

    A bare ``RuntimeFlags()`` relies on a default_factory that anchors
    data_dir to the installed aiwf package's own file location -- which is
    usually correct, but silently wrong if a custom models/ckpt directory is
    configured in the GUI's Settings -> Launch options (saved to
    launch.json) and not picked up here, or if anything about the install
    location doesn't line up with this checkout. This pins data_dir to the
    project root the smoke test itself lives in (this script's grandparent
    folder) and then layers the saved launch.json on top, exactly like
    aiwf.bootstrap does at real app startup, so the smoke test looks for
    models in the same place the GUI does.
    """
    from aiwf.core.config.launch import load_launch_settings, merge_launch_settings
    from aiwf.core.config.settings import RuntimeFlags

    flags = RuntimeFlags(data_dir=ROOT)
    launch_path = ROOT / "launch.json"
    saved = load_launch_settings(launch_path)
    if saved is not None:
        flags = merge_launch_settings(flags, saved)
    return flags


def _print_model_paths(flags) -> None:
    # stderr only -- stdout from --enumerate-checkpoints / --enumerate-vae is
    # parsed line-by-line by smoke_test.bat, so diagnostics must not leak in.
    print(f"data_dir:   {flags.data_dir}", file=sys.stderr)
    print(f"models_dir: {flags.resolved_models_dir()}", file=sys.stderr)
    print(f"ckpt_dir:   {flags.resolved_ckpt_dir()}", file=sys.stderr)
    for extra in flags.resolved_extra_ckpt_dirs():
        print(f"extra ckpt: {extra}", file=sys.stderr)


def _print_video_pipeline_registry(flags) -> None:
    from aiwf.core.config.settings import UserSettings
    from aiwf.services.pipeline_registry import PipelineRegistry

    print("Registered video pipelines:")
    for pipeline in PipelineRegistry(flags, UserSettings()).video_pipelines():
        state = "ready" if pipeline.ready else "needs setup"
        print(f"  {pipeline.id}: {state} - {pipeline.message}")
    print()


def _synthetic_inpaint_inputs(width: int, height: int):
    from PIL import Image, ImageDraw

    source = Image.new("RGB", (width, height), (96, 96, 96))
    src_draw = ImageDraw.Draw(source)
    src_draw.rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=(136, 136, 136))

    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((width // 3, height // 3, width * 2 // 3, height * 2 // 3), fill=255)
    return [source], [mask]


# --------------------------------------------------------------------------
# Image lane
# --------------------------------------------------------------------------

def _select_image_checkpoints(backend):
    checkpoints = backend.list_checkpoints()
    candidates = [c for c in checkpoints if _is_allowed_quant(c.filename)]
    seen_route: set[tuple[str, str]] = set()
    selected = []
    for c in candidates:
        route_key = (c.kind, c.architecture)
        if route_key in seen_route:
            continue
        seen_route.add(route_key)
        selected.append(c)
    return checkpoints, candidates, selected


def _make_image_backend():
    from aiwf.infrastructure.diffusers.backend import DiffusersBackend
    from aiwf.infrastructure.torch.devices import DeviceManager

    flags = _load_runtime_flags()
    devices = DeviceManager(flags)
    return DiffusersBackend(flags, devices)


def enumerate_checkpoints() -> int:
    backend = _make_image_backend()
    _print_model_paths(backend.flags if hasattr(backend, "flags") else _load_runtime_flags())
    _, _, selected = _select_image_checkpoints(backend)
    if not selected:
        print(f"No Q4/Q5 checkpoints found under {backend.ckpt_dir}", file=sys.stderr)
    for c in selected:
        print(c.id)
    return 0


def run_one_checkpoint(
    checkpoint_id: str,
    *,
    steps_override: int | None = None,
    width_override: int | None = None,
    height_override: int | None = None,
    prompt_override: str | None = None,
) -> int:
    from aiwf.core.domain.generation import GenerationMode, GenerationRequest
    from aiwf.infrastructure.diffusers.model_arch import is_inpaint_architecture
    from aiwf.infrastructure.diffusers.model_presets import resolve_model_preset

    backend = _make_image_backend()
    checkpoints = backend.list_checkpoints()
    checkpoint = next((c for c in checkpoints if c.id == checkpoint_id), None)
    if checkpoint is None:
        print(f"FAIL: unknown checkpoint id {checkpoint_id!r}")
        return 1

    if prompt_override is not None:
        prompt = prompt_override
    else:
        try:
            prompt = _read_first_line(PROMPT_IMAGE_FILE)
        except Exception as exc:
            print(f"FAIL: could not read prompt file: {exc}")
            return 1

    print(f"--- {checkpoint.filename} ({checkpoint.architecture}) ---")
    print(f"Prompt: {prompt!r}")
    preset = resolve_model_preset({}, checkpoint.id, checkpoint.architecture)
    steps = int(steps_override if steps_override is not None else preset.get("steps", 4))
    cfg_scale = float(preset.get("cfg_scale", 7.0))
    sampler = str(preset.get("sampler", "euler_a"))
    scheduler = str(preset.get("scheduler", "automatic"))
    width = int(width_override if width_override is not None else preset.get("width", 512))
    height = int(height_override if height_override is not None else preset.get("height", 512))
    mode = GenerationMode.INPAINT if is_inpaint_architecture(checkpoint.architecture) else GenerationMode.TXT2IMG
    if mode == GenerationMode.INPAINT and steps < 2:
        steps = 2
    print(
        f"Smoke settings: mode={mode.value} steps={steps} cfg={cfg_scale} "
        f"sampler={sampler} scheduler={scheduler} size={width}x{height}"
    )
    request = GenerationRequest(
        mode=mode,
        prompt=prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        sampler=sampler,
        scheduler=scheduler,
        batch_size=1,
        batch_count=1,
        checkpoint_id=checkpoint.id,
        save_images=False,
    )
    init_images = mask_images = None
    if mode == GenerationMode.INPAINT:
        init_images, mask_images = _synthetic_inpaint_inputs(width, height)
    try:
        result = backend.generate(request, init_images=init_images, mask_images=mask_images)
        n_images = len(getattr(result, "images", []) or [])
        print(f"PASS: generated {n_images} image(s)")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        from aiwf.core.domain.errors import ModelNotFoundError

        if not isinstance(exc, ModelNotFoundError):
            traceback.print_exc(limit=6)
        return 1


def run_image_lane(
    list_only: bool,
    *,
    steps_override: int | None = None,
    width_override: int | None = None,
    height_override: int | None = None,
    prompt_override: str | None = None,
) -> int:
    _print_header("IMAGE LANE")
    backend = _make_image_backend()
    checkpoints, candidates, selected = _select_image_checkpoints(backend)
    if not checkpoints:
        print(f"No checkpoints found under {backend.ckpt_dir}")
        return 1

    print(f"{len(checkpoints)} checkpoint(s) total, {len(candidates)} Q4/Q5-eligible, "
          f"{len(selected)} selected (one per architecture):")
    for c in selected:
        print(f"  - [{c.architecture}] {c.filename}")

    if list_only:
        return 0

    failures = 0
    for checkpoint in selected:
        rc = run_one_checkpoint(
            checkpoint.id,
            steps_override=steps_override,
            width_override=width_override,
            height_override=height_override,
            prompt_override=prompt_override,
        )
        failures += rc
        print()

    print(f"Image lane: {len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Video lane (Wan)
# --------------------------------------------------------------------------

def _wan_vae_files() -> list[Path]:
    flags = _load_runtime_flags()
    vae_dir = flags.resolved_models_dir() / "VAE"
    vae_files = sorted(vae_dir.glob("*wan*vae*.safetensors")) + sorted(vae_dir.glob("wan*.safetensors"))
    return sorted({p.resolve() for p in vae_files})


def enumerate_vae() -> int:
    _print_model_paths(_load_runtime_flags())
    files = _wan_vae_files()
    if not files:
        flags = _load_runtime_flags()
        print(f"No Wan VAE files found under {flags.resolved_models_dir() / 'VAE'}", file=sys.stderr)
    for p in files:
        print(str(p))
    return 0


def run_one_vae(vae_path: str) -> int:
    import torch
    from aiwf.infrastructure.wan.pipeline import _load_wan_vae

    p = Path(vae_path)
    try:
        vae = _load_wan_vae(str(p), torch.float16)
        z_dim = getattr(getattr(vae, "config", None), "z_dim", None)
        print(f"PASS: {p.name} -> loaded (z_dim={z_dim})")
        return 0
    except Exception as exc:
        print(f"FAIL: {p.name} -> {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
        return 1


def run_video_gen() -> int:
    from aiwf.core.config.settings import UserSettings

    flags = _load_runtime_flags()
    _print_model_paths(flags)
    gguf_dir = flags.resolved_models_dir() / "wan" / "GGUF"
    gguf_files = sorted(gguf_dir.glob("*.gguf")) if gguf_dir.is_dir() else []
    high_candidates = [p for p in gguf_files if "high" in p.name.lower() and _is_allowed_quant(p.name)]
    low_candidates = [p for p in gguf_files if "low" in p.name.lower() and _is_allowed_quant(p.name)]

    if not high_candidates or not low_candidates:
        print("No Q4/Q5 Wan high+low GGUF pair found -- skipping real generation pass.")
        return 0

    try:
        prompt = _read_first_line(PROMPT_VIDEO_FILE)
    except Exception as exc:
        print(f"FAIL: could not read prompt file: {exc}")
        return 1

    high_path = high_candidates[0]
    low_path = low_candidates[0]
    vae_files = _wan_vae_files()
    wan_vae_21 = next((p for p in vae_files if "2.1" in p.name or "wan21" in p.name.lower() or "_21" in p.name.lower()), None)

    print("Real generation pass:")
    print(f"  High noise: {high_path.name}")
    print(f"  Low noise:  {low_path.name}")
    print(f"  VAE:        {wan_vae_21.name if wan_vae_21 else '(auto)'}")
    print(f"  Prompt:     {prompt!r}")

    try:
        from PIL import Image
        from aiwf.core.domain.wan import WAN_RUNTIME_HIGH_LOW, WanI2VRequest
        from aiwf.services.wan import WanService

        settings = UserSettings()
        service = WanService(flags, settings)

        request = WanI2VRequest(
            prompt=prompt,
            num_frames=9,
            steps=1,
            high_noise_steps=1,
            low_noise_steps=1,
            width=256,
            height=256,
            runtime_mode=WAN_RUNTIME_HIGH_LOW,
            high_noise_model_id=str(high_path),
            low_noise_model_id=str(low_path),
            vae_id=str(wan_vae_21) if wan_vae_21 else "",
        )
        # A14B GGUF route is image-to-video only; synthesize a blank init frame.
        init_image = Image.new("RGB", (request.width, request.height), (128, 128, 128))

        preflight = service.preflight(request, image_present=True)
        if not preflight.ok:
            print(f"FAIL (preflight): {preflight.message()}")
            return 1

        result = service.generate(request, init_image)
        print(f"PASS: generated {result.frame_count} frame(s) -> {result.output_path}")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=8)
        return 1


def run_video_lane(list_only: bool) -> int:
    _print_header("VIDEO LANE")
    flags = _load_runtime_flags()
    _print_video_pipeline_registry(flags)

    print("VAE loader check (Wan 2.1 16ch vs Wan 2.2 48ch detection):")
    vae_files = _wan_vae_files()
    if not vae_files:
        print("  (no Wan VAE files found)")
    failures = 0
    for vae_path in vae_files:
        failures += run_one_vae(str(vae_path))

    if list_only:
        return 0

    print()
    failures += run_video_gen()
    return 1 if failures else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="store_true", help="Image lane only")
    parser.add_argument("--video", action="store_true", help="Video lane only")
    parser.add_argument("--list", action="store_true", help="List selected checkpoints/VAEs, do no GPU work")
    parser.add_argument("--enumerate-checkpoints", action="store_true", help="Print selected image checkpoint ids, one per line")
    parser.add_argument("--checkpoint", metavar="ID", help="Run only this one image checkpoint id")
    parser.add_argument("--steps", type=int, help="Override image smoke step count")
    parser.add_argument("--width", type=int, help="Override image smoke width")
    parser.add_argument("--height", type=int, help="Override image smoke height")
    parser.add_argument("--prompt", help="Override the image smoke prompt")
    parser.add_argument("--enumerate-vae", action="store_true", help="Print discovered Wan VAE paths, one per line")
    parser.add_argument("--vae", metavar="PATH", help="Run only this one Wan VAE load check")
    parser.add_argument("--video-gen", action="store_true", help="Run only the Wan real-generation pass")
    args = parser.parse_args()

    if args.enumerate_checkpoints:
        return enumerate_checkpoints()
    if args.checkpoint:
        return run_one_checkpoint(
            args.checkpoint,
            steps_override=args.steps,
            width_override=args.width,
            height_override=args.height,
            prompt_override=args.prompt,
        )
    if args.enumerate_vae:
        return enumerate_vae()
    if args.vae:
        return run_one_vae(args.vae)
    if args.video_gen:
        return run_video_gen()

    run_image = args.image or not args.video
    run_video = args.video or not args.image

    exit_code = 0
    if run_image:
        exit_code |= run_image_lane(
            args.list,
            steps_override=args.steps,
            width_override=args.width,
            height_override=args.height,
            prompt_override=args.prompt,
        )
    if run_video:
        exit_code |= run_video_lane(args.list)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
