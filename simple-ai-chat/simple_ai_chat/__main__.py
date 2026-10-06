"""Command-line tools for checking the registry and VRAM plans without a GPU.

    python -m simple_ai_chat models
    python -m simple_ai_chat simulate qwen3-asr-0.6b qwen-image-2.1-pe-t2i+qwen-image-2.1 qwen3-tts-1.7b-customvoice
    python -m simple_ai_chat command bonsai2-27b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends.fake import FakeBackend
from .backends.llama_server import build_command
from .manager import LoadError, ModelManager
from .registry import KNOWN_BACKENDS, Registry, Residency

ROOT = Path(__file__).resolve().parent.parent


def _load_registry(args: argparse.Namespace) -> Registry:
    return Registry.from_yaml(args.models, args.hardware)


def cmd_models(registry: Registry, _args: argparse.Namespace) -> int:
    hw = registry.hardware
    print(f"{hw.gpu_name}: budget {hw.vram_budget_gib:.2f} GiB "
          f"({hw.gpu_total_gib:.1f} total - {hw.reserved_gib:.1f} reserved - {hw.safety_margin_gib:.1f} margin)\n")
    print(f"{'id':32} {'on':3} {'residency':10} {'backend':12} {'VRAM':>7}  capabilities")
    for spec in registry:
        vram = f"{spec.vram_gib():.2f}" if spec.on_gpu else "cpu"
        flag = "yes" if spec.enabled else "-"
        print(f"{spec.id:32} {flag:3} {spec.residency.value:10} {spec.backend:12} {vram:>7}  "
              f"{', '.join(spec.capabilities)}{'  [exclusive]' if spec.exclusive else ''}")
    return 0


def cmd_simulate(registry: Registry, args: argparse.Namespace) -> int:
    fake = FakeBackend()
    manager = ModelManager(registry, {name: fake for name in KNOWN_BACKENDS})
    for spec in registry.enabled():
        if spec.residency is Residency.PINNED:
            manager.ensure_loaded(spec.id)
    print("start:", json.dumps(manager.snapshot()))
    for chain in args.model_ids:
        # "a+b" runs a and b as one chain: pinned models are restored only after b.
        with manager.hold_restore():
            for model_id in chain.split("+"):
                try:
                    with manager.using(model_id) as plan:
                        print(f"\n> {model_id}: {plan.action.value} ({plan.reason}); "
                              f"need {plan.need_gib:.2f} GiB, evict {list(plan.evict) or '-'}")
                        print("  during:", json.dumps(manager.snapshot()))
                except LoadError as exc:
                    print(f"\n> {model_id}: {exc}")
        print("  after: ", json.dumps(manager.snapshot()))
    return 0


def cmd_command(registry: Registry, args: argparse.Namespace) -> int:
    spec = registry.get(args.model_id)
    if spec.backend not in ("llama", "llama-prism"):
        print(f"{spec.id} uses backend {spec.backend!r}; no llama-server command", file=sys.stderr)
        return 1
    binary = registry.hardware.binaries.get(spec.backend, "llama-server")
    cmd = build_command(spec, binary=binary, port=args.port, models_dir=registry.hardware.models_dir)
    print(" ".join(cmd))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simple-ai-chat")
    parser.add_argument("--models", default=str(ROOT / "config" / "models.yaml"))
    parser.add_argument("--hardware", default=str(ROOT / "config" / "hardware.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list models with VRAM estimates")
    simulate = sub.add_parser("simulate", help="dry-run a sequence of jobs through the scheduler")
    simulate.add_argument("model_ids", nargs="+")
    command = sub.add_parser("command", help="print the llama-server command for a model")
    command.add_argument("model_id")
    command.add_argument("--port", type=int, default=8100)

    args = parser.parse_args(argv)
    registry = _load_registry(args)
    handlers = {"models": cmd_models, "simulate": cmd_simulate, "command": cmd_command}
    return handlers[args.command](registry, args)


if __name__ == "__main__":
    raise SystemExit(main())
