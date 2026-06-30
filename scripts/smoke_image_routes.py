"""Run the no-UI image route smoke suite.

Each checkpoint runs through scripts/smoke_backend.py in a separate Python
process. A failed checkpoint is recorded, then the next route still runs.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_IMAGE_PROMPT = "a studio product photo of a red ceramic teapot on a gray table"


@dataclass(frozen=True)
class ImageRoute:
    checkpoint_id: str
    label: str
    family: str = "image"
    mode: str = "txt2img"
    smoke_steps: int = 2
    smoke_width: int = 512
    smoke_height: int = 512


@dataclass(frozen=True)
class RouteResult:
    checkpoint_id: str
    label: str
    ok: bool
    returncode: int | None
    elapsed_seconds: float
    timed_out: bool = False


IMAGE_ROUTES: tuple[ImageRoute, ...] = (
    ImageRoute("flux-kontext-4bit-fp4", "Flux Kontext FP4", family="flux"),
    ImageRoute("fluxFusionV24StepsGGUFNF4_V2NF4", "Flux NF4", family="flux"),
    ImageRoute("flux1-dev-Q4_K_M", "Flux GGUF dev", family="flux"),
    ImageRoute("fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4", "Z-Image GGUF", family="z_image"),
    ImageRoute("dreamshaperXL_lightningInpaint", "SDXL inpaint", family="sdxl", mode="inpaint"),
    ImageRoute("realisticVisionV60-inpainting15", "SD 1.5 inpaint", family="sd15", mode="inpaint"),
    ImageRoute("svdq-int4_r32-qwen-image-lightningv1.0-4steps", "Qwen Nunchaku 4-step", family="qwen_image"),
    ImageRoute("Sana_Sprint_0.6B_1024px_diffusers", "Sana Sprint", family="sana"),
    ImageRoute("FLUX.2-klein-4B", "Flux.2 Klein 4B", family="flux2_klein"),
)

LARGE_IMAGE_ROUTES: tuple[ImageRoute, ...] = (
    ImageRoute("Qwen-Image", "Qwen Image Diffusers", family="qwen_image", smoke_steps=1),
)


def _route_lookup() -> dict[str, ImageRoute]:
    return {route.checkpoint_id: route for route in (*IMAGE_ROUTES, *LARGE_IMAGE_ROUTES)}


def _selected_routes(only: list[str], skip: list[str], *, include_large: bool = False) -> list[ImageRoute]:
    lookup = _route_lookup()
    if only:
        routes = []
        for checkpoint_id in only:
            routes.append(lookup.get(checkpoint_id, ImageRoute(checkpoint_id, checkpoint_id)))
    else:
        routes = list(IMAGE_ROUTES)
        if include_large:
            routes.extend(LARGE_IMAGE_ROUTES)
    skip_set = set(skip)
    return [route for route in routes if route.checkpoint_id not in skip_set]


def _stream_process(command: list[str], *, timeout_seconds: int) -> tuple[int | None, bool]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    timed_out = False

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def flush_lines() -> None:
        while True:
            try:
                print(lines.get_nowait(), end="", flush=True)
            except queue.Empty:
                return

    while True:
        flush_lines()
        if process.poll() is not None:
            reader.join(timeout=5)
            flush_lines()
            break
        if timeout_seconds > 0 and time.monotonic() - started > timeout_seconds:
            timed_out = True
            process.kill()
            process.wait(timeout=10)
            reader.join(timeout=5)
            flush_lines()
            print(f"TIMEOUT: stopped after {timeout_seconds} seconds", flush=True)
            break
        time.sleep(0.1)

    return process.returncode, timed_out


def run_route(
    route: ImageRoute,
    *,
    python: Path,
    timeout_seconds: int,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    prompt: str | None = None,
) -> RouteResult:
    print()
    print("=" * 78)
    print(f"{route.label}: {route.checkpoint_id}")
    print("=" * 78)
    started = time.monotonic()
    command = [
        str(python),
        str(ROOT / "scripts" / "smoke_backend.py"),
        "--checkpoint",
        route.checkpoint_id,
    ]
    if steps is not None:
        command.extend(["--steps", str(steps)])
    if width is not None:
        command.extend(["--width", str(width)])
    if height is not None:
        command.extend(["--height", str(height)])
    if prompt is not None:
        command.extend(["--prompt", prompt])
    try:
        returncode, timed_out = _stream_process(command, timeout_seconds=timeout_seconds)
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f"FAIL: runner error for {route.checkpoint_id}: {type(exc).__name__}: {exc}", flush=True)
        return RouteResult(route.checkpoint_id, route.label, False, None, elapsed, False)

    elapsed = time.monotonic() - started
    ok = returncode == 0 and not timed_out
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {route.checkpoint_id} ({elapsed:.1f}s)", flush=True)
    return RouteResult(route.checkpoint_id, route.label, ok, returncode, elapsed, timed_out)


def route_smoke_plan(
    routes: Iterable[ImageRoute],
    *,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    prompt: str | None = None,
) -> dict[str, object]:
    """Return a no-GPU plan for the exact bounded smoke commands."""
    selected = list(routes)
    route_payloads: list[dict[str, object]] = []
    for route in selected:
        route_steps = int(steps if steps is not None else route.smoke_steps)
        route_width = int(width if width is not None else route.smoke_width)
        route_height = int(height if height is not None else route.smoke_height)
        route_prompt = prompt or DEFAULT_IMAGE_PROMPT
        command = [
            "venv\\Scripts\\python.exe",
            "scripts\\smoke_backend.py",
            "--checkpoint",
            route.checkpoint_id,
            "--steps",
            str(route_steps),
            "--width",
            str(route_width),
            "--height",
            str(route_height),
            "--prompt",
            route_prompt,
        ]
        route_payloads.append(
            {
                "checkpoint_id": route.checkpoint_id,
                "label": route.label,
                "family": route.family,
                "mode": route.mode,
                "steps": route_steps,
                "width": route_width,
                "height": route_height,
                "prompt": route_prompt,
                "command": command,
            }
        )
    return {
        "schema": 1,
        "kind": "image_route_smoke_plan",
        "route_count": len(route_payloads),
        "routes": route_payloads,
    }


def run_suite(
    routes: Iterable[ImageRoute],
    *,
    python: Path,
    timeout_seconds: int,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    prompt: str | None = None,
    runner: Callable[[ImageRoute, Path, int], RouteResult] | None = None,
) -> list[RouteResult]:
    results: list[RouteResult] = []
    call = runner or (
        lambda route, py, timeout: run_route(
            route,
            python=py,
            timeout_seconds=timeout,
            steps=steps,
            width=width,
            height=height,
            prompt=prompt,
        )
    )
    for route in routes:
        results.append(call(route, python, timeout_seconds))
    return results


def _print_list(routes: Iterable[ImageRoute]) -> None:
    for route in routes:
        print(f"{route.checkpoint_id}\t{route.label}")


def _print_summary(results: list[RouteResult]) -> None:
    passed = sum(1 for result in results if result.ok)
    total = len(results)
    print()
    print("=" * 78)
    print(f"Image route smoke summary: {passed}/{total} passed")
    print("=" * 78)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        timeout = " timeout" if result.timed_out else ""
        print(f"{status}: {result.checkpoint_id} ({result.elapsed_seconds:.1f}s{returncode_text(result)}{timeout})")


def returncode_text(result: RouteResult) -> str:
    if result.returncode is None:
        return ""
    return f", rc={result.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the route chain without running it.")
    parser.add_argument("--only", nargs="*", default=[], help="Run only these checkpoint ids.")
    parser.add_argument("--skip", nargs="*", default=[], help="Skip these checkpoint ids.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-checkpoint timeout. Use 0 to disable.",
    )
    parser.add_argument("--steps", type=int, help="Override image smoke step count.")
    parser.add_argument("--width", type=int, help="Override image smoke width.")
    parser.add_argument("--height", type=int, help="Override image smoke height.")
    parser.add_argument("--prompt", default=DEFAULT_IMAGE_PROMPT, help="Image smoke prompt override.")
    parser.add_argument("--include-large", action="store_true", help="Include large full Diffusers routes.")
    parser.add_argument("--plan-json", action="store_true", help="Print bounded smoke plan JSON without running.")
    parser.add_argument("--json", action="store_true", help="Print JSON results after the text summary.")
    parser.add_argument("--always-zero", action="store_true", help="Return 0 even if one or more routes fail.")
    args = parser.parse_args()

    routes = _selected_routes(args.only, args.skip, include_large=args.include_large)
    if args.list:
        _print_list(routes)
        return 0
    if args.plan_json:
        print(
            json.dumps(
                route_smoke_plan(
                    routes,
                    steps=args.steps,
                    width=args.width,
                    height=args.height,
                    prompt=args.prompt,
                ),
                indent=2,
            )
        )
        return 0
    if not routes:
        print("No image routes selected.")
        return 1

    python = ROOT / "venv" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)

    results = run_suite(
        routes,
        python=python,
        timeout_seconds=max(0, int(args.timeout_seconds)),
        steps=args.steps,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
    )
    _print_summary(results)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))

    failed = [result for result in results if not result.ok]
    if failed and not args.always_zero:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
