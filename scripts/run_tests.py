from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = ROOT / "tests"
INDIVIDUAL_ROOT = TESTS_ROOT / "individual_tests"


SUITES: dict[str, tuple[str, ...]] = {
    "api": (
        "test_access.py",
        "test_api_controlnet_enhance.py",
        "test_api_parity.py",
        "test_api_security.py",
    ),
    "core": (
        "test_bootstrap_context.py",
        "test_bootstrap_env.py",
        "test_domain.py",
        "test_engine_domain.py",
        "test_extra_networks.py",
        "test_infotext.py",
        "test_launch.py",
        "test_metadata.py",
        "test_queue.py",
        "test_queue_history.py",
        "test_settings.py",
        "test_settings_backend_restart.py",
        "test_storage.py",
        "test_test_runner.py",
        "test_tags.py",
        "test_theme.py",
        "test_web_registry.py",
    ),
    "engines": (
        "test_app_startup.py",
        "test_devices.py",
        "test_engine_supervisor.py",
        "test_onnx_session.py",
        "test_ollama_client.py",
        "test_pipeline_benchmark.py",
        "test_pipeline_registry.py",
        "test_pipeline_preflight.py",
        "test_process_supervisor.py",
        "test_torch_attention.py",
        "test_vram_budget.py",
        "test_worker_probe.py",
        "test_worker_tenant.py",
    ),
    "generation": (
        "test_checkpoint_selection.py",
        "test_checkpoints.py",
        "test_controlnet_catalog.py",
        "test_controlnet_compat.py",
        "test_controlnet_plot_plugins.py",
        "test_controlnet_preprocess.py",
        "test_default_negative_and_clamp.py",
        "test_generate_wiring.py",
        "test_generation_prompts.py",
        "test_mask.py",
        "test_outpaint.py",
        "test_preload_guard.py",
        "test_prompt_encode.py",
        "test_prompt_dynamics.py",
        "test_prompt_processor.py",
        "test_prompt_style.py",
        "test_resolution_buckets.py",
        "test_sampler_scheduler.py",
        "test_samplers.py",
        "test_studio.py",
        "test_studio_lora_stack.py",
        "test_style_presets.py",
    ),
    "models": (
        "test_civitai_browser.py",
        "test_download_safety.py",
        "test_model_arch.py",
        "test_model_catalog.py",
        "test_model_download.py",
        "test_model_info_lookup.py",
        "test_model_inventory.py",
        "test_model_manager_ops.py",
        "test_model_ops.py",
        "test_model_path_imports.py",
        "test_model_profile.py",
        "test_model_scan_paths.py",
        "test_quantization.py",
    ),
    "optimization": (
        "tests/test_optimization.py",
    ),
    "services": (
        "test_audio.py",
        "test_client_log.py",
        "test_dev_diagnostics.py",
        "test_enhance.py",
        "test_failure_archive.py",
        "test_faceswap.py",
        "test_gallery_select.py",
        "test_photo_restore.py",
        "test_prompt_tools.py",
        "test_segment.py",
        "test_segment_blur.py",
        "test_segment_presets.py",
        "test_upscale.py",
        "test_workflow.py",
    ),
    "training": (
        "test_dataset_validator.py",
        "test_ed2_installer.py",
        "test_ed2_runner.py",
        "test_ed2_studio_compat.py",
        "test_training_config_builders.py",
        "test_training_engine_status.py",
        "test_training_tab.py",
    ),
    "ui": (
        "test_gallery_select.py",
        "test_settings.py",
        "test_studio.py",
        "test_theme.py",
        "test_web_registry.py",
    ),
    "video": (
        "test_comfy_backend.py",
        "test_rife.py",
        "test_video_export.py",
        "test_video_processing.py",
        "test_video_tools.py",
        "test_vsr.py",
    ),
    "wan": (
        "test_wan.py",
        "test_wan_acceleration.py",
        "test_wan_gguf_runtime.py",
        "test_wan_models.py",
        "test_wan_native_denoise.py",
        "test_wan_native_runtime.py",
        "test_wan_quant_format.py",
        "test_wan_runtime_readiness.py",
        "test_wan_sliced_sampler.py",
        "test_wan_vram.py",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AIWF Studio test suites without hunting through test files.",
    )
    parser.add_argument(
        "selection",
        nargs="*",
        help="Suite names or individual test files. Example: core wan test_launch.py",
    )
    parser.add_argument("--suite", action="append", default=[], help="Named suite to run. Can be repeated.")
    parser.add_argument("--test", action="append", default=[], help="Individual test file/path/nodeid. Can be repeated.")
    parser.add_argument("--full", action="store_true", help="Run the full test suite.")
    parser.add_argument("--list", action="store_true", help="List available suites and exit.")
    parser.add_argument("--no-quiet", action="store_true", help="Do not pass -q to pytest.")
    parser.add_argument("--pytest-arg", action="append", default=[], help="Extra argument passed to pytest. Can be repeated.")
    return parser


def suite_names() -> list[str]:
    return sorted(SUITES)


def list_suites() -> str:
    lines = ["Available suites:"]
    for name in suite_names():
        lines.append(f"  {name:<11} {len(SUITES[name])} files")
    lines.append("")
    lines.append("Examples:")
    lines.append("  python scripts/run_tests.py --full")
    lines.append("  python scripts/run_tests.py core wan")
    lines.append("  python scripts/run_tests.py --suite training --suite engines")
    lines.append("  python scripts/run_tests.py --test test_launch.py")
    lines.append("  python scripts/run_tests.py --test test_launch.py --pytest-arg=-x")
    return "\n".join(lines)


def resolve_test_path(value: str) -> str:
    if "::" in value:
        file_part, node_part = value.split("::", 1)
        return f"{resolve_test_path(file_part)}::{node_part}"

    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                (ROOT / path),
                (TESTS_ROOT / path),
                (INDIVIDUAL_ROOT / path),
            ]
        )
        if not path.name.startswith("test_") and path.suffix != ".py":
            candidates.append(INDIVIDUAL_ROOT / f"test_{value}.py")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise ValueError(f"Unknown test file: {value}")


def paths_for_suite(name: str) -> list[str]:
    key = name.strip().lower()
    if key in {"full", "all"}:
        return [str(TESTS_ROOT)]
    if key in {"individual", "individuals"}:
        return [str(INDIVIDUAL_ROOT)]
    if key not in SUITES:
        raise ValueError(f"Unknown suite: {name}")
    return [resolve_test_path(filename) for filename in SUITES[key]]


def resolve_targets(*, full: bool, suites: list[str], tests: list[str], selections: list[str]) -> list[str]:
    if full:
        return [str(TESTS_ROOT)]

    targets: list[str] = []
    for name in suites:
        targets.extend(paths_for_suite(name))
    for value in selections:
        if value.lower() in SUITES or value.lower() in {"full", "all", "individual", "individuals"}:
            targets.extend(paths_for_suite(value))
        else:
            targets.append(resolve_test_path(value))
    for value in tests:
        targets.append(resolve_test_path(value))

    if not targets:
        raise ValueError("No suite or test selected.")
    return _dedupe(targets)


def build_pytest_command(targets: list[str], *, quiet: bool = True, pytest_args: list[str] | None = None) -> list[str]:
    command = [sys.executable, "-m", "pytest"]
    if quiet:
        command.append("-q")
    command.extend(targets)
    command.extend(pytest_args or [])
    return command


def interactive_args() -> argparse.Namespace:
    print(list_suites())
    print("")
    raw = input("Run suites/tests (comma separated, or 'full'): ").strip()
    if not raw:
        raw = "full"
    selections = [part.strip() for part in raw.split(",") if part.strip()]
    return argparse.Namespace(
        selection=selections,
        suite=[],
        test=[],
        full=any(item.lower() in {"full", "all"} for item in selections),
        list=False,
        no_quiet=False,
        pytest_arg=[],
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print(list_suites())
        return 0
    if not args.full and not args.suite and not args.test and not args.selection:
        args = interactive_args()

    try:
        targets = resolve_targets(
            full=args.full,
            suites=args.suite,
            tests=args.test,
            selections=args.selection,
        )
    except ValueError as exc:
        parser.error(str(exc))

    command = build_pytest_command(targets, quiet=not args.no_quiet, pytest_args=args.pytest_arg)
    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=str(ROOT))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
