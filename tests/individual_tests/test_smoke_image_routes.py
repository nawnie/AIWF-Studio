from pathlib import Path

from scripts.smoke_image_routes import (
    ImageRoute,
    ROOT,
    RouteResult,
    _selected_routes,
    run_route,
    run_suite,
)


def test_default_image_route_chain_includes_requested_paths():
    ids = [route.checkpoint_id for route in _selected_routes([], [])]

    assert "flux-kontext-4bit-fp4" in ids
    assert "fluxFusionV24StepsGGUFNF4_V2NF4" in ids
    assert "flux1-dev-Q4_K_M" in ids
    assert "fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM" in ids
    assert "fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4" in ids
    assert "dreamshaperXL_lightningInpaint" in ids
    assert "realisticVisionV60-inpainting15" in ids
    assert "svdq-int4_r32-qwen-image-lightningv1.0-4steps" in ids
    assert "Sana_Sprint_0.6B_1024px_diffusers" in ids
    assert "FLUX.2-klein-4B" in ids


def test_full_qwen_image_is_large_opt_in_route():
    default_ids = [route.checkpoint_id for route in _selected_routes([], [])]
    large_ids = [route.checkpoint_id for route in _selected_routes([], [], include_large=True)]
    explicit_ids = [route.checkpoint_id for route in _selected_routes(["Qwen-Image"], [])]

    assert "Qwen-Image" not in default_ids
    assert "Qwen-Image" in large_ids
    assert explicit_ids == ["Qwen-Image"]


def test_run_suite_keeps_running_after_failure():
    calls: list[str] = []

    def fake_runner(route: ImageRoute, python: Path, timeout_seconds: int) -> RouteResult:
        calls.append(route.checkpoint_id)
        return RouteResult(
            checkpoint_id=route.checkpoint_id,
            label=route.label,
            ok=route.checkpoint_id != "fails",
            returncode=1 if route.checkpoint_id == "fails" else 0,
            elapsed_seconds=0.1,
        )

    routes = [ImageRoute("fails", "Fails"), ImageRoute("passes", "Passes")]
    results = run_suite(routes, python=Path("python"), timeout_seconds=1, runner=fake_runner)

    assert calls == ["fails", "passes"]
    assert [result.ok for result in results] == [False, True]


def test_run_route_passes_image_overrides(monkeypatch):
    commands: list[list[str]] = []

    def fake_stream(command: list[str], *, timeout_seconds: int):
        commands.append(command)
        return 0, False

    monkeypatch.setattr("scripts.smoke_image_routes._stream_process", fake_stream)

    result = run_route(
        ImageRoute("demo", "Demo"),
        python=Path("python"),
        timeout_seconds=1,
        steps=2,
        width=512,
        height=512,
    )

    assert result.ok is True
    assert commands == [
        [
            "python",
            str(ROOT / "scripts" / "smoke_backend.py"),
            "--checkpoint",
            "demo",
            "--steps",
            "2",
            "--width",
            "512",
            "--height",
            "512",
        ]
    ]


def test_run_route_passes_prompt_override(monkeypatch):
    commands: list[list[str]] = []

    def fake_stream(command: list[str], *, timeout_seconds: int):
        commands.append(command)
        return 0, False

    monkeypatch.setattr("scripts.smoke_image_routes._stream_process", fake_stream)

    result = run_route(
        ImageRoute("demo", "Demo"),
        python=Path("python"),
        timeout_seconds=1,
        prompt="a neutral smoke prompt",
    )

    assert result.ok is True
    assert commands == [
        [
            "python",
            str(ROOT / "scripts" / "smoke_backend.py"),
            "--checkpoint",
            "demo",
            "--prompt",
            "a neutral smoke prompt",
        ]
    ]
