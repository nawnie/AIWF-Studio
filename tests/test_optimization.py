from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

from aiwf.core.domain.optimization import (
    CapabilityFeature,
    CapabilityReport,
    GpuCapability,
    ModelFamily,
    OptimizationRequest,
    PipelineKind,
)
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.services.benchmark_receipts import BenchmarkReceiptService
from aiwf.services.optimization import CapabilityDetector, OptimizationPlanner, default_optimization_profiles
from aiwf.services.optimization_diagnostics import OptimizationDiagnosticsService


def test_default_profiles_keep_risky_features_off():
    profiles = default_optimization_profiles()

    balanced = profiles["balanced_sdpa_fp16"]
    assert balanced.attention_backend.name == "sdpa"
    assert balanced.compile_policy.enabled is False
    assert balanced.quant_policy.enabled is False
    assert balanced.engine_policy.enabled is False

    low_vram = profiles["low_vram_model_offload"]
    assert low_vram.memory_policy.cpu_offload == "model"
    assert low_vram.vae_policy.tiling == "auto"


def test_capability_detector_reports_optional_packages_without_importing_them(monkeypatch):
    def fake_version(package: str) -> str:
        if package == "torch":
            return "2.6.0"
        if package == "xformers":
            return "0.0.30"
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr("aiwf.services.optimization.metadata.version", fake_version)
    detector = CapabilityDetector(
        core_packages=("torch",),
        optional_packages={
            "attention.xformers": "xformers",
            "engine.tensorrt": "tensorrt",
        },
    )

    report = detector.detect(include_gpu=False)

    assert report.packages == {"torch": "2.6.0"}
    assert report.features["attention.xformers"].available is True
    assert report.features["attention.xformers"].version == "0.0.30"
    assert report.features["engine.tensorrt"].available is False


def test_planner_enables_xformers_only_when_available():
    capabilities = CapabilityReport(
        report_id="cap-1",
        features={"attention.xformers": CapabilityFeature(available=True, version="0.0.30")},
    )
    plan = OptimizationPlanner().resolve(
        OptimizationRequest(experimental_flags={"attention.xformers": True}),
        capabilities=capabilities,
    )

    assert plan.profile_id == "balanced_sdpa_fp16"
    assert plan.effective_profile.attention_backend.name == "xformers"
    assert any(decision.key == "attention.xformers" and decision.decision == "enabled" for decision in plan.decisions)


def test_planner_blocks_compile_for_arbitrary_lora_hotswap():
    capabilities = CapabilityReport(report_id="cap-1")
    plan = OptimizationPlanner().resolve(
        OptimizationRequest(
            experimental_flags={"compile.unet": True},
            lora_count=1,
            width=1024,
            height=1024,
        ),
        capabilities=capabilities,
    )

    assert plan.blocked is True
    assert plan.fallback_profile_id == "safe_eager_cuda"
    assert plan.effective_profile.compile_policy.enabled is False
    assert any(decision.key == "compile.unet" and decision.decision == "blocked" for decision in plan.decisions)


def test_fast_mode_requires_explicit_recipe():
    plan = OptimizationPlanner().resolve(
        OptimizationRequest(
            profile_id="fast_method_recipe",
            pipeline_kind=PipelineKind.FAST,
            model_family=ModelFamily.SDXL,
        )
    )

    assert plan.blocked is True
    assert any(decision.key == "fast_method" and decision.decision == "blocked" for decision in plan.decisions)


def test_fast_mode_applies_lightning_recipe():
    plan = OptimizationPlanner().resolve(
        OptimizationRequest(
            profile_id="fast_method_recipe",
            pipeline_kind=PipelineKind.FAST,
            model_family=ModelFamily.SDXL,
            fast_method="sdxl_lightning",
        )
    )

    assert plan.blocked is False
    assert plan.effective_profile.fast_method.name == "sdxl_lightning"
    assert plan.effective_profile.fast_method.cfg_scale == 0.0


def test_planner_disables_flash_and_sage_when_packages_missing():
    capabilities = CapabilityReport(report_id="cap-1", features={})

    plan = OptimizationPlanner().resolve(
        OptimizationRequest(experimental_flags={"attention.flash": True, "attention.sage": True}),
        capabilities=capabilities,
    )

    assert plan.effective_profile.attention_backend.name == "sdpa"
    assert any(decision.key == "attention.flash" and decision.decision == "disabled" for decision in plan.decisions)
    assert any(decision.key == "attention.sage" and decision.decision == "disabled" for decision in plan.decisions)


def test_planner_disables_onnx_runtime_without_gpu_package():
    capabilities = CapabilityReport(
        report_id="cap-1",
        features={
            "engine.onnx_runtime": CapabilityFeature(available=True, version="1.23.2"),
            "engine.onnx_runtime_gpu": CapabilityFeature(available=False, reason="missing"),
        },
    )

    plan = OptimizationPlanner().resolve(
        OptimizationRequest(experimental_flags={"engine.onnx_runtime": True}),
        capabilities=capabilities,
    )

    assert plan.effective_profile.engine_policy.enabled is False
    assert any(decision.key == "engine.onnx_runtime" and decision.decision == "disabled" for decision in plan.decisions)


def test_planner_gates_fp4_nvfp4_on_ada_gpu():
    capabilities = CapabilityReport(
        report_id="cap-4070",
        gpu=GpuCapability(name="RTX 4070 Ti SUPER", compute_capability="8.9"),
    )

    plan = OptimizationPlanner().resolve(
        OptimizationRequest(experimental_flags={"quant.fp4_nvfp4": True}),
        capabilities=capabilities,
    )

    assert plan.effective_profile.quant_policy.enabled is False
    assert any("not a runtime speed path" in decision.reason for decision in plan.decisions)


def test_planner_marks_vae_tiling_visible_and_warns_for_baseline():
    plan = OptimizationPlanner().resolve(
        OptimizationRequest(vae_tiling_requested=True, strict_quality_baseline=True)
    )

    assert plan.effective_profile.vae_policy.tiling == "on"
    assert plan.effective_profile.output_changes_visible is True
    assert any(decision.key == "vae.tiling" and decision.decision == "visible" for decision in plan.decisions)
    assert any(decision.key == "vae.tiling.baseline" and decision.severity == "warning" for decision in plan.decisions)


def test_fast_mode_blocks_unknown_method_without_recipe():
    plan = OptimizationPlanner().resolve(
        OptimizationRequest(
            profile_id="fast_method_recipe",
            pipeline_kind=PipelineKind.FAST,
            fast_method="experimental_magic",
        )
    )

    assert plan.blocked is True
    assert any(decision.key == "fast_method" and decision.decision == "blocked" for decision in plan.decisions)


def test_receipt_service_writes_profile_and_capability_metadata(tmp_path: Path):
    capabilities = CapabilityReport(
        report_id="cap-4070",
        packages={"torch": "2.6.0"},
        gpu=GpuCapability(name="RTX 4070 Ti SUPER", compute_capability="8.9", vram_total_bytes=16),
    )
    plan = OptimizationPlanner().resolve(OptimizationRequest(), capabilities=capabilities)
    service = BenchmarkReceiptService(tmp_path)

    receipt = service.build_receipt(
        plan=plan,
        capability_report=capabilities,
        model={"family": "sdxl", "checkpoint_sha256": "abc"},
        pipeline={"kind": "txt2img", "scheduler_class": "EulerDiscreteScheduler"},
        generation={"seed": 123, "width": 1024, "height": 1024},
    )
    path = service.write(receipt)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["receipt_id"] == receipt.receipt_id
    assert data["system"]["capability_report_id"] == "cap-4070"
    assert data["optimization_profile"]["profile_id"] == "balanced_sdpa_fp16"
    assert data["dependencies"]["torch"] == "2.6.0"


def test_diagnostics_reports_recorded_only_balanced_profile(tmp_path: Path):
    service = OptimizationDiagnosticsService(
        flags=RuntimeFlags(data_dir=tmp_path),
        settings=UserSettings(),
        detector=CapabilityDetector(core_packages=(), optional_packages={}),
        planner=OptimizationPlanner(),
        output_dir=tmp_path,
    )

    status = service.status()

    assert status["profile_id"] == "balanced_sdpa_fp16"
    assert status["active_runtime_paths"] == []
    assert any("recorded" in item.lower() for item in status["runtime_mismatches"])
    assert status["promotion_gates"]["status"] == "benchmark_required"
    assert status["known_failures"]


def test_diagnostics_marks_failed_receipts_ineligible(tmp_path: Path):
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    (bench / "failed.json").write_text(
        json.dumps({"status": "failed", "benchmark_id": "failed-one"}),
        encoding="utf-8",
    )
    service = OptimizationDiagnosticsService(
        flags=RuntimeFlags(data_dir=tmp_path),
        settings=UserSettings(),
        detector=CapabilityDetector(core_packages=(), optional_packages={}),
        planner=OptimizationPlanner(),
        output_dir=tmp_path,
    )

    status = service.status()

    assert status["latest_receipts"][0]["status"] == "failed"
    assert status["promotion_gates"]["status"] == "ineligible"


def test_diagnostics_marks_speedup_as_promotion_candidate(tmp_path: Path):
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    (bench / "baseline.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "benchmark_id": "baseline",
                "result": {"steps_per_second": 1.0},
                "optimization_profile": {"profile_id": "balanced_sdpa_fp16"},
            }
        ),
        encoding="utf-8",
    )
    (bench / "candidate.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "benchmark_id": "candidate",
                "result": {"steps_per_second": 1.2},
                "optimization_profile": {"profile_id": "experimental_feature_flags"},
            }
        ),
        encoding="utf-8",
    )
    service = OptimizationDiagnosticsService(
        flags=RuntimeFlags(data_dir=tmp_path),
        settings=UserSettings(),
        detector=CapabilityDetector(core_packages=(), optional_packages={}),
        planner=OptimizationPlanner(),
        output_dir=tmp_path,
    )

    status = service.status()

    assert status["promotion_gates"]["status"] == "promotion_candidate"
    assert status["promotion_gates"]["candidate"] is True
