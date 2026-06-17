from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.optimization import CapabilityReport, OptimizationRequest, OptimizationPlan
from aiwf.services.optimization import CapabilityDetector, OptimizationPlanner


KNOWN_OPTIMIZATION_FAILURES: tuple[dict[str, str], ...] = (
    {
        "id": "kernels_windows_import",
        "status": "blocked",
        "summary": "`kernels` test lane broke Diffusers Wan import in the copied accelerator venv.",
    },
    {
        "id": "torchao_torch_triton",
        "status": "blocked",
        "summary": "TorchAO test installs failed to import cleanly with the current Torch/Triton stack.",
    },
    {
        "id": "sageattention_2_windows",
        "status": "unavailable",
        "summary": "SageAttention 2.x was not available from the current Windows/PyPI lane.",
    },
    {
        "id": "wan_fp8_scaled_mm_fallback",
        "status": "active_investigation",
        "summary": "Wan FP8 safetensors repeatedly fell back from `_scaled_mm` to bf16 linear.",
    },
    {
        "id": "wan_model_offload_16gb_oom",
        "status": "observed",
        "summary": "Wan model offload OOMed on a 16 GB GPU during a 384x384 smoke test.",
    },
)


class OptimizationDiagnosticsService:
    """Read-only optimization status and promotion-gate reporting."""

    def __init__(
        self,
        *,
        flags: RuntimeFlags,
        settings: UserSettings,
        detector: CapabilityDetector,
        planner: OptimizationPlanner,
        output_dir: Path | str,
    ) -> None:
        self.flags = flags
        self.settings = settings
        self.detector = detector
        self.planner = planner
        self.output_dir = Path(output_dir)

    def status(self) -> dict[str, Any]:
        capability_report = self.detector.detect(include_gpu=True)
        request = OptimizationRequest(
            profile_id=getattr(self.settings, "optimization_profile_id", "balanced_sdpa_fp16")
        )
        plan = self.planner.resolve(request, capabilities=capability_report)
        receipts = self._recent_receipts(limit=12)
        return {
            "profile_id": plan.profile_id,
            "requested_profile_id": plan.requested_profile_id,
            "runtime_flags": self._runtime_flags(),
            "capability_report": capability_report.model_dump(mode="json"),
            "planner_decisions": [decision.model_dump(mode="json") for decision in plan.decisions],
            "active_runtime_paths": self._active_runtime_paths(),
            "runtime_mismatches": self._runtime_mismatches(plan, capability_report),
            "known_failures": list(KNOWN_OPTIMIZATION_FAILURES),
            "latest_receipts": receipts,
            "promotion_gates": self._promotion_gates(receipts),
        }

    def status_markdown(self) -> str:
        status = self.status()
        active = status["active_runtime_paths"] or ["Recorded only"]
        blocked = [
            decision
            for decision in status["planner_decisions"]
            if decision.get("decision") in {"blocked", "disabled"}
        ]
        gates = status["promotion_gates"]
        lines = [
            f"**Profile:** `{status['profile_id']}`",
            f"**Runtime:** {', '.join(f'`{item}`' for item in active)}",
            f"**Promotion:** {gates['status']}",
        ]
        if blocked:
            lines.append("**Blocked:** " + "; ".join(item["reason"] for item in blocked[:3]))
        elif status["runtime_mismatches"]:
            lines.append("**Mismatches:** " + "; ".join(status["runtime_mismatches"][:3]))
        else:
            lines.append("**Planner:** no blocking decisions")
        return "  \n".join(lines)

    def _runtime_flags(self) -> dict[str, Any]:
        names = (
            "inference_backend",
            "onnx_provider",
            "xformers",
            "opt_sdp_attention",
            "opt_split_attention",
            "channels_last",
            "torch_compile",
            "torchao",
            "fp8_quant",
            "cuda_graphs",
            "lowvram",
            "medvram",
        )
        return {name: getattr(self.flags, name, None) for name in names}

    def _active_runtime_paths(self) -> list[str]:
        active: list[str] = []
        if getattr(self.flags, "inference_backend", "diffusers") != "diffusers":
            active.append(f"backend:{self.flags.inference_backend}")
        if getattr(self.flags, "xformers", False):
            active.append("xFormers flag")
        if getattr(self.flags, "opt_sdp_attention", False) or getattr(self.flags, "opt_split_attention", False):
            active.append("SDP attention flag")
        if getattr(self.flags, "channels_last", False):
            active.append("channels-last flag")
        if getattr(self.flags, "torch_compile", False):
            active.append("torch.compile flag")
        if getattr(self.flags, "torchao", False):
            active.append("TorchAO flag")
        if getattr(self.flags, "fp8_quant", False):
            active.append("FP8 quant flag")
        if getattr(self.flags, "cuda_graphs", False):
            active.append("CUDA graphs flag")
        if getattr(self.flags, "lowvram", False):
            active.append("Low VRAM launch flag")
        return active

    def _runtime_mismatches(self, plan: OptimizationPlan, capability_report: CapabilityReport) -> list[str]:
        mismatches: list[str] = []
        flags = self._runtime_flags()
        features = capability_report.features
        if flags.get("xformers") and not features.get("attention.xformers", None):
            mismatches.append("xFormers launch flag is set but no capability entry was detected.")
        if flags.get("xformers") and features.get("attention.xformers") and not features["attention.xformers"].available:
            mismatches.append("xFormers launch flag is set but xFormers is unavailable.")
        if flags.get("torchao") and features.get("quant.torchao_fp8") and not features["quant.torchao_fp8"].available:
            mismatches.append("TorchAO launch flag is set but torchao is unavailable.")
        if flags.get("inference_backend") == "onnx" and not features.get("engine.onnx_runtime", None):
            mismatches.append("ONNX backend is selected but ONNX Runtime capability was not detected.")
        if not self._active_runtime_paths() and plan.profile_id != "safe_eager_cuda":
            mismatches.append("Selected profile is recorded for diagnostics; no backend-changing runtime flag is active.")
        return mismatches

    def _recent_receipts(self, *, limit: int) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        bench = self.output_dir / "benchmarks"
        if bench.exists():
            candidates.extend(bench.glob("*.json"))
            candidates.extend((bench / "receipts").glob("**/*.json"))
        receipts: list[dict[str, Any]] = []
        for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            receipts.append(
                {
                    "path": str(path),
                    "status": data.get("status", ""),
                    "benchmark_id": data.get("benchmark_id") or data.get("receipt_id") or path.stem,
                    "profile_id": (
                        (data.get("optimization_profile") or {}).get("profile_id")
                        if isinstance(data.get("optimization_profile"), dict)
                        else None
                    ),
                    "elapsed_seconds": (data.get("result") or {}).get("elapsed_seconds"),
                    "throughput": self._receipt_throughput(data),
                }
            )
        return receipts

    @staticmethod
    def _receipt_throughput(data: dict[str, Any]) -> float | None:
        result = data.get("result") or {}
        for key in ("steps_per_second", "frames_per_second", "units_per_second"):
            value = result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _promotion_gates(receipts: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [receipt for receipt in receipts if receipt.get("status") == "completed"]
        failed = [receipt for receipt in receipts if receipt.get("status") == "failed"]
        if failed and not completed:
            return {
                "status": "ineligible",
                "reason": "Only failed benchmark receipts are available.",
                "candidate": False,
            }
        if len(completed) < 2:
            return {
                "status": "benchmark_required",
                "reason": "Need at least baseline and candidate receipts before promotion.",
                "candidate": False,
            }
        throughputs = [r["throughput"] for r in completed if isinstance(r.get("throughput"), (int, float))]
        if len(throughputs) >= 2:
            baseline = min(throughputs)
            candidate = max(throughputs)
            gain = (candidate - baseline) / baseline if baseline > 0 else 0.0
            if gain >= 0.10:
                return {
                    "status": "promotion_candidate",
                    "reason": f"Best completed receipt is {gain:.1%} faster than baseline.",
                    "candidate": True,
                    "speed_gain": gain,
                }
        return {
            "status": "benchmarked_not_promotable",
            "reason": "Receipts exist, but no qualifying speed or VRAM improvement is recorded.",
            "candidate": False,
        }
