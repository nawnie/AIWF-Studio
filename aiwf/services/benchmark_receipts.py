from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiwf import __version__
from aiwf.core.domain.optimization import BenchmarkReceipt, CapabilityReport, GpuCapability, OptimizationPlan


class BenchmarkReceiptService:
    """Persist benchmark and generation receipts for optimization decisions."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def receipt_dir(self, created_at: datetime | None = None) -> Path:
        stamp = (created_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        return self.root / "benchmarks" / "receipts" / stamp

    def write(self, receipt: BenchmarkReceipt) -> Path:
        path = self.receipt_dir() / f"{receipt.receipt_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def build_receipt(
        self,
        *,
        plan: OptimizationPlan,
        capability_report: CapabilityReport | None = None,
        model: dict[str, Any] | None = None,
        pipeline: dict[str, Any] | None = None,
        generation: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> BenchmarkReceipt:
        created_at = datetime.now(timezone.utc).isoformat()
        dependencies = capability_report.packages if capability_report else {}
        gpu = capability_report.gpu if capability_report else None
        payload = {
            "created_at": created_at,
            "app_version": __version__,
            "profile_id": plan.profile_id,
            "model": model or {},
            "pipeline": pipeline or {},
            "generation": generation or {},
        }
        receipt_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20]
        return BenchmarkReceipt(
            receipt_id=receipt_id,
            created_at=created_at,
            aiwf={"app_version": __version__, "profile_registry_version": plan.effective_profile.profile_version},
            system={
                "capability_report_id": capability_report.report_id if capability_report else None,
            },
            dependencies=dict(dependencies),
            gpu=gpu or GpuCapability(),
            model=model or {},
            pipeline=pipeline or {},
            optimization_profile=plan.effective_profile.model_dump(mode="json"),
            generation=generation or {},
            status=status,
        )
