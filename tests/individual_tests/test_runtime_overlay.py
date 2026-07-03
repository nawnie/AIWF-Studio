from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aiwf.core.domain.runtime_overlay import RuntimeOverlayReceipt, RuntimeOverlayValidateRequest
from aiwf.services.runtime_overlay import RuntimeOverlayService


def _ctx(tmp_path: Path):
    return SimpleNamespace(flags=SimpleNamespace(data_dir=tmp_path))


def test_builtin_registry_contains_flux_text_encoder_overlay(tmp_path):
    service = RuntimeOverlayService(_ctx(tmp_path))
    registry = service.registry()
    ids = {overlay.id for overlay in registry.overlays}
    assert "flux1.text_encoder_3b" in ids
    flux_overlay = next(overlay for overlay in registry.overlays if overlay.id == "flux1.text_encoder_3b")
    assert "flux" in flux_overlay.families
    assert "before_prompt_encode" in flux_overlay.phases
    assert flux_overlay.memory_lease.cpu_ram_mb >= 4096


def test_validate_orders_and_accumulates_overlay_memory(tmp_path):
    service = RuntimeOverlayService(_ctx(tmp_path))
    result = service.validate(
        RuntimeOverlayValidateRequest(
            model_family="flux",
            overlays=["diffusers.runtime_lora_adapter", "flux1.text_encoder_3b"],
            lora_count=1,
        )
    )
    assert result.valid
    assert result.plan.ordered_overlay_ids[0] == "flux1.text_encoder_3b"
    assert result.plan.memory_lease.cpu_ram_mb >= 4096
    assert result.plan.transaction_required


def test_validate_blocks_wrong_family(tmp_path):
    service = RuntimeOverlayService(_ctx(tmp_path))
    result = service.validate(
        RuntimeOverlayValidateRequest(model_family="sdxl", overlays=["flux1.text_encoder_3b"])
    )
    assert not result.valid
    assert result.blocked
    assert result.errors


def test_receipt_roundtrip(tmp_path):
    service = RuntimeOverlayService(_ctx(tmp_path))
    receipt = service.write_receipt(
        RuntimeOverlayReceipt(
            job_id="job-1",
            model_family="flux",
            model_id="flux-dev",
            overlays=[{"id": "flux1.text_encoder_3b"}],
        )
    )
    rows = service.receipts()
    assert rows
    assert rows[0].id == receipt.id
    assert rows[0].overlays[0]["id"] == "flux1.text_encoder_3b"
