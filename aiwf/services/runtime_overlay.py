from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aiwf.core.domain.runtime_overlay import (
    RuntimeOverlayContract,
    RuntimeOverlayMemoryLease,
    RuntimeOverlayReceipt,
    RuntimeOverlayRegistry,
    RuntimeOverlayValidateRequest,
    RuntimeOverlayValidationResult,
    RuntimeOverlayPlan,
)

logger = logging.getLogger(__name__)

_PLUGIN_MANIFEST_NAMES = ("aiwf-plugin.json", "plugin.json", "manifest.json")


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item) for item in value]
    return [str(value)]


def _overlay_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("id") or value.get("overlayId") or value.get("patchId") or "").strip()
    return str(value or "").strip()


def _contract_from_raw(raw: dict[str, Any], *, source: str) -> RuntimeOverlayContract | None:
    payload = dict(raw)
    payload.setdefault("source", source)
    payload.setdefault("id", payload.get("overlayId") or payload.get("patchId") or payload.get("name"))
    payload.setdefault("label", payload.get("name") or payload.get("id") or "Runtime overlay")
    payload["families"] = _normalize_list(payload.get("families") or payload.get("family") or ["unknown"])
    payload["targets"] = _normalize_list(payload.get("targets") or payload.get("target"))
    payload["phases"] = _normalize_list(payload.get("phases") or payload.get("phase"))
    payload["inputs"] = _normalize_list(payload.get("inputs"))
    payload["produces"] = _normalize_list(payload.get("produces") or payload.get("outputs"))
    try:
        return RuntimeOverlayContract.model_validate(payload)
    except Exception:
        logger.warning("Skipping invalid runtime overlay contract from %s", source, exc_info=True)
        return None


def builtin_runtime_overlay_contracts() -> list[RuntimeOverlayContract]:
    """Built-in overlay contracts.

    The Flux.1 3B text-encoder entry documents the runtime overlay pattern AIWF
    already started: swap or select a lighter prompt encoder without forking the
    whole model family path.
    """

    return [
        RuntimeOverlayContract(
            id="flux1.text_encoder_3b",
            label="Flux.1 3B text encoder overlay",
            status="started",
            source="aiwf-core",
            families=["flux"],
            targets=["text_encoder_2", "prompt_encoder"],
            phases=["before_model_load", "before_prompt_encode", "after_prompt_encode", "receipt_write"],
            inputs=["prompt", "model"],
            produces=["conditioning", "metadata"],
            changes_pixels=True,
            requires_gpu=False,
            safe_with_compile=True,
            safe_with_controlnet=True,
            safe_with_lora=True,
            memory_lease=RuntimeOverlayMemoryLease(
                vram_mb=0,
                cpu_ram_mb=4096,
                ssd_cache_mb=0,
                policy="family-default-user-overridable",
            ),
            summary=(
                "Existing AIWF pattern: route Flux.1 prompt encoding through a lighter 3B text encoder "
                "as a reversible runtime overlay with explicit prompt-cache, dtype, and offload receipts."
            ),
            receipt_fields=["overlay_id", "text_encoder_id", "text_encoder_dtype", "prompt_cache_key", "offload_policy"],
        ),
        RuntimeOverlayContract(
            id="diffusers.runtime_lora_adapter",
            label="Diffusers runtime LoRA adapter",
            status="active",
            source="aiwf-core",
            families=["sd15", "sdxl", "sd35", "flux"],
            targets=["unet", "transformer", "text_encoder"],
            phases=["after_model_load", "before_sample", "receipt_write"],
            inputs=["model", "prompt"],
            produces=["model", "metadata"],
            changes_pixels=True,
            requires_gpu=True,
            safe_with_compile=False,
            safe_with_controlnet=True,
            safe_with_lora=True,
            memory_lease=RuntimeOverlayMemoryLease(vram_mb=256, cpu_ram_mb=512, policy="sum-active-adapters"),
            summary="Runtime adapter loading for supported Diffusers image families, captured as an overlay transaction.",
            receipt_fields=["lora_id", "lora_scale", "adapter_target", "adapter_hash"],
        ),
        RuntimeOverlayContract(
            id="controlnet.conditioning_sidecar",
            label="ControlNet conditioning sidecar",
            status="adapter-ready",
            source="aiwf-core",
            families=["sd15", "sdxl"],
            targets=["controlnet", "conditioning", "unet"],
            phases=["before_sample", "during_sample_cfg", "receipt_write"],
            inputs=["image", "conditioning"],
            produces=["conditioning", "metadata"],
            changes_pixels=True,
            requires_gpu=True,
            safe_with_compile=False,
            safe_with_controlnet=True,
            safe_with_lora=True,
            memory_lease=RuntimeOverlayMemoryLease(vram_mb=1024, cpu_ram_mb=512, policy="per-control-unit"),
            summary="Control image/preprocessor outputs become declared conditioning overlays instead of implicit sampler mutation.",
            receipt_fields=["control_model_id", "control_module", "weight", "guidance_start", "guidance_end"],
        ),
        RuntimeOverlayContract(
            id="freeu.unet_output_blocks",
            label="FreeU UNet output block overlay",
            status="candidate",
            source="community-contract",
            families=["sd15", "sdxl"],
            targets=["unet.output_blocks"],
            phases=["before_sample", "receipt_write"],
            inputs=["latent", "model"],
            produces=["latent", "metadata"],
            changes_pixels=True,
            requires_gpu=True,
            safe_with_compile=False,
            safe_with_controlnet=True,
            safe_with_lora=True,
            memory_lease=RuntimeOverlayMemoryLease(vram_mb=128, policy="small-patch"),
            summary="Forge-like FreeU behavior expressed as an AIWF-declared overlay with deterministic order and rollback.",
            receipt_fields=["b1", "b2", "s1", "s2"],
        ),
        RuntimeOverlayContract(
            id="receipt.overlay_writer",
            label="Overlay receipt writer",
            status="active",
            source="aiwf-core",
            families=["all"],
            targets=["metadata"],
            phases=["receipt_write"],
            inputs=["metadata"],
            produces=["metadata", "artifact"],
            changes_pixels=False,
            requires_gpu=False,
            safe_with_compile=True,
            safe_with_controlnet=True,
            safe_with_lora=True,
            memory_lease=RuntimeOverlayMemoryLease(ssd_cache_mb=1, policy="sidecar"),
            summary="Structured record of which reversible overlays affected an output.",
            receipt_fields=["ordered_overlays", "memory_lease", "rollback_status", "compatibility_warnings"],
        ),
    ]


class RuntimeOverlayService:
    """Registry, dry-run validation, and receipt ledger for model overlays.

    This intentionally stops short of executing arbitrary plugin code. Execution
    should be wired per model-family adapter through reversible transactions.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    @property
    def data_dir(self) -> Path:
        flags = getattr(self.ctx, "flags", None)
        return Path(getattr(flags, "data_dir", Path.cwd())).resolve()

    @property
    def runtime_dir(self) -> Path:
        path = self.data_dir / "runtime_overlays"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def receipts_path(self) -> Path:
        return self.runtime_dir / "runtime_overlay_receipts.jsonl"

    def registry(self) -> RuntimeOverlayRegistry:
        return RuntimeOverlayRegistry(overlays=[*builtin_runtime_overlay_contracts(), *self._plugin_contracts()])

    def validate(self, request: RuntimeOverlayValidateRequest) -> RuntimeOverlayValidationResult:
        registry = self.registry()
        overlay_map = {contract.id: contract for contract in registry.overlays}
        requested_ids = [_overlay_id(item) for item in request.overlays]
        requested_ids = [item for item in requested_ids if item]
        family = (request.model_family or "unknown").lower()
        errors: list[str] = []
        warnings: list[str] = []
        ordered: list[RuntimeOverlayContract] = []
        memory = RuntimeOverlayMemoryLease()

        for overlay_id in requested_ids:
            contract = overlay_map.get(overlay_id)
            if contract is None:
                errors.append(f"Unknown runtime overlay: {overlay_id}")
                continue
            families = [item.lower() for item in contract.families]
            if families and "all" not in families and family not in families:
                errors.append(f"{overlay_id} is not declared compatible with model family '{family}'.")
            if request.compile_enabled and not contract.safe_with_compile:
                warnings.append(f"{overlay_id} should disable or invalidate compile caches for this run.")
            if request.controlnet_enabled and not contract.safe_with_controlnet:
                errors.append(f"{overlay_id} is not safe with ControlNet in its current contract.")
            if request.lora_count > 0 and not contract.safe_with_lora:
                errors.append(f"{overlay_id} is not safe with active LoRA adapters.")
            memory = memory.add(contract.memory_lease)
            ordered.append(contract)

        ordered.sort(key=lambda item: (item.phase_index, item.id))
        if not requested_ids:
            warnings.append("No overlays requested; base pipeline remains unmodified.")
        if request.requested_memory_mb and memory.vram_mb > request.requested_memory_mb:
            warnings.append(
                f"Overlay VRAM lease asks for {memory.vram_mb} MB, above requested budget {request.requested_memory_mb} MB."
            )

        plan = RuntimeOverlayPlan(
            model_family=family,
            pipeline_kind=request.pipeline_kind,
            ordered_overlay_ids=[item.id for item in ordered],
            memory_lease=memory,
            transaction_required=bool(ordered),
            receipt_required=bool(ordered),
            rollback_required=bool(ordered),
        )
        return RuntimeOverlayValidationResult(
            valid=not errors,
            blocked=bool(errors),
            errors=errors,
            warnings=warnings,
            plan=plan,
            ordered_overlays=ordered,
        )

    def receipts(self, *, limit: int = 80) -> list[RuntimeOverlayReceipt]:
        if not self.receipts_path.is_file():
            return []
        try:
            lines = self.receipts_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        receipts: list[RuntimeOverlayReceipt] = []
        for line in reversed(lines[-limit:]):
            try:
                raw = json.loads(line)
                receipts.append(RuntimeOverlayReceipt.model_validate(raw))
            except Exception:
                logger.debug("Skipping invalid runtime overlay receipt", exc_info=True)
        return receipts

    def write_receipt(self, receipt: RuntimeOverlayReceipt) -> RuntimeOverlayReceipt:
        self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.receipts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt.model_dump(mode="json", by_alias=True), sort_keys=True) + "\n")
        return receipt

    def _plugin_contracts(self) -> list[RuntimeOverlayContract]:
        plugins_dir = self.data_dir / "plugins"
        if not plugins_dir.is_dir():
            return []
        contracts: list[RuntimeOverlayContract] = []
        for plugin_root in sorted(path for path in plugins_dir.iterdir() if path.is_dir()):
            manifest = self._read_plugin_manifest(plugin_root)
            if not manifest:
                continue
            if manifest.get("enabled") is False:
                continue
            plugin_id = str(manifest.get("id") or plugin_root.name)
            raw_contracts = (
                manifest.get("runtimeOverlays")
                or manifest.get("runtime_overlays")
                or manifest.get("modelPatches")
                or manifest.get("model_patches")
                or manifest.get("patches")
                or []
            )
            if not isinstance(raw_contracts, list):
                continue
            for raw in raw_contracts:
                if not isinstance(raw, dict):
                    continue
                contract = _contract_from_raw(raw, source=plugin_id)
                if contract is not None:
                    contracts.append(contract)
        return contracts

    @staticmethod
    def _read_plugin_manifest(plugin_root: Path) -> dict[str, Any] | None:
        for name in _PLUGIN_MANIFEST_NAMES:
            path = plugin_root / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.debug("Could not read plugin manifest %s", path, exc_info=True)
                continue
            if isinstance(payload, dict):
                return payload
        return None
