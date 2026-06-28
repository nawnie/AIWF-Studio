# LoRA Pipeline Strategy

AIWF should not grow one-off LoRA code in every UI surface. The intelligent path
is a small shared LoRA planning layer plus pipeline-specific appliers.

## Current State

- SD/SDXL/SD3.5-style Diffusers image pipelines use `aiwf.infrastructure.diffusers.extra_networks.apply_loras`.
- The shared Diffusers helper resolves LoRA files, checks base architecture compatibility, loads adapters, sets adapter weights, and caches a LoRA signature on the pipe.
- Wan uses separate transformer-stage LoRA handling in `aiwf.infrastructure.wan.pipeline` because LoRAs attach to high/low or 5B transformer modules, not the same whole-pipeline adapter surface as SD/SDXL.
- Wan UI and preflight filter 5B vs 14B/A14B LoRAs so obviously wrong files are not offered for the selected runtime.
- Flux/new transformer-image LoRA application is intentionally blocked in the image backend until tested.
- ONNX LoRA support is not implemented because it needs adapter merge/export handling.
- A model-ops worker can fuse multiple LoRAs into a Diffusers output folder for supported base families.

## Recommended Architecture

Use two layers:

1. `LoRAPlan`: a backend-neutral description of selected adapters.
2. Pipeline appliers: small implementations that know how to attach that plan to a specific runtime.

Suggested plan shape:

```python
@dataclass(frozen=True)
class LoRAPlanItem:
    id: str
    path: str
    weight: float
    target_family: str
    stage: str = "default"  # default | high | low | transformer | text_encoder

@dataclass(frozen=True)
class LoRAPlan:
    base_architecture: str
    runtime_family: str
    items: tuple[LoRAPlanItem, ...]
    signature: str
```

The plan builder should:

- resolve user selections against the model catalog
- inspect metadata/header architecture when available
- reject incompatible architecture/runtime pairs before loading a pipeline
- normalize stage names for Wan and future transformer pipelines
- produce a stable signature for cache keys
- return actionable user errors instead of low-level Diffusers exceptions

The applier interface can stay small:

```python
class LoRAApplier(Protocol):
    def supports(self, runtime_family: str, base_architecture: str) -> bool: ...
    def apply(self, target, plan: LoRAPlan) -> list[str]: ...
    def clear(self, target) -> None: ...
```

## Why Not One Universal Script?

A single universal LoRA loader is risky because LoRA attachment points differ:

- SD/SDXL adapters usually load through `pipe.load_lora_weights(...)` and `pipe.set_adapters(...)`.
- Wan applies adapters to transformer modules and may need separate high/low stage handling.
- Flux, Qwen Image, SANA, and other transformer-image families may use different key prefixes and target modules.
- ONNX cannot hotswap standard Diffusers adapters without merge/export work.
- Fused LoRA export is a model operation, not the same thing as runtime LoRA hotswap.

The reusable part is not the low-level load call. The reusable part is selection,
compatibility, metadata, cache signatures, receipts, and user-facing diagnostics.

## Near-Term Work

- Extract a shared `LoRAPlan` builder from the current prompt parsing and catalog resolution.
- Keep `extra_networks.apply_loras` as the SD/SDXL/SD3.5 applier.
- Wrap Wan stage LoRA selection in the same plan shape while leaving the stage-specific applier in Wan.
- Add tests that a plan rejects Flux LoRA requests until Flux support lands.
- Add tests that cache keys include LoRA path, stage, and weight.
- Add receipts listing resolved LoRA path, stage, weight, and detected architecture.

## Success Criteria

- UI surfaces only collect LoRA selections; they do not know pipeline internals.
- Backend preflight catches mismatches before VRAM-heavy loading.
- New pipelines add a focused applier instead of modifying every UI.
- Receipts make it clear which LoRAs actually affected a run.
- Unsupported families fail with clear messages rather than partial output or silent no-op behavior.
