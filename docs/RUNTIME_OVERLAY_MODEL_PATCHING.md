# Runtime Overlay Model Patching

AIWF Runtime Overlay is the model-patching contract for reversible runtime changes to active image, video, and audio models.

It is inspired by the useful part of Forge's model patching idea, but it is not a port. Forge lets extensions reach into the processing object, clone patchers, mutate model options, run sampling, and restore afterward. AIWF should keep the same power while making the contract explicit, typed, permission-aware, and easy to test.

## Core rule

Runtime overlays are temporary transactions:

```text
begin -> lease -> apply -> run -> receipt -> rollback
```

A runtime overlay must declare what it touches before it can touch it.

## Why this exists

AIWF already has the seed of this system in Flux.1 prompt-encoder handling: the app can patch in or route through a lighter 3B text encoder for prompt encoding without forking the whole Flux pipeline path. Runtime Overlay turns that pattern into a formal architecture so future model patches do not become one-off backend branches.

## Built-in overlay contracts

The initial registry includes contracts for:

- `flux1.text_encoder_3b`
- `diffusers.runtime_lora_adapter`
- `controlnet.conditioning_sidecar`
- `freeu.unet_output_blocks`
- `receipt.overlay_writer`

These are contracts first. Engine execution still belongs inside model-family adapters.

## Patch points

Runtime overlays may target declared patch points:

```text
before_prompt_expand
before_prompt_encode
after_prompt_encode
before_model_load
after_model_load
before_sample
during_sample_pre_cfg
during_sample_cfg
during_sample_post_cfg
after_sample
before_vae_decode
after_vae_decode
after_image
receipt_write
```

The order is deterministic. If two overlays share the same phase, they are sorted by id.

## Overlay manifest shape

Plugin manifests may declare overlays with `runtimeOverlays`, `runtime_overlays`, `modelPatches`, `model_patches`, or `patches`.

```json
{
  "id": "example.freeu",
  "name": "FreeU Example",
  "runtimeOverlays": [
    {
      "id": "freeu.unet_output_blocks",
      "label": "FreeU UNet output block overlay",
      "families": ["sd15", "sdxl"],
      "targets": ["unet.output_blocks"],
      "phases": ["before_sample", "receipt_write"],
      "inputs": ["latent", "model"],
      "produces": ["latent", "metadata"],
      "changesPixels": true,
      "requiresGpu": true,
      "safeWithCompile": false,
      "safeWithControlNet": true,
      "safeWithLoRA": true,
      "memoryLease": {
        "vramMb": 128,
        "cpuRamMb": 0,
        "ssdCacheMb": 0,
        "policy": "small-patch"
      },
      "receiptFields": ["b1", "b2", "s1", "s2"]
    }
  ]
}
```

## API

AIWF Studio Pro exposes a dry-run registry and ledger:

```http
GET  /api/pro/runtime-overlays/registry
POST /api/pro/runtime-overlays/validate
GET  /api/pro/runtime-overlays/receipts
POST /api/pro/runtime-overlays/receipts
```

`validate` does not execute model code. It resolves compatibility, deterministic order, memory lease totals, compile warnings, and blocking errors.

## What makes this better than raw patching

Runtime Overlay adds:

- model-family contracts
- declared patch points
- memory leases
- deterministic ordering
- plugin manifest support
- structured receipts
- rollback requirements
- compatibility checks for compile, ControlNet, and LoRA
- a safe dry-run API before any patch touches a model

## Next implementation step

Wire the contracts into model-family adapters:

```text
ModelFamilyAdapter.resolve_runtime_overlays(request)
PatchTransaction.begin()
PatchTransaction.apply()
generate()
PatchTransaction.write_receipt()
PatchTransaction.rollback()
```

The first real adapter should be Flux.1 text-encoder overlay because the behavior already exists in the backend. Convert the current path into an explicit transaction and receipt, then repeat that pattern for LoRA, ControlNet, FreeU, SAG/PAG, IP-Adapter-style modules, and video-specific transformer overlays.
