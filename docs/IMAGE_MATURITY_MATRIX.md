# Image Maturity Bridge

Target: bring every native AIWF core image route to at least `8.0` maturity against the AUTOMATIC1111 core workflow baseline.

## Scope

- Native AIWF features first. Do not make A1111 the runtime dependency.
- A1111 is the parity reference for common image workflows: txt2img, img2img, inpaint, batch, XYZ, hires/refiner, ControlNet, extras, PNG/API replay.
- Flux stays txt2img-only for this release pass. Flux img2img, inpaint, LoRA, hires, VAE, refiner, and ControlNet remain gated for Step 3+.
- No speed or quality claim is valid without a benchmark receipt.

## Route Matrix

| Route | Target | Current | Benchmark kind | Status |
| --- | ---: | ---: | --- | --- |
| Text to image | 8.0 | 8.3 | `txt2img` | Ready |
| Image to image | 8.0 | 8.0 | `img2img` | Ready |
| Inpaint / masked repair | 8.0 | 8.0 | `inpaint` | Ready |
| Hires fix / SDXL refiner | 8.0 | 8.0 | `hires` | Ready |
| ControlNet conditioning | 8.0 | 8.1 | `controlnet` | Ready |
| XYZ plots | 8.0 | 8.0 | `txt2img` | Ready |
| Extras / enhance | 8.0 | 8.0 | Receipt-backed | Ready |
| Segment to inpaint | 8.0 | 8.0 | Receipt-backed | Ready |
| PNG/API replay | 8.0 | 8.0 | `txt2img` | Ready |
| Flux text to image | 8.0 | 8.0 | `txt2img` | Ready |

## Implemented Bridge

- `Image Lab` tab:
  - maturity matrix
  - XYZ plot runner
  - batch img2img/inpaint runner
  - loopback runner
- Native API:
  - `GET /api/v1/image/maturity`
- Benchmark worker:
  - `probe`
  - `txt2img`
  - `img2img`
  - `inpaint`
  - `controlnet`
  - `hires`
  - `wan_i2v`
- Enhance outputs:
  - image-sidecar receipt JSON for upscale, restore, photo restore, and full enhance pipeline runs
- Segment to inpaint:
  - Object Replace preset runs Auto mask into Inpaint and saves a workflow receipt with source, mask, output, and repair settings
- Flux text to image:
  - `scripts/smoke_image_routes.py --plan-json` emits the bounded txt2img smoke plan for Flux, Flux.2 Klein, and Z-Image routes without loading models

## Benchmark Config Examples

```json
{
  "kind": "txt2img",
  "request": {
    "prompt": "studio portrait",
    "steps": 20,
    "width": 512,
    "height": 512,
    "seed": 123
  }
}
```

```json
{
  "kind": "inpaint",
  "init_image": "F:/path/source.png",
  "mask_image": "F:/path/mask.png",
  "request": {
    "prompt": "repair the damaged area",
    "steps": 20,
    "width": 512,
    "height": 512,
    "denoising_strength": 0.75
  }
}
```

```json
{
  "kind": "controlnet",
  "control_image": "F:/path/control.png",
  "request": {
    "prompt": "architectural render",
    "steps": 20,
    "width": 512,
    "height": 512,
    "controlnet_units": [
      {
        "enabled": true,
        "model": "control-model-id",
        "module": "canny",
        "weight": 1.0
      }
    ]
  }
}
```
