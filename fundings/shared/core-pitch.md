# Core Pitch

Use this as reusable source copy. Shorten it for forms.

## One-Liner

AIWF Studio is an open-source, local-first AI workspace that lets individuals run image generation, video generation, chat, model operations, and training on their own hardware instead of depending on centralized cloud platforms.

## Short Pitch

AIWF Studio started as a clean-room rebuild of a familiar Stable Diffusion web UI, but it has grown into local AI infrastructure. The project keeps the user-facing experience simple while separating heavy capabilities into explicit services, backend adapters, and supervised workers.

The goal is to make consumer hardware useful for serious creative AI workflows: image generation, inpainting, ControlNet-style guidance, Wan video generation, local chat, model downloads/metadata, model conversion and quantization experiments, Kohya-style LoRA training, and EveryDream2 full fine-tuning.

The architecture avoids the failure modes common in older local AI tools: no global `shared` object, no hidden monkey-patching as the extension model, no UI callbacks directly doing heavy backend work, and no mandatory startup dependency on optional engines. Requests flow through typed domain models and services, with GPU-heavy work coordinated through tenant supervision and process workers.

## Why It Matters

Most AI tooling pushes users toward cloud services, rented GPUs, and centralized platforms. AIWF Studio takes the opposite route: it treats the user's own machine as the primary compute platform. That improves privacy, resilience, inspectability, and long-term user control.

## Evidence To Cite

- Public repo: https://github.com/nawnie/AIWF-Studio
- Local test collection: 715 tests collected in the project venv.
- Public GitHub test collection: 289 tests collected from main.
- Local-only additions include worker protocol, GPU tenant supervision, chat, training, Wan/GGUF video work, ONNX/Comfy scaffolding, quantization/model ops, and expanded documentation.

## Funding Ask Template

I am requesting non-dilutive funding to spend focused development time on:

1. Hardening the worker/engine boundary so video, training, chat, and image generation can coexist safely.
2. Publishing the local v2 work to GitHub with documentation, tests, and demos.
3. Improving consumer-GPU workflows for Wan/GGUF video and safetensors/quantized model paths.
4. Making local training setup safer through validation, config generation, and engine isolation.

## 30-Second Founder Bio Placeholder

Shawn is an independent builder focused on practical local AI infrastructure. AIWF Studio is built from direct implementation work rather than a pitch deck: typed Python architecture, Gradio/FastAPI UI/API layers, model/runtime services, tests, and local GPU workflows.

Replace this with a stronger personal version before submitting.

