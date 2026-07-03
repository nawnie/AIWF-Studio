# Model Family Attention Policy

This file is release guidance for agents changing image runtime code. Treat it as a guardrail: attention optimizations can change output quality, not only speed or VRAM.

## Core Rule

Do not apply one global attention patch to every model family. Pick the attention path by denoiser family and verify with at least one saved image smoke when the change touches generation.

## Family Policy

- SD 1.x, SD 2.x, SDXL, and SDXL inpaint: use Diffusers `AttnProcessor2_0` / torch SDPA. Do not use the global SageAttention `scaled_dot_product_attention` shim for UNet pipelines. The shim can receive SDPA tensors in torch layout and silently scramble denoise output into static-like images.
- SD3 and SD3.5: keep the FlowMatch/DiT native attention path. Do not apply SD/SDXL UNet processor assumptions, and do not allow SD1/SDXL external VAE selection on SD3 pipelines.
- Flux, Flux Kontext, Flux.2 Klein, and Z-Image: preserve native transformer attention processors. Prefer Diffusers `set_attention_backend` when the transformer exposes it; otherwise use the existing native SDPA/Sage fallback code in `aiwf.infrastructure.torch.attention`.
- Qwen Image and Sana image: keep the model's Diffusers scheduler and native transformer attention behavior unless a family-specific smoke proves a faster backend is safe.
- Wan and Sana video: use their service-specific attention setup. Do not route video attention through the SD/SDXL image UNet policy.

## Regression Checks

- If SDXL output looks like colorful cable-static or temporal noise, check attention before changing VAE files. In the July 2026 regression, baked VAE, no VAE tiling/slicing, and explicit `sdxl_vae` all produced the same bad image; switching from the global SageAttention shim to torch SDPA fixed it.
- For SD/SDXL attention changes, compare AIWF service output against a direct Diffusers control with the same checkpoint, prompt, seed, sampler, steps, CFG, width, and height.
- Keep `tests/individual_tests/test_torch_attention.py` covering the invariant that UNet pipelines skip the global SageAttention call shim even when the launch profile says `sage_sdpa`.

## Code Pointers

- Runtime attention policy: `aiwf/infrastructure/torch/attention.py`
- Diffusers call wrapper: `aiwf/infrastructure/diffusers/backend.py::_call_pipe`
- Model-family presets: `aiwf/infrastructure/diffusers/model_presets.py`
