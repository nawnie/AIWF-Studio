# Research Brief

## Direct Answer

Krea2 is already running through the Turbo model. The slow path was placement: the lowest-memory sequential CPU-offload mode proves compatibility but pays transfer cost every step, so its warm run is still slow. For this workstation the practical ladder is:

1. Low VRAM: sequential CPU offload. Safest fallback, about 34.05 seconds warm at 512x512/8 steps, about 1.29 GiB peak PyTorch allocation.
2. Normal/Mid VRAM: streamed transformer group offload with the VAE resident on GPU. About 29.37 seconds warm at 512x512/8 steps, about 5.68 GiB observed by nvidia-smi. At 768x768 it completed with a 34.20 second warm run and about 8.02 GiB observed, making it the reliable larger-image profile for this pass.
3. High VRAM: FP8 layerwise transformer storage resident on GPU, VAE resident, and text encoding kept on CPU with prompt-cache reuse. About 8.11 seconds warm at 512x512/8 steps, about 14.69 GiB observed by nvidia-smi. A 768x768 high-profile repeat smoke ran into the card's headroom limit and was stopped after the 5-minute timeout, so high mode is a fast 512-class profile, not the large-image default yet.

## Source Map

- Tier 0 local evidence: installed runtime inspection, component-size scan, and three AIWF smoke receipts.
- Tier 1 primary/official evidence: Krea Hugging Face model card, Diffusers Krea2 docs, Diffusers memory/offload docs, and the official Krea GitHub repo.

## Key Findings

- The transformer is the main resource problem, not the VAE. The local model snapshot has about 24.48 GiB of transformer weights, about 8.27 GiB of text encoder weights, and about 0.47 GiB of VAE weights.
- Diffusers supports the exact levers we need locally: group offload, sequential/model offload, layerwise casting, and LoRA adapter loading.
- Mid should be faster than low once warmed because it reduces CPU/GPU transfers by keeping the VAE resident and streaming transformer groups. The measured gain is modest at 512 because the denoiser still moves by group each step.
- High is the profile that changes 512-class warm-run behavior meaningfully. Keeping the transformer resident in FP8 removes most per-step transfer cost; CPU prompt encoding plus the prompt cache avoids trying to fit the text encoder at the same time. At 768x768, this profile hit the card's headroom limit, while mid completed reliably.
- Custom adapters are the promising customization route. Krea's official repo says LoRAs trained on Raw apply to Turbo. Arbitrary VAE or text encoder swaps are not the first speed lever and are more likely to break pipeline contracts.

## Contradictions

There is no single best placement. The slowest mode is the best compatibility mode, mid is the best consumer default and reliable 768 path, and high is the fastest 512-class warm-run mode tested on this 16 GB card. The failed 768 high smoke means larger images need mid/group offload or another memory-saving loader.

## Rejected Sources

Generic blogs, copied notebooks, random forks, and community anecdotes were not used for final claims.

## Open Questions

- High profile needs a second memory-saving adjustment before 768x768 and 1024x1024 can be treated as large-image defaults.
- Split-file or quantized loader support for external FP8/NVFP4/GGUF assets is still future work.

## Next Steps

Run the same three-profile smoke at 768 and 1024, then wire the fastest passing profile into the Pro defaults with an obvious high-VRAM toggle.
