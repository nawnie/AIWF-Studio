# CUDA 13 Runtime Attention Experiment

This note keeps the routing picture in the repo. The venv setup, package logs, benchmark receipts, and image outputs live in the local experiment folder and research receipt from the July 2026 pass.

- Local experiment folder: `F:\AIWF_Studio_attention_opt_bench_20260708-134009`
- Research receipt: `research runs/20260708-134009-cuda13-sageattention-xformers-aiwf-bench`
- Main Studio choice after the pass: Python 3.12, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, Diffusers 0.39.0, SDPA default.

```mermaid
flowchart TD
    A[Studio launch] --> B[Root venv]
    B --> C[Python 3.12]
    C --> D[torch 2.13.0 cu130]
    D --> E[Default attention: SDPA]

    E --> F[SD 1.x / SDXL]
    F --> F1[AttnProcessor2_0 + torch SDPA]

    E --> G[Flux / Flux.2 / Z-Image]
    G --> G1[Keep native transformer processors]
    G1 --> G2[Use Diffusers set_attention_backend only when exposed]

    E --> H[Qwen Image / Sana image]
    H --> H1[Keep native Diffusers transformer path]

    E --> I[Wan / Sana video]
    I --> I1[Service-specific attention setup]
    I1 --> I2[Sage only when route-specific smoke passes]
    I1 --> I3[Fallback: torch SDPA]

    J[xFormers] -. opt-in only .-> F
    K[SageAttention] -. opt-in only .-> I
    K -. do not global-patch .-> G
```

Keep this rule simple: SDPA is the default image runtime. SageAttention and xFormers need a route-specific receipt before they become a default for any family.
