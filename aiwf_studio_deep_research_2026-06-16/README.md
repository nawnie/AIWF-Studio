# AIWF Studio Diffusers Optimization Research Package

**Prepared:** 2026-06-16
**Scope:** local-first Stable Diffusion / Diffusers performance, memory, stability, and quality strategy for AIWF Studio.

## Package contents

| File | Purpose |
|---|---|
| [`aiwf_studio_diffusers_optimization_report.md`](aiwf_studio_diffusers_optimization_report.md) | Main research report with executive summary, findings, architecture, risks, and roadmap. |
| [`compatibility_matrix.md`](compatibility_matrix.md) | Human-readable optimization compatibility matrix and denylist rules. |
| [`compatibility_matrix.csv`](compatibility_matrix.csv) | Machine-readable version of the compatibility matrix. |
| [`default_generation_profiles.md`](default_generation_profiles.md) | Recommended Safe/Balanced/Quality/Low VRAM/Fast Mode profiles. |
| [`experimental_feature_flags.md`](experimental_feature_flags.md) | Proposed feature flags, predicates, conflicts, and receipt fields. |
| [`benchmark_protocol.md`](benchmark_protocol.md) | Practical benchmark plan for speed, memory, stability, and quality. |
| [`benchmark_receipt_schema.json`](benchmark_receipt_schema.json) | JSON schema for benchmark receipts. |
| [`benchmark_prompts.json`](benchmark_prompts.json) | Initial test prompt/settings corpus. |
| [`optimization_profile_schema.json`](optimization_profile_schema.json) | Draft JSON schema for AIWF optimization profiles. |
| [`risks_and_mitigations.md`](risks_and_mitigations.md) | Risk register with mitigation patterns. |
| [`roadmap.md`](roadmap.md) | Prioritized â€œDo now / experiment / research / avoidâ€ roadmap. |
| [`research.md`](research.md) | Follow-up research map and experiment queue. |
| [`sources.md`](sources.md) | Source list with links and source-key mapping. |

## Core conclusion

AIWF Studio should implement an optimization substrate before chasing aggressive accelerators. The recommended path is:

1. Safe/Balanced default profiles using PyTorch SDPA, fp16, scheduler recipes, and optional channels-last.
2. Explicit Low VRAM policy using VAE slicing/tiling and model offload.
3. Prompt, LoRA, inpaint, ControlNet, and hires workflows as first-class services.
4. Benchmark receipts for every optimization candidate.
5. xFormers, compile, FreeU/PAG, LCM/Lightning, TensorRT, torchao, and ModelOpt behind feature flags until receipts prove safety.

## How to use this package

Start with the main report, then use:

- `roadmap.md` for implementation order.
- `compatibility_matrix.csv` to seed an internal compatibility registry.
- `optimization_profile_schema.json` and `benchmark_receipt_schema.json` as draft interfaces.
- `benchmark_protocol.md` before promoting any optimization to default.

## Compatibility stance

This package assumes:
- `transformers>=4.44,<5` remains the mainline constraint.
- Windows + NVIDIA consumer GPUs are supported.
- Optional heavy dependencies must not be required at boot.
- Quality-changing features must be user-visible.
- Fast/distilled methods are separate Fast Mode recipes, not hidden optimizations.
