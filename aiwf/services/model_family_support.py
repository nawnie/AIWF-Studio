from __future__ import annotations

"""Source-of-truth model family support matrix for AIWF Studio.

This module is intentionally import-light. It describes what the codebase can
currently recognize, load, smoke, and must still block. Runtime readiness is
merged in at request time from the existing pipeline readiness ledger, but the
family/precision map below is grounded in the loader/preflight modules rather
than README claims.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from aiwf.core.config.settings import RuntimeFlags, UserSettings


_PRECISION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:^|[_\-.])iq4_(?:nl|xs)(?:[_\-.]|$)", "IQ4"),
    (r"(?:^|[_\-.])iq3_(?:xxs|xs|s|m)(?:[_\-.]|$)", "IQ3"),
    (r"(?:^|[_\-.])iq2_(?:xxs|xs|s|m)(?:[_\-.]|$)", "IQ2"),
    (r"(?:^|[_\-.])q2[_\-.]?k(?:[_\-.]|$)", "Q2_K"),
    (r"(?:^|[_\-.])q3[_\-.]?k[_\-.]?s(?:[_\-.]|$)", "Q3_K_S"),
    (r"(?:^|[_\-.])q3[_\-.]?k[_\-.]?m(?:[_\-.]|$)", "Q3_K_M"),
    (r"(?:^|[_\-.])q3[_\-.]?k[_\-.]?l(?:[_\-.]|$)", "Q3_K_L"),
    (r"(?:^|[_\-.])q4[_\-.]?k[_\-.]?s(?:[_\-.]|$)", "Q4_K_S"),
    (r"(?:^|[_\-.])q4[_\-.]?k[_\-.]?m(?:[_\-.]|$)", "Q4_K_M"),
    (r"(?:^|[_\-.])q4[_\-.]?0(?:[_\-.]|$)", "Q4_0"),
    (r"(?:^|[_\-.])q4[_\-.]?1(?:[_\-.]|$)", "Q4_1"),
    (r"(?:^|[_\-.])q5[_\-.]?k[_\-.]?s(?:[_\-.]|$)", "Q5_K_S"),
    (r"(?:^|[_\-.])q5[_\-.]?k[_\-.]?m(?:[_\-.]|$)", "Q5_K_M"),
    (r"(?:^|[_\-.])q5[_\-.]?0(?:[_\-.]|$)", "Q5_0"),
    (r"(?:^|[_\-.])q6[_\-.]?k(?:[_\-.]|$)", "Q6_K"),
    (r"(?:^|[_\-.])q8[_\-.]?0(?:[_\-.]|$)", "Q8_0"),
    (r"(?:^|[_\-.])q([2-8])(?:[_\-.]|$)", "Q{group}"),
    (r"(?:^|[_\-.])(bnb[_\-.])?nf4(?:[_\-.]|$)", "NF4"),
    (r"(?:^|[_\-.])(nv)?fp4(?:[_\-.]|$)", "FP4"),
    (r"(?:^|[_\-.])int4(?:[_\-.]|$)", "INT4"),
    (r"(?:^|[_\-.])(svdq[_\-.])?int4(?:[_\-.]|$)", "INT4"),
    (r"(?:^|[_\-.])(bnb[_\-.])?int8(?:[_\-.]|$)", "INT8"),
    (r"(?:^|[_\-.])8bit(?:[_\-.]|$)", "INT8"),
    (r"(?:^|[_\-.])(float8|f8|fp8|f8_e4m3fn|e4m3)(?:[_\-.]|$)", "FP8"),
    (r"(?:^|[_\-.])(bfloat16|bf16)(?:[_\-.]|$)", "BF16"),
    (r"(?:^|[_\-.])(float16|fp16|f16|half)(?:[_\-.]|$)", "FP16"),
    (r"(?:^|[_\-.])(float32|fp32|f32)(?:[_\-.]|$)", "FP32"),
)


def normalize_precision_label(value: str | None) -> str:
    """Normalize common filename/header precision spellings for display and filtering."""
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    if not text:
        return ""
    aliases = {
        "F8": "FP8",
        "F8_E4M3": "FP8",
        "F8_E4M3FN": "FP8",
        "FLOAT8": "FP8",
        "F16": "FP16",
        "FLOAT16": "FP16",
        "HALF": "FP16",
        "B_FLOAT16": "BF16",
        "BFloat16".upper(): "BF16",
        "F32": "FP32",
        "FLOAT32": "FP32",
        "Q4KM": "Q4_K_M",
        "Q5KM": "Q5_K_M",
        "Q3KM": "Q3_K_M",
        "Q8": "Q8_0",
    }
    return aliases.get(text, text)


def detect_precision_from_name(filename: str | Path | None) -> str:
    """Return the most specific quant/precision token visible in a filename."""
    name = Path(str(filename or "")).name.lower()
    if not name:
        return ""
    padded = f"_{name}_"
    for pattern, label in _PRECISION_PATTERNS:
        match = re.search(pattern, padded, flags=re.IGNORECASE)
        if not match:
            continue
        if label == "Q{group}":
            return f"Q{match.group(1)}"
        if label == "FP4" and match.group(1):
            return "NVFP4"
        return normalize_precision_label(label)
    # Some community files concatenate tags, e.g. GGUFQ4KM, without a separator.
    compact_patterns = (
        (r"q3k[\-_]?s", "Q3_K_S"),
        (r"q3k[\-_]?m", "Q3_K_M"),
        (r"q3k[\-_]?l", "Q3_K_L"),
        (r"q4k[\-_]?s", "Q4_K_S"),
        (r"q4k[\-_]?m", "Q4_K_M"),
        (r"q5k[\-_]?s", "Q5_K_S"),
        (r"q5k[\-_]?m", "Q5_K_M"),
        (r"q6k", "Q6_K"),
        (r"q8_?0", "Q8_0"),
    )
    for pattern, label in compact_patterns:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return label
    return ""


def precision_bucket(value: str | None) -> str:
    """Coarse bucket used for compatibility checks and matrix grouping."""
    precision = normalize_precision_label(value)
    if not precision:
        return "unknown"
    if precision.startswith("Q") or precision.startswith("IQ"):
        match = re.match(r"I?Q(\d)", precision)
        return f"q{match.group(1)}" if match else precision.lower()
    if precision in {"NF4", "FP4", "NVFP4", "INT4"}:
        return "4bit"
    if precision in {"FP8", "INT8"}:
        return "8bit"
    if precision in {"BF16", "FP16", "FP32"}:
        return precision.lower()
    return precision.lower()


def supported_precision_names() -> list[str]:
    return [
        "FP32",
        "FP16",
        "BF16",
        "FP8",
        "INT8",
        "NF4",
        "FP4",
        "NVFP4",
        "INT4",
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_1",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_0",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "IQ2",
        "IQ3",
        "IQ4",
    ]


def _precision(name: str, status: str, loader: str, notes: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "loader": loader, "notes": notes}


def _route(id: str, status: str, kind: str, entrypoint: str, notes: str = "") -> dict[str, str]:
    return {"id": id, "status": status, "kind": kind, "entrypoint": entrypoint, "notes": notes}


def _family(
    *,
    id: str,
    label: str,
    category: str,
    status: str,
    summary: str,
    storage: list[str],
    precisions: list[dict[str, str]],
    routes: list[dict[str, str]],
    sidecars: list[str],
    lora: str,
    blockers: list[str],
    modules: list[str],
) -> dict[str, Any]:
    return {
        "id": id,
        "label": label,
        "category": category,
        "status": status,
        "summary": summary,
        "storage": storage,
        "precisions": precisions,
        "routes": routes,
        "sidecars": sidecars,
        "lora": lora,
        "blockers": blockers,
        "modules": modules,
    }


def static_model_family_support() -> list[dict[str, Any]]:
    """Code-grounded family map before local inventory/readiness is merged."""
    return [
        _family(
            id="sd15",
            label="Stable Diffusion 1.5",
            category="image",
            status="supported",
            summary="Classic Diffusers single-file/folder image route with txt2img, img2img, inpaint, embeddings, ControlNet, LoRA, and optional FP8 UNet storage.",
            storage=[".safetensors", ".ckpt", ".pt", "Diffusers folder"],
            precisions=[
                _precision("FP32", "runtime", "torch_dtype from DeviceManager / --no-half"),
                _precision("FP16", "supported", "StableDiffusionPipeline.from_single_file/from_pretrained"),
                _precision("FP8 storage", "optional", "UNet enable_layerwise_casting via --fp8", "storage-only VRAM saver; compute remains fp16/bf16"),
            ],
            routes=[
                _route("diffusers", "supported", "txt2img/img2img", "DiffusersBackend.load_checkpoint"),
                _route("inpaint", "supported", "inpaint", "DiffusersBackend._load_inpaint_checkpoint"),
                _route("controlnet", "supported", "conditioning", "build_controlnet_pipeline"),
            ],
            sidecars=["VAE", "LoRA", "embeddings", "ControlNet", "upscalers"],
            lora="Runtime prompt LoRA adapters are supported for classic SD routes.",
            blockers=["Quantized single-file SD checkpoints beyond FP8 storage are not a dedicated route."],
            modules=["aiwf.infrastructure.diffusers.backend", "aiwf.infrastructure.diffusers.model_arch"],
        ),
        _family(
            id="sdxl",
            label="Stable Diffusion XL",
            category="image",
            status="supported",
            summary="SDXL base/refiner/inpaint support with single-file or folder loading, dual-encoder embeddings, LoRA, ControlNet, and optional FP8 UNet storage.",
            storage=[".safetensors", ".ckpt", ".pt", "Diffusers folder"],
            precisions=[
                _precision("FP32", "runtime", "--no-half"),
                _precision("FP16", "supported", "StableDiffusionXLPipeline"),
                _precision("BF16", "runtime", "device dtype when configured/supported"),
                _precision("FP8 storage", "optional", "UNet enable_layerwise_casting via --fp8"),
            ],
            routes=[
                _route("diffusers", "supported", "txt2img/img2img", "StableDiffusionXLPipeline / StableDiffusionXLImg2ImgPipeline"),
                _route("sdxl-refiner", "supported", "refiner", "DiffusersBackend._load_refiner_checkpoint"),
                _route("inpaint", "supported", "inpaint", "StableDiffusionXLInpaintPipeline"),
            ],
            sidecars=["VAE", "LoRA", "dual CLIP embeddings", "ControlNet", "refiner"],
            lora="Runtime LoRA tags are supported; SDXL embeddings require clip_l/clip_g safetensors.",
            blockers=["SD1.5 embeddings/LoRAs are skipped or should be labeled incompatible for SDXL."],
            modules=["aiwf.infrastructure.diffusers.backend", "aiwf.infrastructure.diffusers.model_arch"],
        ),
        _family(
            id="sd35",
            label="Stable Diffusion 3.5",
            category="image",
            status="supported-gated",
            summary="SD3.5 Diffusers route with txt2img/img2img/inpaint classes and explicit gated-config access checks for Large single-file checkpoints.",
            storage=[".safetensors", "Diffusers folder"],
            precisions=[
                _precision("BF16", "preferred", "DiffusersBackend._dtype_for_architecture"),
                _precision("FP16", "fallback", "DeviceManager dtype"),
                _precision("FP32", "runtime", "--no-half"),
            ],
            routes=[
                _route("diffusers", "supported", "txt2img/img2img", "StableDiffusion3Pipeline / StableDiffusion3Img2ImgPipeline"),
                _route("inpaint", "supported", "inpaint", "StableDiffusion3InpaintPipeline"),
            ],
            sidecars=["SD3 text encoders/config cache", "VAE", "LoRA when compatible"],
            lora="Treat as family-specific; do not reuse SD1.5/SDXL LoRAs without explicit metadata.",
            blockers=["SD3.5 Large single-file checkpoints need gated config files cached or account access."],
            modules=["aiwf.web.pro_api", "aiwf.infrastructure.diffusers.backend"],
        ),
        _family(
            id="flux",
            label="Flux / Flux Fill / Flux Kontext",
            category="image",
            status="supported-experimental-quants",
            summary="Single-transformer Flux route with shared CLIP-L/T5/AE sidecars, GGUF support, bitsandbytes 4-bit safetensors loader, Flux Fill inpaint, and Kontext folder route.",
            storage=[".safetensors transformer", ".gguf transformer", "FluxKontext Diffusers folder"],
            precisions=[
                _precision("BF16", "preferred", "FluxTransformer2DModel.from_single_file"),
                _precision("FP16", "fallback", "Flux T5 and VAE compute fallback"),
                _precision("FP8", "partial", "--fluxfp8 runtime dtype / patched Flux converter", "schema-mismatched FP8 files are blocked"),
                _precision("NF4", "supported", "bitsandbytes 4-bit safetensors loader"),
                _precision("FP4", "supported", "bitsandbytes 4-bit safetensors loader"),
                _precision("GGUF Q4/Q5/Q8", "experimental", "GGUFQuantizationConfig"),
            ],
            routes=[
                _route("flux", "supported", "txt2img", "DiffusersBackend._load_flux_checkpoint"),
                _route("flux-fill", "supported", "inpaint", "DiffusersBackend._load_flux_fill_pipeline"),
                _route("flux-kontext", "folder-only", "image/text route", "DiffusersBackend._load_flux_kontext_checkpoint"),
            ],
            sidecars=["CLIP-L", "T5-XXL fp16/fp8", "ae.safetensors", "Flux LoRA", "Flux VAE"],
            lora="Base Flux supports runtime LoRA; Flux.2/Z/Qwen/Sana are intentionally blocked until family loaders handle adapters.",
            blockers=["Known bad Flux FP8/GGUF-NF4 assets are blocked from normal selection.", "Kontext is folder-only, not raw single-file."],
            modules=["aiwf.infrastructure.diffusers.backend", "aiwf.infrastructure.diffusers.flux_bnb_loader", "aiwf.infrastructure.quant.bnb_nf4_format"],
        ),
        _family(
            id="flux2_klein",
            label="Flux.2 Klein",
            category="image",
            status="experimental",
            summary="Flux.2 Klein route uses new Diffusers Flux2KleinPipeline when available, single transformer or folder, Qwen3 text encoder components, NF4 encoder fallback, and GGUF transformer loading.",
            storage=[".safetensors transformer", ".gguf transformer", "Diffusers folder"],
            precisions=[
                _precision("BF16", "preferred", "Flux2Transformer2DModel / AutoencoderKLFlux2"),
                _precision("FP16", "fallback", "dtype fallback"),
                _precision("FP8", "runtime", "--fluxfp8 coerced through gguf compute fallback where needed"),
                _precision("NF4", "text-encoder", "BitsAndBytesConfig load_in_4bit for Qwen3 encoder"),
                _precision("GGUF Q4/Q5/Q8", "experimental", "GGUFQuantizationConfig for transformer"),
                _precision("INT8", "missing", "not currently a named loader route in AIWF", "candidate parity item; requires route-specific smoke"),
            ],
            routes=[_route("flux2-klein", "experimental", "txt2img", "DiffusersBackend._load_flux2_klein_checkpoint")],
            sidecars=["Flux2 components folder", "Qwen3 text encoder", "tokenizer", "scheduler", "Flux2 VAE"],
            lora="Runtime LoRA disabled by code for Flux.2 Klein until adapter path is implemented.",
            blockers=["Needs newer Diffusers/Transformers stack.", "int8/convrot is listed as missing until loader support and receipts exist."],
            modules=["aiwf.infrastructure.diffusers.backend", "aiwf.infrastructure.diffusers.model_arch"],
        ),
        _family(
            id="z_image",
            label="Z-Image",
            category="image",
            status="experimental-blocked-on-windows-gguf",
            summary="Z-Image route uses ZImagePipeline, ZImageTransformer2DModel, component folder, Qwen3 text encoder with NF4 option, and GGUF transformer loading when platform kernels allow it.",
            storage=[".safetensors transformer", ".gguf transformer", "Diffusers folder"],
            precisions=[
                _precision("BF16", "preferred", "ZImagePipeline / ZImageTransformer2DModel"),
                _precision("FP16", "fallback", "dtype fallback"),
                _precision("NF4", "text-encoder", "BitsAndBytesConfig load_in_4bit"),
                _precision("GGUF Q4/Q5/Q8", "platform-limited", "GGUFQuantizationConfig", "Windows GGUF is blocked unless kernels exist"),
                _precision("FP8", "candidate", "single-transformer route", "needs bounded smoke receipt"),
            ],
            routes=[_route("z-image", "experimental", "txt2img", "DiffusersBackend._load_z_image_checkpoint")],
            sidecars=["Z-Image components folder", "Qwen3 text encoder", "tokenizer", "VAE", "scheduler"],
            lora="Runtime LoRA disabled until Z-Image adapter support exists.",
            blockers=["Z-Image GGUF is blocked on Windows in model_blocks.py due fused kernel availability and VRAM paging."],
            modules=["aiwf.infrastructure.diffusers.backend", "aiwf.infrastructure.diffusers.model_blocks"],
        ),
        _family(
            id="qwen_image",
            label="Qwen Image / Nunchaku",
            category="image",
            status="partial",
            summary="Full Qwen Image Diffusers folder route plus isolated Qwen Nunchaku Lightning sidecar runtime for SVDQ-int4 single transformer.",
            storage=["Diffusers folder", "Nunchaku .safetensors transformer"],
            precisions=[
                _precision("BF16", "preferred", "QwenImagePipeline"),
                _precision("FP16", "fallback", "dtype fallback"),
                _precision("INT4", "supported-sidecar", "QwenNunchakuService", "svdq-int4 transformer plus base components"),
                _precision("GGUF", "missing", "no native LLM/VL GGUF worker yet"),
            ],
            routes=[
                _route("qwen-image", "supported-when-folder-installed", "txt2img", "DiffusersBackend._load_qwen_image_checkpoint"),
                _route("qwen-nunchaku", "sidecar", "txt2img", "QwenNunchakuService.generate"),
            ],
            sidecars=["Qwen Image base Diffusers folder", "Nunchaku engine venv", "runner script", "single transformer"],
            lora="Runtime LoRA disabled until Qwen adapter route is implemented.",
            blockers=["LLM/VL GGUF rows remain metadata-only until a local worker/API route exists."],
            modules=["aiwf.services.qwen_nunchaku", "aiwf.infrastructure.diffusers.backend", "aiwf.services.pipeline_preflight"],
        ),
        _family(
            id="sana",
            label="Sana / Sana Sprint",
            category="image",
            status="supported-smoked",
            summary="Sana and Sana Sprint full Diffusers-folder image routes, with bounded Sana Sprint smoke evidence in the QA matrix.",
            storage=["Diffusers folder with model_index.json"],
            precisions=[
                _precision("BF16", "preferred", "SanaPipeline/SanaSprintPipeline"),
                _precision("FP16", "fallback", "dtype fallback"),
                _precision("FP8/int8/4bit", "not-a-current-route", "none", "do not expose without a dedicated loader and receipt"),
            ],
            routes=[_route("sana", "supported-smoked", "txt2img", "DiffusersBackend._load_sana_checkpoint")],
            sidecars=["Sana Diffusers snapshot", "tokenizer/text encoder inside folder"],
            lora="Runtime LoRA disabled by code for Sana until adapter support exists.",
            blockers=["Requires full folder, not loose single-file transformer."],
            modules=["aiwf.infrastructure.diffusers.backend", "aiwf.services.pipeline_registry"],
        ),
        _family(
            id="sana_video",
            label="Sana Video",
            category="video",
            status="supported-smoked-silent",
            summary="SANA-Video 2B 480p Diffusers route with quantization/tiling settings and silent MP4 smoke evidence; audio is a post-process lane.",
            storage=["Diffusers folder"],
            precisions=[
                _precision("BF16", "preferred", "SanaVideoService"),
                _precision("FP16", "fallback", "SanaVideoService"),
                _precision("bitsandbytes", "optional", "SanaVideoService quantization setting"),
            ],
            routes=[_route("sana-video", "supported-smoked", "t2v/i2v video", "SanaVideoService.generate")],
            sidecars=["SANA-Video snapshot", "SageAttention optional", "MMAudio post-process"],
            lora="No Sana Video LoRA route is declared in this code pass.",
            blockers=["Generated audio is not part of the main smoke; use audio/mux post-process."],
            modules=["aiwf.services.sana_video", "aiwf.core.domain.sana_video", "aiwf.services.pipeline_preflight"],
        ),
        _family(
            id="wan",
            label="Wan Video",
            category="video",
            status="supported-plus-sidecars",
            summary="Wan fast 5B I2V plus experimental high/low FP8 and GGUF model-pair routes with explicit VAE, text encoder, LoRA, offload, sampler, and sigma controls.",
            storage=["Diffusers folder", ".safetensors transformer", ".gguf high/low transformers"],
            precisions=[
                _precision("BF16", "preferred", "WanImageToVideoPipeline / native runner"),
                _precision("FP16", "fallback", "dtype fallback"),
                _precision("FP8", "supported-experimental", "Comfy scaled-FP8 dequant/native FP8 path"),
                _precision("GGUF Q4/Q5", "experimental-supported", "Wan GGUF runtime high/low pair"),
                _precision("GGUF Q3/Q6", "metadata-only", "readiness classifier", "smoke separately before marketing"),
            ],
            routes=[
                _route("wan-fast-5b", "supported", "image-to-video", "WanService.generate / WanI2VBackend"),
                _route("wan-high-low-fp8", "experimental", "image-to-video", "WAN_RUNTIME_HIGH_LOW_FP8"),
                _route("wan-gguf", "experimental", "image-to-video", "WAN_RUNTIME_HIGH_LOW"),
            ],
            sidecars=["high-noise transformer", "low-noise transformer", "Wan VAE", "UMT5/Wan text encoder", "Wan LoRA high/low", "offload plan"],
            lora="Single high/low LoRA fields exist; multi-LoRA stack UI/runtime is the next family patch item.",
            blockers=["T2V 1.3B, Animate, and Fun-Control/control are explicitly unsupported by the current I2V route."],
            modules=["aiwf.core.domain.wan", "aiwf.services.wan", "aiwf.infrastructure.wan.pipeline", "aiwf.services.wan_models"],
        ),
        _family(
            id="ltx",
            label="LTX Video",
            category="video",
            status="partial-supported",
            summary="LTX 2B Diffusers route and LTX 2.3 isolated worker route with HF/converted Gemma support; native Gemma GGUF and FP4/NVFP4 remain blocked until a hidden-state-capable backend exists.",
            storage=[".safetensors checkpoint", "HF-shaped Gemma folder", "Gemma GGUF metadata probe", "T5XXL safetensors"],
            precisions=[
                _precision("BF16", "preferred", "LTX worker / LTX 2B Diffusers dtype"),
                _precision("FP16", "fallback", "LTX 2B T5XXL fp16"),
                _precision("FP8", "supported-smoked", "LTX 2.3 one-stage fp8-cast"),
                _precision("FP8 scaled-mm", "configured", "LtxVideoRequest quantization"),
                _precision("NVFP4/FP4", "blocked", "readiness classifier", "no stable FP4/NVFP4 route"),
                _precision("Gemma Q3_K_M GGUF", "blocked-probe-only", "probe_ltx_runtime", "needs hidden-state tuple + attention mask backend"),
            ],
            routes=[
                _route("ltx-2b-diffusers", "supported-smoked", "text/image-to-video", "run_ltx2b_diffusers"),
                _route("ltx-one-stage-hf-gemma", "supported-smoked", "video", "LtxService.generate"),
                _route("ltx-one-stage-heretic-gguf", "blocked-probe-only", "metadata probe", "LtxService.probe_native_gemma_gguf"),
            ],
            sidecars=["Gemma text encoder folder", "T5XXL fp16", "spatial upscaler", "video/audio VAE", "offload"],
            lora="LTX LoRA files are inventoried; runtime adapter application needs route-specific implementation/receipts.",
            blockers=["Native Gemma GGUF cannot generate until a backend returns every hidden-state layer and attention mask.", "BF16 22B on Windows is blocked unless explicitly retested."],
            modules=["aiwf.core.domain.ltx", "aiwf.services.ltx", "aiwf.services.ltx_diffusers", "aiwf.services.pipeline_preflight"],
        ),
        _family(
            id="onnx",
            label="ONNX Image",
            category="image",
            status="blocked-until-folder",
            summary="Optional ONNX image route with provider preflight for CUDA/DirectML/CPU; blocked until the expected model folder exists.",
            storage=["ONNX folder"],
            precisions=[
                _precision("FP16/FP32", "provider-dependent", "ONNX Runtime"),
                _precision("INT8", "not-declared", "none", "requires explicit quantized ONNX model contract"),
            ],
            routes=[_route("onnx", "blocked-cleanly", "txt2img", "aiwf.infrastructure.onnx.backend")],
            sidecars=["text_encoder/model.onnx", "unet/model.onnx", "vae_decoder/model.onnx", "tokenizer"],
            lora="No ONNX LoRA route declared.",
            blockers=["models/onnx folder and provider-specific receipts required."],
            modules=["aiwf.services.pipeline_preflight", "aiwf.infrastructure.onnx"],
        ),
        _family(
            id="llm_vl",
            label="LLM / VL GGUF",
            category="assistant",
            status="metadata-only",
            summary="GGUF LLM/VL assets are inventoried and can support future chat/VL or text-encoder probes, but no native worker/API route is complete in this code pass.",
            storage=[".gguf"],
            precisions=[
                _precision("Q2/Q3/Q4/Q5/Q6/Q8", "metadata-only", "model_header / pipeline_readiness"),
                _precision("FP16/BF16", "metadata-only", "model_header"),
            ],
            routes=[_route("llm-vl-worker", "missing", "chat/VL", "not implemented")],
            sidecars=["tokenizer", "processor", "vision projector depending on model"],
            lora="Not applicable until LLM/VL worker exists.",
            blockers=["Keep as metadata-only until a GGUF worker/API route exists."],
            modules=["aiwf.infrastructure.model_header", "aiwf.services.pipeline_readiness"],
        ),
    ]


def _family_id_for_readiness_family(value: str) -> str:
    key = str(value or "").lower().replace("-", "_")
    if key in {"checkpoint", "image", "diffusers"}:
        return "sd15"
    if key == "flux_fill":
        return "flux"
    if key == "flux2":
        return "flux2_klein"
    if key == "zimage":
        return "z_image"
    if key == "qwen":
        return "qwen_image"
    if key == "sana_video":
        return "sana_video"
    return key


def _record_precision(record: Any) -> str:
    quant = normalize_precision_label(getattr(record, "quantization", "") or "")
    if quant:
        return quant
    path = getattr(record, "path", "") or ""
    return detect_precision_from_name(path)


def _readiness_overlay(flags: RuntimeFlags, settings: UserSettings) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from aiwf.services.pipeline_readiness import collect_pipeline_readiness

        records = collect_pipeline_readiness(flags, settings, include_downloads=False)
    except Exception as exc:
        return {"error": f"readiness unavailable: {type(exc).__name__}: {exc}"}, []

    counts_by_family: dict[str, Counter] = defaultdict(Counter)
    precision_by_family: dict[str, Counter] = defaultdict(Counter)
    blocked: list[dict[str, Any]] = []
    for record in records:
        family_id = _family_id_for_readiness_family(getattr(record, "family", ""))
        status = str(getattr(record, "status", "") or "unknown")
        counts_by_family[family_id][status] += 1
        precision = _record_precision(record)
        if precision:
            precision_by_family[family_id][precision] += 1
        if status in {"broken-runtime", "blocked-cleanly", "unsupported-no-route"}:
            blocked.append(
                {
                    "family": family_id,
                    "status": status,
                    "path": str(getattr(record, "path", "") or ""),
                    "route": str(getattr(record, "route", "") or ""),
                    "reason": str(getattr(record, "reason", "") or ""),
                    "suggestedAction": str(getattr(record, "suggested_action", "") or ""),
                }
            )
    return {
        "recordCount": len(records),
        "countsByFamily": {key: dict(value) for key, value in sorted(counts_by_family.items())},
        "precisionByFamily": {key: dict(value) for key, value in sorted(precision_by_family.items())},
    }, blocked[:80]


def build_model_family_matrix(flags: RuntimeFlags, settings: UserSettings | None = None) -> dict[str, Any]:
    """Return the family support matrix plus local readiness evidence."""
    user_settings = settings or UserSettings()
    readiness, blockers = _readiness_overlay(flags, user_settings)
    counts_by_family = readiness.get("countsByFamily", {}) if isinstance(readiness, dict) else {}
    precision_by_family = readiness.get("precisionByFamily", {}) if isinstance(readiness, dict) else {}
    families: list[dict[str, Any]] = []
    for base in static_model_family_support():
        item = dict(base)
        item["localReadiness"] = counts_by_family.get(item["id"], {}) if isinstance(counts_by_family, dict) else {}
        item["localDetectedPrecisions"] = precision_by_family.get(item["id"], {}) if isinstance(precision_by_family, dict) else {}
        families.append(item)

    return {
        "schema": "aiwf.model-family-support.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "code-indexed support matrix plus live pipeline_readiness overlay; no model weights loaded",
        "precisionVocabulary": supported_precision_names(),
        "readiness": readiness,
        "blockedExamples": blockers,
        "families": families,
    }
