"""Static knowledge about Qwen-Image 2.1 as it runs in ComfyUI.

Everything here was taken from primary sources at build time (2026-10-06):

* Comfy-Org/Qwen-Image-2.1 repack file list and sizes (Hugging Face API).
* ComfyUI core sources: ``comfy_extras/nodes_qwen.py`` (TextEncodeQwenImage21,
  QwenImage21Cache), ``nodes_model_patch.py`` (ModelPatchLoader,
  ZImageFunControlnet which also drives the Qwen 2.1 Fun ControlNet),
  ``nodes_custom_sampler.py`` (ManualSigmas, SamplerCustom),
  ``nodes_model_advanced.py`` (ModelSamplingFlux), ``nodes_apg.py``,
  ``nodes_fresca.py``, ``nodes_textgen.py`` (TextGenerate), ``nodes.py``.
* Official Comfy-Org workflow templates ``image_qwen_image_2_1_t2i``,
  ``image_qwen_image_2_1_image_edit`` and
  ``image_qwen_image_2_1_background_removal`` (MIT).
* Accelerator LoRA model cards (Turbo8, Pruna, Viggle, Alibaba PAI Fun-Acc).
* Quantization measurements published by Unsloth (INT8 vs FP8 LPIPS) and the
  16 GB community reports linked in the README.

Nothing in this module talks to the network or to ComfyUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Model files (Comfy-Org/Qwen-Image-2.1 repack)
# ---------------------------------------------------------------------------

COMFY_ORG_REPO = "Comfy-Org/Qwen-Image-2.1"
BASE_REPO = "Qwen/Qwen-Image-2.1"
MIN_COMFYUI_VERSION = "0.37.0"  # TextEncodeQwenImage21 landed in core here (PR #16400)


@dataclass(frozen=True)
class ModelFile:
    filename: str
    folder: str  # ComfyUI models subfolder
    size_gb: float
    label: str
    note: str = ""
    recommended_16gb: bool = False

    @property
    def download_url(self) -> str:
        return f"https://huggingface.co/{COMFY_ORG_REPO}/resolve/main/{self.folder}/{self.filename}"


DIFFUSION_MODELS: tuple[ModelFile, ...] = (
    ModelFile(
        "qwen_image_2.1_int8_convrot.safetensors", "diffusion_models", 7.26,
        "INT8 ConvRot (recommended)",
        "Rotation-based int8. Unsloth measured LPIPS 0.064 vs bf16 (fp8 is 0.112 at the same size). "
        "Official template default.",
        recommended_16gb=True,
    ),
    ModelFile(
        "qwen_image_2.1_bf16.safetensors", "diffusion_models", 14.23,
        "BF16 (reference quality)",
        "Does not leave room for anything else on a 16 GB card; ComfyUI will offload and run slower.",
    ),
)

TEXT_ENCODERS: tuple[ModelFile, ...] = (
    ModelFile(
        "qwen3vl_8b_int8_convrot.safetensors", "text_encoders", 9.35,
        "Qwen3-VL 8B INT8 ConvRot (recommended)",
        "Encodes once, then ComfyUI unloads it before sampling. Validated on a 4060 Ti 16 GB with no offload "
        "(peak ~15 GB at 1024x1024).",
        recommended_16gb=True,
    ),
    ModelFile(
        "qwen3vl_8b_w4a8.safetensors", "text_encoders", 6.31,
        "Qwen3-VL 8B W4A8 (headroom)",
        "Use when stacking 6+ references or editing at native 2K and the int8 encoder makes ComfyUI thrash. "
        "Quality cost of 4-bit vision-language encoding is not published; compare on your own prompts.",
    ),
    ModelFile(
        "qwen3vl_8b_bf16.safetensors", "text_encoders", 17.53,
        "Qwen3-VL 8B BF16",
        "Larger than the whole card. Only sensible with CLIPLoader device=cpu and lots of system RAM.",
    ),
)

VAES: tuple[ModelFile, ...] = (
    ModelFile(
        "qwen_image_2.1_vae_bf16.safetensors", "vae", 0.68,
        "RGBA VAE BF16 (required)",
        "64-channel, 16x spatial compression, carries the alpha channel.",
        recommended_16gb=True,
    ),
)

PROMPT_ENHANCERS: tuple[ModelFile, ...] = (
    ModelFile(
        "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors", "text_encoders", 9.47,
        "Prompt enhancer (text to image)",
        "Qwen3.5 9B fine-tune. Rewrites a short prompt into a long scene description. Optional; adds a model load.",
    ),
    ModelFile(
        "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors", "text_encoders", 9.47,
        "Prompt enhancer (image edit)",
        "Sees the reference images and rewrites a vague edit instruction into a precise one. Optional.",
    ),
)

CONTROLNET_PATCHES: tuple[ModelFile, ...] = (
    ModelFile(
        "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors", "model_patches", 3.78,
        "Fun ControlNet Union INT8 (recommended)",
        "Alibaba PAI union control: Canny, Depth, Grayscale, HED, Lineart, MLSD, Pose, Scribble + inpainting.",
        recommended_16gb=True,
    ),
    ModelFile(
        "qwen_image_2.1_fun_controlnet_union_bf16.safetensors", "model_patches", 7.55,
        "Fun ControlNet Union BF16",
    ),
)

ALL_MODEL_FILES: tuple[ModelFile, ...] = (
    DIFFUSION_MODELS + TEXT_ENCODERS + VAES + PROMPT_ENHANCERS + CONTROLNET_PATCHES
)

RECOMMENDED_16GB = {
    "dit": "qwen_image_2.1_int8_convrot.safetensors",
    "text_encoder": "qwen3vl_8b_int8_convrot.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    "controlnet": "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
    "pe_t2i": "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors",
    "pe_i2i": "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors",
}

QUANT_GUIDANCE_16GB = """\
Quantization pick for a 16 GB RTX 4070 Ti SUPER (int8 ConvRot everywhere):

* DiT: qwen_image_2.1_int8_convrot (7.26 GB). Same size as fp8 but measurably
  closer to bf16 (Unsloth: LPIPS 0.064 vs 0.112 for fp8, SSIM 0.936 vs 0.899).
  fp8 did not run faster in published tests, so there is no reason to take its
  quality hit.
* Text encoder: qwen3vl_8b_int8_convrot (9.35 GB). It only lives on the GPU
  while encoding; ComfyUI evicts it before the DiT samples, so the two never
  need to fit together. Measured on a 4060 Ti 16 GB: ~20 s per 1024x1024
  image at 25 steps, ~60 s for an edit, peak ~15 GB, no offload flags.
* Drop to the W4A8 encoder (6.31 GB) only when you stack many references or
  render native 2K and the log shows the encoder being paged repeatedly.
* GGUF Q4_K_M (4.6 GB) is a fallback for 8-12 GB cards, costs quality and needs
  the ComfyUI-GGUF custom nodes plus a patched Qwen3-VL loader; not needed on
  16 GB. There is no official Nunchaku/SVDQuant build for 2.1.
* Speed: keep the 25-step base for finals. For iteration use the Turbo8 LoRA
  (8 steps, no CFG, still supports editing, references and RGBA).
"""

# ---------------------------------------------------------------------------
# Resolutions
# ---------------------------------------------------------------------------

# Native 2K sizes from the model card. Multiples of 32; the DiT groups latents
# in 2x2 blocks so 32 px is the safe step.
NATIVE_2K = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# ~1 megapixel working sizes (fast on 16 GB, the official template default).
ONE_MP = {
    "1:1": (1024, 1024),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}

RESOLUTION_STEP = 32


def snap_resolution(width: int, height: int, step: int = RESOLUTION_STEP) -> tuple[int, int]:
    """Round a size to the DiT's grid (multiples of 32, never below 32)."""
    w = max(step, int(round(width / step)) * step)
    h = max(step, int(round(height / step)) * step)
    return w, h


# ---------------------------------------------------------------------------
# Sampling presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerPreset:
    key: str
    label: str
    steps: int
    cfg: float
    sampler: str = "euler"
    scheduler: str = "simple"
    negative: str = ""
    fix_guidance: bool = False  # APG + FreSca stack from the "Fix LoRA" recipe
    note: str = ""


FIX_LORA_NEGATIVE = (
    "artifacts, gpt-image, washed-out colors, low quality, low resolution, AI slop, deviantart, "
    "sloppy lines, rough sketch, blurry, indistinct, missing fingers, badly drawn hands, wrong number of fingers"
)

SAMPLER_PRESETS: dict[str, SamplerPreset] = {
    "official_25": SamplerPreset(
        "official_25", "Official: 25 steps, CFG 1 (euler/simple)", 25, 1.0,
        note="Comfy-Org template default. Negative prompt is ignored at CFG 1.",
    ),
    "quality_40": SamplerPreset(
        "quality_40", "Quality: 40 steps, CFG 1", 40, 1.0,
        note="Model card default step count. ~1.6x slower than 25 steps.",
    ),
    "dense_cfg2": SamplerPreset(
        "dense_cfg2", "Dense prompt: 25 steps, CFG 2", 25, 2.0,
        negative="blurry, low quality, watermark, text artifacts",
        note="Follows long prompts and typography more tightly; edges get sharper/harsher. CFG 5+ degrades.",
    ),
    "fix_lora": SamplerPreset(
        "fix_lora", "Fix-LoRA recipe: 20 steps, CFG 3, seeds_2/sgm_uniform, APG+FreSca", 20, 3.0,
        sampler="seeds_2", scheduler="sgm_uniform", negative=FIX_LORA_NEGATIVE, fix_guidance=True,
        note="Community recipe for the washed-out/blotchy look; add the e-n-v-y 'Qwen-Image 2.1 Fix' LoRA at 1.0.",
    ),
}

# APG / FreSca values from the Fix LoRA workflow (APG 1, 10, 0.3 ; FreSca 1, 2, 8)
FIX_GUIDANCE = {"apg": {"eta": 1.0, "norm_threshold": 10.0, "momentum": 0.3},
                "fresca": {"scale_low": 1.0, "scale_high": 2.0, "freq_cutoff": 8}}

# ---------------------------------------------------------------------------
# Few-step accelerator LoRAs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Accelerator:
    key: str
    label: str
    repo: str
    filename: str
    size_gb: float
    steps: int
    cfg: float = 1.0
    strength: float = 1.0
    mode: str = "ksampler"  # "ksampler" (steps + scheduler) or "sigmas" (ManualSigmas + SamplerCustom)
    sampler: str = "euler"
    scheduler: str = "simple"
    sigmas: tuple[float, ...] = ()
    model_sampling_flux: tuple[float, float] | None = None  # (max_shift, base_shift)
    supports_edit: bool = True
    supports_rgba: bool = True
    status: str = "verified"  # verified | approximate | experimental
    note: str = ""

    @property
    def download_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"

    def sigmas_string(self) -> str:
        return ", ".join(f"{s:.9g}" for s in self.sigmas)


ACCELERATORS: dict[str, Accelerator] = {
    "none": Accelerator("none", "None (full-step base model)", "", "", 0.0, 25),
    "turbo8": Accelerator(
        "turbo8", "Turbo8 — 8 steps, no CFG (recommended for iteration)",
        "chriswritescode/Turbo8-LoRA-Qwen-Image-2.1", "turbo8_lora_step2500.safetensors", 1.36,
        steps=8, cfg=1.0, mode="ksampler", sampler="euler", scheduler="simple",
        model_sampling_flux=(0.6935, 0.5),
        note="Rank-128 DMD2 distill. Covers T2I, editing with up to 10 refs, RGBA and extraction. "
             "Dense/long text degrades vs 40 steps. Uses ModelSamplingFlux max_shift 0.6935 / base_shift 0.5 "
             "per its model card.",
    ),
    "pruna8": Accelerator(
        "pruna8", "Pruna 8-step (ManualSigmas)",
        "NidAll/pruna-image-2.1-comfyui-loras", "p_qwen_image_2.1_8step_v0.1.safetensors", 0.34,
        steps=8, mode="sigmas",
        sigmas=(1.0, 0.933333333, 0.857142857, 0.769230769, 0.666666667, 0.545454545, 0.4, 0.222222222, 0.0),
        status="approximate",
        note="v0.1, trained at 1K; up to 3 references tested. Exact sigma schedule required (trailing 0 added so "
             "the last step lands on clean).",
    ),
    "pruna5": Accelerator(
        "pruna5", "Pruna 5-step (ManualSigmas)",
        "NidAll/pruna-image-2.1-comfyui-loras", "p_qwen_image_2.1_5step_v0.1.safetensors", 0.34,
        steps=5, mode="sigmas", sigmas=(1.0, 0.94, 0.857142857, 0.666666667, 0.4, 0.0),
        status="approximate",
        note="Fastest; visibly below the 8-step. Same caveats as Pruna 8.",
    ),
    "viggle6": Accelerator(
        "viggle6", "Viggle Turbo v0.3 — 6 steps (ManualSigmas, approximate)",
        "Viggle/Qwen-Image-2.1-viggle-turbo",
        "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors", 0.7,
        steps=6, mode="sigmas", sigmas=(1.0, 0.9375, 0.875, 0.75, 0.5, 0.25, 0.0),
        status="approximate",
        note="Official path uses Viggle's custom node (unmerged LoRA + shift_terminal=null). Loading through the "
             "core LoRA node is an approximation; multi-reference identity drops more than Turbo8.",
    ),
    "fun_acc4": Accelerator(
        "fun_acc4", "Alibaba PAI Fun-Acc — 4 steps (official distill, ManualSigmas, experimental in ComfyUI)",
        "alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs", "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors", 0.35,
        steps=4, mode="sigmas", sigmas=(1.0, 0.9169867, 0.7861579, 0.549491, 0.0),
        status="experimental",
        note="Parallel Decoding Distillation; sigma schedule from pdd_config.json. Published for Diffusers; the "
             "ComfyUI key layout may need a community conversion. Some edits come out darker/blurrier.",
    ),
}

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

RGBA_PROMPT_HINT = (
    "This is an RGBA image with transparency. The subject is isolated on a fully transparent background "
    "(alpha channel), no scenery, no backdrop."
)

EDIT_PROMPT_PRESETS: dict[str, str] = {
    "Remove background (RGBA PNG)": "Remove the background, and output a PNG image",
    "Extract subject (RGBA)": "Extract the main subject from <image1> onto a fully transparent background, "
                              "output a PNG with alpha, keep every edge and fine hair detail",
    "Put outfit from image2 on image1": "Keep the character and pose in <image1> unchanged, put the outfit from "
                                        "<image2> on the character, keep lighting and background of <image1>",
    "Place product from image2 into image1": "Place the product from <image2> into the scene of <image1> on the "
                                             "marked spot, match perspective, lighting and shadows",
    "Apply style of image2 to image1": "Redraw <image1> in the art style of <image2>, keep composition and identity",
    "Edit only the marked area": "Change only the area marked in red in <image1>: ",
    "Replace text": "Replace the text in <image1> with \"NEW TEXT\", keep font weight, color and placement",
}

NEGATIVE_PRESETS: dict[str, str] = {
    "(none — CFG 1 ignores it)": "",
    "Quality": "blurry, low quality, watermark, jpeg artifacts, text artifacts, extra fingers",
    "Fix-LoRA recipe": FIX_LORA_NEGATIVE,
}

# ---------------------------------------------------------------------------
# Node signatures used for offline validation of built prompts
# (required input names per class_type, from ComfyUI 0.39.0 sources)
# ---------------------------------------------------------------------------

NODE_SIGNATURES: dict[str, dict[str, Sequence[str]]] = {
    "UNETLoader": {"required": ("unet_name", "weight_dtype")},
    "CLIPLoader": {"required": ("clip_name", "type"), "optional": ("device",)},
    "VAELoader": {"required": ("vae_name",)},
    "LoraLoaderModelOnly": {"required": ("model", "lora_name", "strength_model")},
    "TextEncodeQwenImage21": {"required": ("clip", "prompt", "negative_prompt"),
                              "optional": ("vae", "resolution"), "autogrow": "images.image_"},
    "QwenImage21Cache": {"required": ("model", "device", "dtype")},
    "EmptyLatentImage": {"required": ("width", "height", "batch_size")},
    "KSampler": {"required": ("model", "seed", "steps", "cfg", "sampler_name", "scheduler",
                              "positive", "negative", "latent_image", "denoise")},
    "SamplerCustom": {"required": ("model", "add_noise", "noise_seed", "cfg", "positive", "negative",
                                   "sampler", "sigmas", "latent_image")},
    "KSamplerSelect": {"required": ("sampler_name",)},
    "ManualSigmas": {"required": ("sigmas",)},
    "ModelSamplingFlux": {"required": ("model", "max_shift", "base_shift", "width", "height")},
    "APG": {"required": ("model", "eta", "norm_threshold", "momentum")},
    "FreSca": {"required": ("model", "scale_low", "scale_high", "freq_cutoff")},
    "VAEDecode": {"required": ("samples", "vae")},
    "SaveImage": {"required": ("images", "filename_prefix")},
    "SaveImageAdvanced": {"required": ("images", "filename_prefix", "format")},
    "LoadImage": {"required": ("image",)},
    "ImageToMask": {"required": ("image", "channel")},
    "ImageBatch": {"required": ("image1", "image2")},
    "ModelPatchLoader": {"required": ("name",)},
    "ZImageFunControlnet": {"required": ("model", "model_patch", "vae", "strength"),
                            "optional": ("image", "inpaint_image", "mask", "start_percent", "end_percent")},
    "TextGenerate": {"required": ("clip", "prompt", "max_length", "sampling_mode"),
                     "optional": ("image", "video", "audio", "thinking", "use_default_template", "mtp",
                                  "system_prompt")},
}

CORE_NODE_TYPES = frozenset(NODE_SIGNATURES)
