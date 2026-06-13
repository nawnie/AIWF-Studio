"""Wan 2.2 image-to-video backend (diffusers WanImageToVideoPipeline).

Heavy imports (torch/diffusers) are lazy so the app loads fine without the Wan
stack installed or the model downloaded. Tuned for consumer GPUs via CPU
offloading + VAE tiling — slow on 8 GB, but it runs.

Supports:
- Full diffusers layouts (model_index.json + subfolders) via from_pretrained
- Standalone `.safetensors` transformer weights (Comfy diffusion_models style),
  including native Comfy scaled-FP8, loaded into diffusers WanTransformer3DModel.
- Shared sampler path: UMT5, scheduler, dual-stage denoise, VAE decode (like Comfy KSampler).
- Only the transformer loader differs by format; GGUF needs UnetLoaderGGUF-style load
  (`transformer_runtime.py`), not full dequant into diffusers weights.

For dual high/low (Wan 2.2 14B I2V), the caller (WanService) must pre-resolve
and pass `components_base` (the path to the Wan2.2-TI2V-5B-Diffusers layout
or equivalent providing text_encoder/tokenizer/scheduler). The internal
finder is only a best-effort fallback and is intentionally cwd-independent.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COMFY_FP8_METADATA_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".pre_quant_scale",
    ".input_scale",
    ".scale_weight",
    ".scale_input",
)


def _torch_native_fp8_available() -> bool:
    try:
        import torch

        return bool(
            torch.cuda.is_available()
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "_scaled_mm")
        )
    except Exception:
        return False


def _is_native_comfy_fp8_transformer(path: str | None) -> bool:
    if not path:
        return False
    pp = Path(path)
    return pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(pp)


class WanUnavailable(RuntimeError):
    """Raised when the Wan deps or model are missing/unloadable."""


WAN_I2V_A14B_TRANSFORMER_CONFIG = {
    "added_kv_proj_dim": None,
    "attention_head_dim": 128,
    "cross_attn_norm": True,
    "eps": 1e-06,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "in_channels": 36,
    "num_attention_heads": 40,
    "num_layers": 40,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "text_dim": 4096,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}


def _video_status(message: str) -> None:
    print(f"[AIWF] Video: {message}", flush=True)


def _safetensors_uses_comfy_fp8_quant(path: Path) -> bool:
    if path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
    except Exception:
        return False

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            key_l = key.lower()
            if key_l.endswith(_COMFY_FP8_METADATA_SUFFIXES):
                return True
            try:
                dtype = handle.get_slice(key).get_dtype()
            except Exception:
                continue
            if str(dtype).upper().startswith("F8"):
                return True
    return False


def _normalize_wan_transformer_key(key: str) -> str:
    """Normalize original/Comfy Wan keys before applying diffusers renames."""
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _dequantize_comfy_fp8_state_dict(sd: dict, torch_dtype=None) -> dict:
    """Convert Comfy FP8 Wan tensors to regular torch tensors for diffusers.

    Comfy-style FP8 safetensors store quantized weights plus scalar
    ``*.weight_scale`` tensors. Diffusers does not consume those sidecar keys,
    so AIWF expands the weights before loading them into the Wan transformer.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is required to reach this path
        torch = None

    out: dict = {}
    for key, value in sd.items():
        key_l = key.lower()
        if key_l.endswith(_COMFY_FP8_METADATA_SUFFIXES):
            continue

        next_value = value
        dtype_name = str(getattr(value, "dtype", "")).lower()
        if "float8" in dtype_name:
            scale = None
            if key.endswith(".weight"):
                base_key = key.removesuffix(".weight")
                scale = sd.get(f"{base_key}.weight_scale")
                if scale is None:
                    scale = sd.get(f"{base_key}.scale_weight")
            next_value = value.float()
            if scale is not None:
                next_value = next_value * scale.float()
            if torch_dtype is not None:
                next_value = next_value.to(dtype=torch_dtype)
        elif torch_dtype is not None and torch is not None and getattr(value, "is_floating_point", lambda: False)():
            next_value = value.to(dtype=torch_dtype)

        out[key] = next_value
    return out


def _new_fp8_scaled_linear(in_features: int, out_features: int, bias: bool):
    import torch

    class FP8ScaledLinear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.weight = torch.nn.Parameter(
                torch.empty((out_features, in_features), device="meta", dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            if bias:
                self.bias = torch.nn.Parameter(torch.empty((out_features,), device="meta"), requires_grad=False)
            else:
                self.register_parameter("bias", None)
            self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32), persistent=True)

        def forward(self, input):
            import torch.nn.functional as F

            can_scaled_mm = (
                input.is_cuda
                and self.weight.is_cuda
                and hasattr(torch, "_scaled_mm")
                and self.in_features % 16 == 0
                and self.out_features % 16 == 0
            )
            if can_scaled_mm:
                original_shape = input.shape[:-1]
                x = input.reshape(-1, self.in_features).contiguous()
                m, _k = x.shape
                pad_m = (16 - m % 16) % 16
                if pad_m:
                    x = F.pad(x, (0, 0, 0, pad_m))
                scale_a = torch.ones((), device=x.device, dtype=torch.float32)
                x8 = x.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
                weight_t = self.weight.t().contiguous()
                try:
                    y = torch._scaled_mm(
                        x8,
                        weight_t,
                        scale_a=scale_a,
                        scale_b=self.weight_scale.to(device=x.device, dtype=torch.float32),
                        out_dtype=input.dtype
                        if input.dtype in (torch.float16, torch.bfloat16)
                        else torch.bfloat16,
                    )
                    if pad_m:
                        y = y[:m, :]
                    if self.bias is not None:
                        y = y + self.bias.to(device=y.device, dtype=y.dtype)
                    return y.reshape(*original_shape, self.out_features)
                except Exception as exc:
                    if not getattr(self, "_scaled_mm_warned", False):
                        logger.warning(
                            "FP8ScaledLinear _scaled_mm failed (%s); falling back to bf16 linear for this layer.",
                            exc,
                        )
                        self._scaled_mm_warned = True

            weight = (self.weight.float() * self.weight_scale.float()).contiguous()
            return F.linear(input, weight.to(device=input.device, dtype=input.dtype), self.bias)

    return FP8ScaledLinear(in_features, out_features, bias)


def _module_parent_and_name(root, module_path: str):
    parent = root
    parts = module_path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _replace_linear_with_fp8(root, module_path: str):
    import torch

    parent, name = _module_parent_and_name(root, module_path)
    current = parent[int(name)] if name.isdigit() else getattr(parent, name)
    if current.__class__.__name__ == "FP8ScaledLinear":
        return current
    if not isinstance(current, torch.nn.Linear):
        raise WanUnavailable(f"Expected Linear at '{module_path}', found {type(current).__name__}.")
    replacement = _new_fp8_scaled_linear(current.in_features, current.out_features, current.bias is not None)
    if name.isdigit():
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)
    return replacement


class _LazyWanTransformer:
    pass


def _new_lazy_wan_transformer(config, *, dtype, load_model, before_load=None):
    import contextlib
    import torch

    class LazyWanTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self._dtype = dtype
            self._load_model = load_model
            self._before_load = before_load
            self._loaded_model = None
            self._target_device = None
            # Accelerate's model-offload hook assumes every module in the
            # offload sequence has at least one parameter. The real low stage
            # is intentionally loaded later, so this sentinel keeps the hook
            # chain valid without materializing the 14B transformer early.
            self._aiwf_offload_sentinel = torch.nn.Parameter(
                torch.empty((), dtype=dtype), requires_grad=False
            )

        @property
        def dtype(self):
            if self._loaded_model is not None:
                return self._loaded_model.dtype
            return self._dtype

        def _ensure_loaded(self, device=None):
            target_device = device or self._target_device
            if self._loaded_model is None:
                if self._before_load is not None:
                    self._before_load()
                self._loaded_model = self._load_model()
                if target_device is not None:
                    self._loaded_model.to(target_device)
            elif target_device is not None:
                self._loaded_model.to(target_device)
            return self._loaded_model

        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            self._target_device = self._aiwf_offload_sentinel.device
            if self._loaded_model is not None:
                self._loaded_model.to(*args, **kwargs)
            return result

        @contextlib.contextmanager
        def cache_context(self, name):
            model = self._ensure_loaded()
            with model.cache_context(name):
                yield

        def forward(self, *args, **kwargs):
            device = None
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hasattr(hidden_states, "device"):
                device = hidden_states.device
            model = self._ensure_loaded(device=device)
            return model(*args, **kwargs)

    return LazyWanTransformer()


def _is_gguf_transformer(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() == ".gguf"


def _load_transformer_state_dict(pth: str, *, label: str = "transformer") -> dict:
    """Load and rename standalone Wan transformer weights for diffusers load_state_dict."""
    from aiwf.infrastructure.wan.transformer_runtime import (
        WanTransformerFormat,
        require_diffusers_transformer_path,
    )

    pp = Path(pth)
    fmt = require_diffusers_transformer_path(pp, label=label)
    _video_status(f"Loading transformer weights ({fmt.value}): {pp.name}")
    if fmt == WanTransformerFormat.GGUF_QUANTIZED:
        try:
            raw = _load_gguf_state_dict(pp)
        except Exception as exc:
            raise WanUnavailable(f"GGUF dequant stub failed for {pp.name}. Install `gguf`.") from exc
    else:
        from safetensors.torch import load_file

        raw = load_file(str(pp))
        if _safetensors_uses_comfy_fp8_quant(pp):
            raise WanUnavailable(
                f"Internal loader error: {pp.name} is Comfy FP8 and should use the native FP8 path."
            )
    return _apply_wan_transformer_key_renames(raw)


def _load_gguf_transformer_weights(transformer, path: Path, *, torch_dtype) -> tuple[list[str], list[str]]:
    """Load Wan GGUF transformer weights (mmap + on-the-fly dequant, ComfyUI-GGUF style)."""
    from aiwf.infrastructure.wan.gguf_runtime import load_gguf_transformer_weights

    return load_gguf_transformer_weights(transformer, path, torch_dtype=torch_dtype)


def _load_comfy_fp8_transformer_weights(transformer, path: Path, *, torch_dtype) -> tuple[list[str], list[str]]:
    """Load Comfy scaled-FP8 transformer weights without expanding them to bf16."""
    import torch
    from safetensors import safe_open

    if not _torch_native_fp8_available():
        raise WanUnavailable(
            "This Wan transformer is ComfyUI scaled FP8, but native FP8 execution is unavailable. "
            "Use CUDA torch with float8 and `_scaled_mm`, or use a Diffusers bf16/fp16 transformer."
        )

    non_fp8_state: dict[str, Any] = {}
    loaded_fp8_state_keys: set[str] = set()
    fp8_loaded = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for raw_key in keys:
            raw_key_l = raw_key.lower()
            if raw_key_l.endswith(_COMFY_FP8_METADATA_SUFFIXES):
                continue
            tensor_slice = handle.get_slice(raw_key)
            dtype_name = str(tensor_slice.get_dtype()).upper()
            renamed_key = next(iter(_apply_wan_transformer_key_renames({raw_key: None}).keys()))

            if dtype_name.startswith("F8") and renamed_key.endswith(".weight"):
                module_path = renamed_key.removesuffix(".weight")
                scale_key = raw_key.removesuffix(".weight") + ".weight_scale"
                if scale_key not in keys:
                    scale_key = raw_key.removesuffix(".weight") + ".scale_weight"
                if scale_key not in keys:
                    raise WanUnavailable(f"FP8 tensor '{raw_key}' is missing a weight scale tensor.")

                module = _replace_linear_with_fp8(transformer, module_path)
                weight = handle.get_tensor(raw_key)
                scale = handle.get_tensor(scale_key).to(dtype=torch.float32)
                module.weight = torch.nn.Parameter(weight, requires_grad=False)
                module.weight_scale = scale.reshape(()).to(dtype=torch.float32)
                loaded_fp8_state_keys.add(renamed_key)
                loaded_fp8_state_keys.add(f"{module_path}.weight_scale")
                fp8_loaded += 1
                continue

            tensor = handle.get_tensor(raw_key)
            if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
                tensor = tensor.to(dtype=torch_dtype)
            non_fp8_state[renamed_key] = tensor

    missing, unexpected = transformer.load_state_dict(non_fp8_state, strict=False, assign=True)
    missing = [key for key in missing if key not in loaded_fp8_state_keys]
    logger.info("Loaded %d native FP8 scaled linear weights from %s.", fp8_loaded, path.name)
    return list(missing), list(unexpected)


def _apply_transformer_lora(transformer, lora_path: str | None, *, adapter_name: str, weight: float) -> None:
    if not lora_path:
        return
    try:
        try:
            transformer.load_lora_adapter(lora_path, adapter_name=adapter_name, prefix="transformer")
        except Exception:
            transformer.load_lora_adapter(lora_path, adapter_name=adapter_name, prefix=None)
        transformer.set_adapters(adapter_name, weights=float(weight))
        transformer.fuse_lora(adapter_names=[adapter_name], lora_scale=1.0, safe_fusing=True)
    except Exception as exc:
        raise WanUnavailable(
            f"Could not load Wan LoRA '{Path(lora_path).name}'. Make sure it is a Wan-compatible LoRA."
        ) from exc


def wan_supported() -> bool:
    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: F401

        return True
    except Exception:
        return False


def _require_wan():
    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: F401
    except Exception as exc:  # pragma: no cover - env check
        raise WanUnavailable(
            "Wan video needs a recent diffusers (>=0.35) with WanImageToVideoPipeline. "
            "Update diffusers and transformers, then restart."
        ) from exc

    from aiwf.infrastructure.torch.wan_perf import bootstrap_wan_cuda_settings, describe_missing_comfy_parity
    from aiwf.infrastructure.wan.transformer_runtime import describe_comfy_launcher_parity

    active = bootstrap_wan_cuda_settings()
    if active:
        logger.info("Wan CUDA bootstrap: %s", ", ".join(active))
    for hint in describe_missing_comfy_parity():
        logger.info("Wan perf hint: %s", hint)
    for hint in describe_comfy_launcher_parity():
        logger.info("Wan arch hint: %s", hint)


# Key renames to convert "original/ComfyUI" Wan transformer keys -> diffusers WanTransformer3DModel keys.
# (Subset focused on I2V/TI2V/T2V; harmless/no-op when weights are already in diffusers layout.)
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # norm swap (original uses norm1,norm3,norm2 ordering in places)
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # I2V image embed
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # FLF2V
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # attention
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}


def _apply_wan_transformer_key_renames(sd: dict) -> dict:
    """Return a new state_dict with keys renamed for diffusers WanTransformer3DModel."""
    renamed: dict = {}
    for key, value in sd.items():
        new_key = _normalize_wan_transformer_key(key)
        for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        renamed[new_key] = value
    return renamed


def estimate_gguf_expanded_gb(path: Path) -> float:
    """Rough host RAM needed to fully dequantize a GGUF for diffusers load."""
    from aiwf.infrastructure.wan.transformer_runtime import estimate_gguf_expanded_gb as _estimate

    return _estimate(path)


def _load_gguf_state_dict(path: Path, *, torch_dtype=None) -> dict:
    """DEV STUB: fully dequantize GGUF into diffusers weights. Not for production."""
    import gc

    import gguf
    import numpy as np
    import torch

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    expanded_gb = estimate_gguf_expanded_gb(path)
    if expanded_gb > 12.0 and os.environ.get("AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT") != "1":
        raise WanUnavailable(
            f"GGUF {path.name} would expand to ~{expanded_gb:.0f} GB in RAM when dequantized for diffusers. "
            "AIWF does not run quantized GGUF inference yet (unlike ComfyUI-GGUF). "
            "Use ComfyUI scaled-FP8 `.safetensors` high/low pair instead, or set "
            "AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT=1 if you accept the long load + high RAM use."
        )

    reader = gguf.GGUFReader(str(path))
    total = len(reader.tensors)
    _video_status(f"Dequantizing GGUF {path.name} ({total} tensors, est. ~{expanded_gb:.0f} GB RAM) — this is slow; FP8 safetensors are much faster.")
    sd: dict[str, torch.Tensor] = {}
    for index, tensor in enumerate(reader.tensors, start=1):
        name = tensor.name
        arr = gguf.dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(np.asarray(arr))
        if hasattr(t, "is_floating_point") and t.is_floating_point():
            t = t.to(dtype=torch_dtype)
        sd[name] = t
        if index % 100 == 0 or index == total:
            _video_status(f"GGUF dequant progress: {index}/{total}")
            gc.collect()
    return sd


def _has_wan_text_encoder(path: Path) -> bool:
    text_encoder = path / "text_encoder"
    return (
        (text_encoder / "config.json").is_file()
        and (
            (text_encoder / "model.safetensors").is_file()
            or (text_encoder / "model.safetensors.index.json").is_file()
        )
    )


def _has_wan_tokenizer(path: Path) -> bool:
    tokenizer = path / "tokenizer"
    return (tokenizer / "tokenizer.json").is_file()


def _has_wan_scheduler(path: Path) -> bool:
    return (path / "scheduler" / "scheduler_config.json").is_file()


def _is_wan_components_base(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model_index.json").is_file()
        and _has_wan_text_encoder(path)
        and _has_wan_tokenizer(path)
        and _has_wan_scheduler(path)
    )


def _default_wan_search_roots() -> list[Path]:
    """Return reliable candidate roots for the Wan components base, independent of os.getcwd().

    Uses (in priority):
    - CWD-based (for normal launches from project root)
    - Source-relative: resolve from this file's location up to project root (aiwf/infrastructure/wan -> project)
    - Common fallbacks under the user's home Desktop layout (portable, no hard-coded username)
    """
    roots: list[Path] = []
    # 1. CWD relative (typical when launched via launch.py / webui.bat from project root)
    roots.append(Path("models/wan/Diffusers"))
    roots.append(Path("models/wan"))

    # 2. Module-relative (robust if python started from any cwd, as long as source tree is used)
    try:
        here = Path(__file__).resolve()
        # pipeline.py is at <project>/aiwf/infrastructure/wan/pipeline.py
        # parents[3] -> project root (same convention as RuntimeFlags default)
        project_root = here.parents[3]
        roots.append(project_root / "models" / "wan" / "Diffusers")
        roots.append(project_root / "models" / "wan")
    except Exception:
        pass

    # 3. Portable Desktop fallbacks (covers common Windows dev layout without baking a username)
    home = Path.home()
    for base in (
        home / "Desktop" / "AIWF-Studio",
        home / "Desktop" / "AIWF-Studio - Copy",
    ):
        roots.append(base / "models" / "wan" / "Diffusers")
        roots.append(base / "models" / "wan")

    # Dedup while preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for r in roots:
        rp = r.resolve() if not r.is_absolute() else r
        if rp not in seen:
            seen.add(rp)
            deduped.append(r)
    return deduped


def _find_wan_components_base(search_roots: list[Path] | None = None) -> str | None:
    """Find a full diffusers Wan layout we can use for text_encoder / tokenizer / scheduler when loading single-file weights.

    Upper layers (WanService) should pre-compute and pass an explicit components_base.
    This finder is a best-effort fallback and must be cwd-independent.
    """
    roots = search_roots or _default_wan_search_roots()

    for root in roots:
        preferred = root / "Wan2.2-TI2V-5B-Diffusers"
        if _is_wan_components_base(preferred):
            return str(preferred.resolve())

    # Also try the name directly under each wan root (older layout)
    for root in roots:
        direct = root / "Wan2.2-TI2V-5B-Diffusers"
        rp = direct.resolve() if not direct.is_absolute() else direct
        if _is_wan_components_base(rp):
            return str(rp)

    # Deep scan under wan-containing roots
    for wan_root in roots:
        try:
            base = wan_root if wan_root.is_absolute() else wan_root.resolve()
        except Exception:
            base = wan_root
        if base.exists():
            candidates = [base, *[child for child in base.rglob("*") if child.is_dir()]]
            for child in sorted(candidates):
                if _is_wan_components_base(child):
                    return str(child.resolve())

    return None


def _load_wan_vae(vae_or_base: str, torch_dtype) -> "AutoencoderKLWan":
    """Load Wan VAE from either:
    - a direct path to a .safetensors (Comfy style, recommended for Wan2.1 VAE)
    - or a diffusers folder (vae/ subdir or the root that has vae/)
    - or a base components dir that contains a vae/ subfolder.

    Strips unknown config keys (e.g. 'clip_output': False from some converted 2.2 VAEs)
    that the current diffusers AutoencoderKLWan does not declare.
    """
    from diffusers import AutoencoderKLWan
    import json
    from pathlib import Path as _P

    p = _P(vae_or_base)

    # Direct single-file VAE (most common for user's Comfy Wan 2.1 VAE)
    if p.is_file() and p.suffix.lower() in {".safetensors", ".pth", ".pt"}:
        try:
            _video_status(f"Loading local Wan VAE file: {p.name}")
            vae = AutoencoderKLWan.from_single_file(str(p), torch_dtype=torch_dtype)
            return vae
        except Exception as exc:
            raise WanUnavailable(
                f"Selected VAE '{p.name}' could not be loaded as a Wan VAE. "
                "Choose a Wan VAE file such as 'wan_2.1_vae.safetensors' instead of a generic SD VAE."
            ) from exc

    # If it's a dir that looks like a vae folder itself (has config.json + weights)
    if p.is_dir():
        cfg_path = p / "config.json"
        weights = p / "diffusion_pytorch_model.safetensors"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for bad in ("clip_output", "_diffusers_version"):
                    cfg.pop(bad, None)
                vae = AutoencoderKLWan.from_config(cfg, torch_dtype=torch_dtype)
                if weights.exists():
                    from safetensors.torch import load_file as _load_st
                    sd = _load_st(str(weights))
                    vae.load_state_dict(sd, strict=False)
                    return vae
            except Exception:
                pass
        # or it is a base dir containing vae/ subfolder
        vae_dir = p / "vae"
        cfg_path = vae_dir / "config.json"
        weights = vae_dir / "diffusion_pytorch_model.safetensors"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for bad in ("clip_output", "_diffusers_version"):
                    cfg.pop(bad, None)
                vae = AutoencoderKLWan.from_config(cfg, torch_dtype=torch_dtype)
                if weights.exists():
                    from safetensors.torch import load_file as _load_st
                    sd = _load_st(str(weights))
                    vae.load_state_dict(sd, strict=False)
                    return vae
            except Exception:
                pass

    # Last resort: try from_pretrained on the path (for HF ids or proper diffusers vae)
    try:
        if p.is_dir():
            _video_status(f"Loading local Wan VAE folder: {p}")
        return AutoencoderKLWan.from_pretrained(
            str(p),
            subfolder="vae" if p.is_dir() and (p / "vae").exists() else None,
            torch_dtype=torch_dtype,
            local_files_only=p.is_dir(),
        )
    except Exception:
        # final fallback without subfolder
        return AutoencoderKLWan.from_pretrained(
            str(p),
            torch_dtype=torch_dtype,
            local_files_only=p.is_dir(),
        )


def _load_umt5_text_encoder(text_encoder_dir: Path, torch_dtype):
    """Load Wan's UMT5 text encoder from either HF shards or AIWF's local single-file layout."""
    import json
    import torch
    from transformers import UMT5Config, UMT5EncoderModel

    text_encoder_dir = Path(text_encoder_dir)
    config_path = text_encoder_dir / "config.json"
    single_file = text_encoder_dir / "model.safetensors"

    if config_path.is_file() and single_file.is_file():
        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        _video_status(f"Loading local UMT5 text encoder file: {single_file.name}")
        config = UMT5Config.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        with init_empty_weights():
            model = UMT5EncoderModel(config)

        expected_keys = set(model.state_dict().keys())
        raw = load_file(str(single_file), device="cpu")
        state_dict = {}
        skipped_keys: list[str] = []
        for key, value in raw.items():
            if key in {"spiece_model", "scaled_fp8"}:
                skipped_keys.append(key)
                continue
            if key not in expected_keys:
                skipped_keys.append(key)
                continue
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                value = value.to(dtype=torch_dtype)
            state_dict[key] = value

        if (
            "shared.weight" in state_dict
            and "encoder.embed_tokens.weight" in expected_keys
            and "encoder.embed_tokens.weight" not in state_dict
        ):
            state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            logger.warning("Wan UMT5 text encoder loaded with missing keys: %s", missing[:20])
        if unexpected:
            logger.debug("Wan UMT5 text encoder ignored unexpected keys: %s", unexpected[:20])
        if skipped_keys:
            logger.debug("Wan UMT5 text encoder skipped non-model keys: %s", skipped_keys[:20])
        model.eval()
        return model

    return UMT5EncoderModel.from_pretrained(
        str(text_encoder_dir), torch_dtype=torch_dtype, local_files_only=text_encoder_dir.is_dir()
    )


def _materialize_wan_rope_buffers(transformer) -> int:
    """Recompute Wan rotary-embedding buffers on a real (non-meta) device.

    ``WanRotaryPosEmbed`` registers ``freqs_cos`` / ``freqs_sin`` as *non-persistent*
    buffers computed in ``__init__`` purely from config. They therefore:
      * appear in NO checkpoint (so ``load_state_dict`` never fills them), and
      * are created on the ``meta`` device when the module is built under
        ``init_empty_weights()``.

    Left on meta, the first ``.to()`` / ``enable_*_cpu_offload`` / pinned-cache move
    crashes with ``NotImplementedError: Cannot copy out of meta tensor; no data!``.
    Because the values depend only on config (not trained weights), we rebuild a
    fresh rope module on CPU and copy its real buffers in. Returns how many modules
    were fixed.
    """
    try:
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    except Exception:
        return 0

    fixed = 0
    for module in transformer.modules():
        if not isinstance(module, WanRotaryPosEmbed):
            continue
        needs_fix = any(
            getattr(module, name, None) is not None and module._buffers.get(name) is not None
            and module._buffers[name].device.type == "meta"
            for name in ("freqs_cos", "freqs_sin")
        )
        if not needs_fix:
            continue
        # Rebuilt outside any init_empty_weights() context -> real CPU tensors.
        fresh = WanRotaryPosEmbed(
            attention_head_dim=module.attention_head_dim,
            patch_size=module.patch_size,
            max_seq_len=module.max_seq_len,
        )
        module.register_buffer("freqs_cos", fresh.freqs_cos.detach().clone(), persistent=False)
        module.register_buffer("freqs_sin", fresh.freqs_sin.detach().clone(), persistent=False)
        fixed += 1
    if fixed:
        logger.debug("Materialized rope buffers on %d WanRotaryPosEmbed module(s).", fixed)
    return fixed


def _empty_wan_transformer(config):
    """Create a Wan transformer shell without random parameter allocation.

    The shell is built under ``init_empty_weights`` (all weights on meta, filled
    later from the checkpoint), but the rotary-embedding buffers are immediately
    materialized on CPU because they are config-derived and never appear in any
    checkpoint — otherwise they would remain meta and crash every later device move.
    """
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    with init_empty_weights():
        model = WanTransformer3DModel.from_config(config)
    _materialize_wan_rope_buffers(model)
    return model


def _boundary_ratio_for_step_split(scheduler, *, total_steps: int, high_steps: int) -> float:
    """Map a Comfy-style step split to Diffusers' boundary-timestep ratio."""
    import torch

    total = max(1, int(total_steps))
    high = max(1, min(int(high_steps), total - 1))
    scheduler.set_timesteps(total, device=torch.device("cpu"))
    timesteps = scheduler.timesteps.detach().cpu().float()
    if high >= len(timesteps):
        return 1.0
    previous_t = float(timesteps[high - 1])
    next_t = float(timesteps[high])
    boundary_timestep = (previous_t + next_t) / 2.0
    train_steps = float(getattr(scheduler.config, "num_train_timesteps", 1000) or 1000)
    return min(1.0, max(0.0, boundary_timestep / train_steps))


def _free_cuda_memory() -> None:
    """Best-effort VRAM cleanup before loading heavy Wan transformers."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _recover_cuda_after_pin_memory_failure() -> None:
    """Clear pending CUDA allocator/error state after page-locked host allocation fails."""
    import gc

    for _ in range(2):
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass


def _wan_cache_mode(offload: str, *, fast_fp8_pair: bool) -> str:
    """Return how the dual-transformer CPU/GPU cache should behave.

    - ``none``: accelerate owns placement (sequential / generic model offload).
    - ``gpu_swap``: keep high/low on CPU between stages; stream active stage to GPU
      (model offload + native FP8, 12–16 GB cards).
    - ``full``: legacy pinned-CPU cache (only when the full pipeline stays on GPU).
    """
    if offload == "model" and fast_fp8_pair:
        return "gpu_swap"
    if offload == "none":
        return "full"
    return "none"


def _ensure_wan_attention_processors(transformer, name: str = "transformer") -> None:
    """Guarantee WanAttention modules use WanAttnProcessor (not AttnProcessor2_0).

    LoRA fuse / accelerate offload hooks can leave a generic processor attached;
    that crashes with ``AttributeError: 'WanAttention' object has no attribute 'spatial_norm'``.
    """
    if transformer is None:
        return
    try:
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor
    except Exception:
        return

    fixed = 0
    for module in transformer.modules():
        if not isinstance(module, WanAttention):
            continue
        proc = getattr(module, "processor", None)
        if proc is None or proc.__class__.__name__ not in ("WanAttnProcessor", "WanAttnProcessor2_0"):
            module.set_processor(WanAttnProcessor())
            fixed += 1
        if not hasattr(module, "spatial_norm"):
            module.spatial_norm = None
    if fixed:
        logger.debug("Reset %d WanAttention processor(s) on %s", fixed, name)


def _apply_wan_attention_optimizations(transformer, name: str = "transformer") -> None:
    """Apply Wan-specific attention/conv optimizations (SDP flash, sageattention, channels_last)."""
    from aiwf.infrastructure.torch.wan_perf import apply_wan_transformer_optimizations
    from aiwf.infrastructure.wan.sliced_sampler import install_temporal_chunk_forward

    active = apply_wan_transformer_optimizations(transformer, name=name)
    if active:
        _video_status(f"{name} optimizations: {', '.join(active)}")
    if install_temporal_chunk_forward(transformer, name=name):
        _video_status(
            f"{name}: temporal chunk denoise active "
            f"(chunk={getattr(transformer, '_aiwf_chunk_size', '?')}, "
            f"overlap={getattr(transformer, '_aiwf_chunk_overlap', '?')})."
        )

# Dual-Buffer Virtual VRAM Cache per user's detailed guidance.
# Keeps heavy models (14B Wan high/low, text encoder, vae) pinned in CPU RAM.
# Fast PCIe streaming for swaps instead of disk reloads on del/reload.
# Used to coordinate high <-> low without co-existing in VRAM, and keep objects alive.
class AIWFModelCacheManager:
    def __init__(self, device="cuda"):
        self.device = device
        # Active storage in CPU RAM (prevents disk reads on swap)
        self.cpu_cache = {}
        # Tracking what is currently occupying VRAM space
        self.active_in_vram = None
        # Set to False on the first pin_memory failure so subsequent models
        # never attempt pinning.  A failed pin can leave the CUDA driver context
        # in a dirty state; retrying (especially from a background thread while
        # CUDA runs on the main thread) reproduces the 0xC0000005 AV crash.
        self._global_pin_enabled = True

    def register_model(self, model_key, model_object, *, pin: bool = True):
        """
        Store model weights in host CPU RAM for fast high/low swaps.

        Page-locked pinning is optional — on 16 GB cards with a 14B transformer,
        pinning can fail or trigger spurious CUDA OOM from prior GPU pressure, so
        we fall back to ordinary CPU tensors.
        """
        if model_key in self.cpu_cache:
            return
        logger.info("[AIWF] Registering %s to CPU cache (pin=%s)", model_key, pin)

        _free_cuda_memory()
        try:
            model_object.to("cpu")
        except Exception as exc:
            logger.warning("[AIWF] Could not move %s to CPU before cache register: %s", model_key, exc)

        # Pinning is page-locked host memory backed by the CUDA driver. On a 16 GB
        # card a 14B FP8 transformer can exhaust it; the FIRST failure leaves the CUDA
        # context in an error state, and continuing to call pin_memory on every
        # remaining tensor spams hundreds of warnings and can escalate to a hard
        # process crash (0xC0000005 access violation). So we disable pinning for the
        # WHOLE model on the first failure and fall back to ordinary CPU tensors —
        # swaps still work, just without the page-locked speedup.
        #
        # Critically, we also update _global_pin_enabled so that the NEXT model
        # (e.g. wan_low loaded in a background thread) never tries pin_memory either.
        # Calling pin_memory from a background thread while CUDA is active on the
        # main thread is what triggers the 0xC0000005 crash in practice.
        pin_state = {"enabled": bool(pin) and self._global_pin_enabled, "warned": False}

        def _store_tensor(tensor) -> None:
            if tensor is None:
                return
            if tensor.device.type != "cpu":
                tensor.data = tensor.data.to("cpu", non_blocking=False)
            if not pin_state["enabled"]:
                return
            try:
                tensor.data = tensor.data.pin_memory()
            except Exception as exc:  # RuntimeError / CUDA OOM (and anything else)
                pin_state["enabled"] = False
                self._global_pin_enabled = False  # propagate: no more pinning on this cache
                if not pin_state["warned"]:
                    pin_state["warned"] = True
                    logger.warning(
                        "[AIWF] pin_memory failed for %s (%s); keeping it on unpinned CPU RAM "
                        "for the rest of this model (this is normal for a 14B FP8 model on 16 GB).",
                        model_key, exc,
                    )
                    # pin_memory() failing with a CUDA error can leave the CUDA driver
                    # in a dirty state — subsequent CUDA allocs (e.g. text_encoder.to("cuda"))
                    # will report false OOMs until the async error is cleared.
                    # synchronize + empty_cache flushes the error queue.
                    _recover_cuda_after_pin_memory_failure()

        for param in model_object.parameters():
            _store_tensor(param)
            if param.grad is not None:
                param.grad = None

        for buffer in model_object.buffers():
            _store_tensor(buffer)

        self.cpu_cache[model_key] = model_object

    def load_to_vram(self, target_key):
        """
        Instantly streams from pinned CPU RAM to VRAM via PCIe (non-blocking + sync).
        Automatically unloads previous to avoid VRAM co-existence.
        """
        import torch

        if self.active_in_vram == target_key:
            return

        if self.active_in_vram is not None:
            self.unload_from_vram(self.active_in_vram)

        if target_key not in self.cpu_cache:
            print(f"[AIWF] Warning: {target_key} not registered in cache; skipping fast load.")
            return

        print(f"[AIWF] Streaming {target_key} to VRAM via PCIe (fast swap, no disk)...")
        target_model = self.cpu_cache[target_key]

        target_model.to(self.device, non_blocking=True)
        self.active_in_vram = target_key

        torch.cuda.current_stream().synchronize()

    def unload_from_vram(self, target_key):
        """
        Evicts from VRAM back to pinned CPU RAM (keeps object alive, no del/reload).
        Aggressive flush.
        """
        import gc

        import torch

        if target_key not in self.cpu_cache:
            return

        print(f"[AIWF] Evicting {target_key} back to Pinned CPU RAM (for headroom)...")
        model = self.cpu_cache[target_key]

        model.to("cpu", non_blocking=True)
        self.active_in_vram = None

        # Aggressive context flushing
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


class WanI2VBackend:
    """Loads a Wan image-to-video pipeline once and reuses it across renders.
    Uses AIWFModelCacheManager for pinned-CPU zero-disk swaps between high/low 14B
    and lighter components (text/VAE). This eliminates disk reloads on swaps and
    keeps models in RAM for instant PCIe .to() instead of del + safetensors load.
    """

    def __init__(self, *, async_offload: bool = True, pinned_memory: bool = True) -> None:
        self._pipe = None
        self._key = None
        self._cache_mode = "none"
        self._async_offload = bool(async_offload)
        self._pinned_memory = bool(pinned_memory)
        self.cache = AIWFModelCacheManager(device="cuda")
        self._reset_low_preload_state()

    def _reset_low_preload_state(self) -> None:
        thread = getattr(self, "_low_preload_thread", None)
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=0.1)
            except Exception:
                pass
        self._preloaded_low = None
        self._low_preload_spec: dict[str, Any] | None = None
        self._low_preload_thread: threading.Thread | None = None
        self._low_preload_done = threading.Event()
        self._low_preload_done.set()
        self._low_preload_started = False
        self._low_preload_error: BaseException | None = None

    def _materialize_wan_transformer(
        self,
        target,
        weight_path: str,
        *,
        label: str,
        lora_path: str | None,
        lora_scale: float,
        lora_adapter: str,
    ) -> tuple[list[str], list[str]]:
        import torch

        pp = Path(weight_path)
        if pp.suffix.lower() == ".gguf":
            _video_status(f"Loading GGUF {label} (mmap + on-the-fly dequant): {pp.name}")
            miss, unex = _load_gguf_transformer_weights(target, pp, torch_dtype=torch.bfloat16)
        elif pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(pp):
            _video_status(f"Loading native Comfy FP8 {label}: {pp.name}")
            miss, unex = _load_comfy_fp8_transformer_weights(target, pp, torch_dtype=torch.bfloat16)
        else:
            sd = _load_transformer_state_dict(weight_path, label=label)
            _video_status(f"Applying {label}: {pp.name}")
            miss, unex = target.load_state_dict(sd, strict=False, assign=True)
            del sd
        if lora_path:
            _video_status(f"Applying {label} LoRA: {Path(lora_path).name}")
        _apply_transformer_lora(target, lora_path, adapter_name=lora_adapter, weight=lora_scale)
        _ensure_wan_attention_processors(target, label)
        _apply_wan_attention_optimizations(target, label)
        return list(miss), list(unex)

    def _run_low_preload_worker(self) -> None:
        spec = self._low_preload_spec
        if spec is None:
            self._low_preload_done.set()
            return
        try:
            low = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
            miss, unex = self._materialize_wan_transformer(
                low,
                spec["low_path"],
                label="low-noise transformer",
                lora_path=spec.get("low_lora_path"),
                lora_scale=float(spec.get("low_lora_scale", 1.0)),
                lora_adapter="wan_low_lora",
            )
            if miss or unex:
                logger.warning(
                    "Low-noise background preload (%s): %d missing %d unexpected",
                    Path(spec["low_path"]).name,
                    len(miss),
                    len(unex),
                )
            self._preloaded_low = low
            if spec.get("use_cache"):
                self.cache.register_model("wan_low", low, pin=bool(spec.get("pin_tensors")))
            _video_status("Background low-noise transformer ready in CPU cache (fast VRAM swap at boundary).")
        except BaseException as exc:
            self._low_preload_error = exc
            logger.exception("Background low-noise transformer preload failed")
        finally:
            self._low_preload_done.set()

    def _maybe_start_background_low_preload(self) -> None:
        """Start loading low-noise weights to CPU while high stage runs on GPU."""
        if self._low_preload_spec is None or self._preloaded_low is not None:
            return
        if self._low_preload_started:
            return
        # Disk-sequential mode: don't start a background thread — low will load
        # synchronously at the boundary point AFTER wan_high is freed from the
        # CPU cache.  Starting a thread here (with CUDA active + no pinned memory)
        # is exactly what causes the 0xC0000005 AV crash on Windows.
        if not self._async_offload:
            return
        self._low_preload_started = True
        self._low_preload_done.clear()
        self._low_preload_error = None
        _video_status(
            "Background: loading low-noise transformer to CPU while high stage runs "
            "(boundary swap will stream to VRAM, not reload from disk)."
        )
        self._low_preload_thread = threading.Thread(
            target=self._run_low_preload_worker,
            name="aiwf-wan-low-preload",
            daemon=True,
        )
        self._low_preload_thread.start()

    def _ensure_low_preloaded(self) -> None:
        if self._preloaded_low is not None:
            return
        if self._low_preload_spec is None:
            raise WanUnavailable("Low-noise transformer was not configured for this pipeline.")
        if not self._low_preload_started:
            _video_status("Low-noise transformer not preloaded yet — loading now before boundary swap.")
            self._run_low_preload_worker()
            return
        if not self._low_preload_done.is_set():
            _video_status("Waiting for background low-noise preload to finish...")
        self._low_preload_done.wait()
        if self._low_preload_error is not None:
            raise WanUnavailable(
                f"Low-noise transformer preload failed: {self._low_preload_error}"
            ) from self._low_preload_error
        if self._preloaded_low is None:
            raise WanUnavailable("Low-noise transformer preload finished without a model.")

    def available(self) -> bool:
        return wan_supported()

    def unload(self) -> None:
        self._reset_low_preload_state()
        # Drop any GPU-resident cached stage before releasing references so the
        # VRAM is actually reclaimed (not just unreferenced).
        try:
            if self.cache.active_in_vram is not None:
                self.cache.unload_from_vram(self.cache.active_in_vram)
        except Exception:
            pass
        self._pipe = None
        self._key = None
        self._cache_mode = "none"
        self._preloaded_low = None
        self.cache.cpu_cache.clear()
        self.cache.active_in_vram = None
        _free_cuda_memory()

    def _aspect_resize(self, pipe, image, max_area: int):
        """Resize the image to a Wan-valid size near ``max_area`` (model-aware)."""
        import numpy as np

        ar = image.height / image.width
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = max(mod, round(np.sqrt(max_area * ar)) // mod * mod)
        width = max(mod, round(np.sqrt(max_area / ar)) // mod * mod)
        return image.resize((width, height)), int(height), int(width)

    def _ensure(
        self,
        *,
        high_noise_model_id: str | None = None,
        low_noise_model_id: str | None = None,
        boundary_ratio: float | None = None,
        vae_id: str | None = None,
        high_noise_lora_id: str | None = None,
        high_noise_lora_scale: float = 1.0,
        low_noise_lora_id: str | None = None,
        low_noise_lora_scale: float = 1.0,
        components_base: str | None = None,
        offload: str,
        flow_shift: float,
    ):
        # Dual high/low is the only supported layout for Wan 2.2 I2V.
        if not (high_noise_model_id and low_noise_model_id):
            raise WanUnavailable(
                "Wan 2.2 image-to-video requires both a high-noise and a low-noise model "
                "(two-stage transformer pair). Select both before generating."
            )

        key = (
            "dual",
            high_noise_model_id,
            low_noise_model_id,
            boundary_ratio,
            vae_id or "default",
            high_noise_lora_id,
            round(float(high_noise_lora_scale), 3),
            low_noise_lora_id,
            round(float(low_noise_lora_scale), 3),
            components_base or "auto",
            offload,
            round(float(flow_shift), 3),
        )

        if self._pipe is not None and self._key == key:
            return self._pipe

        # Switching to a different model set: fully evict the previous pipeline and
        # its pinned CPU cache (and flush VRAM) BEFORE building the new one. Otherwise
        # the prior generation's high/low transformers linger in RAM/VRAM and cause
        # OOM on the new load — even at low resolution, because it's weight memory,
        # not activation memory, that dominates for 14B models.
        if self._pipe is not None or self.cache.cpu_cache:
            _video_status("Releasing previous video pipeline before loading the new model set.")
            self.unload()

        _require_wan()
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        logger.info(
            "Loading Wan I2V pipeline (dual high/low): high=%s low=%s boundary=%s vae=%s (offload=%s)",
            high_noise_model_id, low_noise_model_id, boundary_ratio, vae_id, offload,
        )
        _video_status(
            f"Loading video pipeline with local base components and dual transformers ({Path(high_noise_model_id).name} / {Path(low_noise_model_id).name})."
        )
        fast_fp8_pair = (
            (
                _torch_native_fp8_available()
                and _is_native_comfy_fp8_transformer(high_noise_model_id)
                and _is_native_comfy_fp8_transformer(low_noise_model_id)
            )
            or (
                _is_gguf_transformer(high_noise_model_id)
                and _is_gguf_transformer(low_noise_model_id)
            )
        )
        cache_mode = _wan_cache_mode(offload, fast_fp8_pair=fast_fp8_pair)

        pipe = self._load_dual_pipeline(
            high_path=high_noise_model_id,
            low_path=low_noise_model_id,
            boundary_ratio=boundary_ratio or 0.875,
            vae_id=vae_id,
            high_lora_path=high_noise_lora_id,
            high_lora_scale=high_noise_lora_scale,
            low_lora_path=low_noise_lora_id,
            low_lora_scale=low_noise_lora_scale,
            components_base=components_base,
            offload=offload,
            cache_mode=cache_mode,
            flow_shift=flow_shift,
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=float(flow_shift))

        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            if fast_fp8_pair:
                _video_status(
                    "Using fast quantized placement: keeping the active Wan transformer on GPU while offloading text encoder/VAE."
                )
                original_seq = getattr(pipe, "model_cpu_offload_seq", None)
                original_exclude = list(getattr(pipe, "_exclude_from_cpu_offload", []) or [])
                # Build safe offload seq - some manual Wan assemblies don't have image_encoder
                seq_parts = ["text_encoder"]
                if hasattr(pipe, "image_encoder") and getattr(pipe, "image_encoder", None) is not None:
                    seq_parts.append("image_encoder")
                seq_parts.append("vae")
                pipe.model_cpu_offload_seq = "->".join(seq_parts)
                pipe._exclude_from_cpu_offload = sorted(
                    set(original_exclude).union({"transformer", "transformer_2"})
                )
                try:
                    pipe.enable_model_cpu_offload()
                finally:
                    if original_seq:
                        pipe.model_cpu_offload_seq = original_seq
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        _ensure_wan_attention_processors(pipe.transformer, "high-noise transformer")
        if cache_mode == "gpu_swap":
            _free_cuda_memory()
            if not self.cache._global_pin_enabled:
                # Disk-sequential mode: pin_memory failed, meaning wan_high (~14GB) is
                # stored in ordinary (unpinned) CPU RAM.  If we load it to VRAM NOW,
                # encode_prompt will OOM because the text encoder also needs VRAM (UMT5
                # is ~4-5 GB) and 14 + 5 > 16 GB.
                #
                # Instead, register a one-shot pre-forward hook on the real transformer.
                # The hook fires just before the FIRST denoising forward call (after
                # encode_prompt has finished and released its VRAM), loads wan_high then,
                # and immediately removes itself so it never fires again.
                _loaded = [False]

                def _deferred_vram_load(module, args, _cache=self.cache):
                    if not _loaded[0]:
                        _loaded[0] = True
                        _video_status(
                            "Disk-sequential: deferred wan_high VRAM load "
                            "(after encode_prompt released its VRAM)."
                        )
                        _cache.load_to_vram("wan_high")

                pipe.transformer.register_forward_pre_hook(_deferred_vram_load)
                _video_status(
                    "Disk-sequential: wan_high VRAM load deferred to first denoising step "
                    "(encode_prompt needs that VRAM first)."
                )
            else:
                # Normal path: pre-load to VRAM now that text encoder / VAE offload
                # hooks are installed and the previous pipeline is fully evicted.
                self.cache.load_to_vram("wan_high")

        for method in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, method)()
            except Exception:
                pass

        self._pipe = pipe
        self._key = key
        self._cache_mode = cache_mode
        return pipe

    def _load_dual_pipeline(
        self,
        high_path: str,
        low_path: str,
        *,
        boundary_ratio: float,
        vae_id: str | None = None,
        high_lora_path: str | None = None,
        high_lora_scale: float = 1.0,
        low_lora_path: str | None = None,
        low_lora_scale: float = 1.0,
        components_base: str | None = None,
        offload: str,
        cache_mode: str = "none",
        flow_shift: float,
    ):
        """Load a WanImageToVideoPipeline configured with transformer (high-noise) + transformer_2 (low-noise)."""
        import torch
        from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import AutoTokenizer

        base = components_base or _find_wan_components_base()
        if base is None:
            raise WanUnavailable(
                "Dual high/low Wan models require a components base (text_encoder + tokenizer + scheduler + model_index.json). "
                "WanService preflight should locate and pass it via components_base. "
                "Ensure models/wan/Diffusers/Wan2.2-TI2V-5B-Diffusers (or equivalent) has the required files. "
                "See docs/WAN_LOCAL_COMPONENTS.md. (No base found via explicit value or search.)"
            )
        _video_status(f"Using local video base: {base}")

        # Prefer explicit user VAE (your Comfy Wan 2.1 VAE .safetensors) because "oddly wan 2.2 uses 2.1 vae".
        # Falls back to cleaned VAE from the base components folder.
        if vae_id:
            vae = _load_wan_vae(vae_id, torch_dtype=torch.float32)
        else:
            vae = _load_wan_vae(base, torch_dtype=torch.float32)

        base_path = Path(base)
        _video_status("Loading local text encoder, tokenizer, and scheduler.")
        text_encoder = _load_umt5_text_encoder(base_path / "text_encoder", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(str(base_path / "tokenizer"), local_files_only=True)
        scheduler = UniPCMultistepScheduler.from_pretrained(
            str(base_path / "scheduler"), local_files_only=True
        )

        use_cache = cache_mode in ("full", "gpu_swap")
        pin_tensors = cache_mode == "full" or (cache_mode == "gpu_swap" and self._pinned_memory)
        _video_status("Building local Wan A14B I2V pipeline components.")
        _video_status("Preparing empty high-noise transformer stage.")
        high_trans = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
        pipe = WanImageToVideoPipeline(
            transformer=high_trans,
            transformer_2=None,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=float(boundary_ratio),
        )
        _video_status("Base video pipeline loaded.")

        # Inject high into the primary transformer (overwrites the base one in-place)
        high_pp = Path(high_path)
        if high_pp.suffix.lower() == ".gguf":
            _video_status(f"Loading GGUF high-noise transformer (mmap + on-the-fly dequant): {high_pp.name}")
            miss_h, unex_h = _load_gguf_transformer_weights(
                pipe.transformer, high_pp, torch_dtype=torch.bfloat16
            )
        elif high_pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(high_pp):
            _video_status(f"Loading native Comfy FP8 high-noise transformer: {high_pp.name}")
            miss_h, unex_h = _load_comfy_fp8_transformer_weights(
                pipe.transformer, high_pp, torch_dtype=torch.bfloat16
            )
        else:
            high_sd = _load_transformer_state_dict(high_path, label="High-noise transformer")
            _video_status(f"Applying high-noise transformer: {Path(high_path).name}")
            miss_h, unex_h = pipe.transformer.load_state_dict(high_sd, strict=False, assign=True)
            del high_sd
        if miss_h or unex_h:
            logger.warning("High-noise weights (%s) vs base: %d missing %d unexpected", Path(high_path).name, len(miss_h), len(unex_h))
        if high_lora_path:
            _video_status(f"Applying high-noise LoRA: {Path(high_lora_path).name}")
        _apply_transformer_lora(
            pipe.transformer,
            high_lora_path,
            adapter_name="wan_high_lora",
            weight=high_lora_scale,
        )
        _ensure_wan_attention_processors(pipe.transformer, "high-noise transformer")

        # Apply memory-efficient attention (SDP + channels_last) to the loaded high stage.
        _apply_wan_attention_optimizations(pipe.transformer, "high-noise transformer")

        if use_cache:
            self.cache.register_model("wan_high", pipe.transformer, pin=pin_tensors)
            if cache_mode == "full":
                self.cache.load_to_vram("wan_high")

        low_pp = Path(low_path)
        # Disk-sequential mode: triggered when pin_memory failed for wan_high.
        #
        # Two problems occur when we try to load both 14B FP8 models at once:
        # 1. Background thread + safetensors mmap + active CUDA → 0xC0000005 AV crash.
        # 2. Synchronous load while wan_high occupies CPU cache → 2×14GB ≈ 28GB peaks
        #    RAM, which is too tight on a 32GB machine and causes another AV.
        #
        # Solution — disk-sequential: defer wan_low entirely. At the boundary point
        # _release_high_stage will DELETE wan_high from the CPU cache (freeing ~14GB),
        # then _load_low_stage loads wan_low from disk into the freed space.
        # No background thread, never two 14B models in RAM simultaneously.
        disk_sequential = use_cache and not self.cache._global_pin_enabled
        if disk_sequential:
            if self._async_offload:
                self._async_offload = False
            _video_status(
                "Disk-sequential mode: wan_high will be freed from CPU cache at the "
                "boundary point and wan_low will load from disk then. "
                "(pin_memory unavailable — loading both 14B models simultaneously would OOM.)"
            )
        defer_low_preload = disk_sequential or (cache_mode == "gpu_swap" and self._async_offload)
        self._reset_low_preload_state()

        if defer_low_preload:
            self._low_preload_spec = {
                "low_path": low_path,
                "low_lora_path": low_lora_path,
                "low_lora_scale": low_lora_scale,
                "use_cache": use_cache,
                "pin_tensors": pin_tensors,
            }
            _video_status(
                "Deferring low-noise load: will preload to CPU in background during high-stage denoising."
            )
            self._preloaded_low = None
        else:
            self._low_preload_spec = None
            _video_status("Preloading low-noise transformer weights to CPU.")
            preloaded_low = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
            miss_l_pre, unex_l_pre = self._materialize_wan_transformer(
                preloaded_low,
                low_path,
                label="low-noise transformer",
                lora_path=low_lora_path,
                lora_scale=low_lora_scale,
                lora_adapter="wan_low_lora",
            )
            if miss_l_pre or unex_l_pre:
                logger.warning(
                    "Low-noise preload (%s): %d missing %d unexpected",
                    low_pp.name,
                    len(miss_l_pre),
                    len(unex_l_pre),
                )
            self._preloaded_low = preloaded_low
            if use_cache:
                self.cache.register_model("wan_low", preloaded_low, pin=pin_tensors)

        def _release_high_stage():
            _video_status(
                "High-noise stage complete; releasing high transformer for low-stage headroom."
            )
            if use_cache:
                self.cache.unload_from_vram("wan_high")
                if disk_sequential:
                    # Free wan_high from the CPU cache BEFORE loading wan_low from disk.
                    # Without this, both 14B FP8 models (~28GB) would live in RAM at
                    # the same time, which OOMs a 32GB machine and causes 0xC0000005.
                    _video_status(
                        "Disk-sequential: evicting wan_high from CPU cache to make room "
                        "for wan_low disk load (~14GB freed)."
                    )
                    self.cache.cpu_cache.pop("wan_high", None)
                    if cache_mode == "gpu_swap":
                        pipe.transformer = None
            elif pipe.transformer is not None:
                try:
                    pipe.transformer.to("cpu")
                except Exception:
                    pass

            import gc
            import torch
            if pipe.transformer is not None and cache_mode != "gpu_swap":
                pipe.transformer = None
            for _ in range(2):
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        def _load_low_stage():
            _video_status("High stage done — swapping to low-noise transformer.")
            import gc

            self._ensure_low_preloaded()
            gc.collect()
            _free_cuda_memory()
            low_trans = self._preloaded_low
            if use_cache:
                self.cache.load_to_vram("wan_low")
            elif offload != "sequential":
                try:
                    low_trans.to("cuda")
                except Exception as exc:
                    logger.warning("Could not move low-noise transformer to CUDA: %s", exc)
            return low_trans

        if defer_low_preload:
            _video_status("Low-noise transformer will load in background; boundary swap streams from CPU cache.")
        else:
            _video_status("Low-noise transformer preloaded to CPU; boundary swap should be fast.")
        pipe.transformer_2 = _new_lazy_wan_transformer(
            WAN_I2V_A14B_TRANSFORMER_CONFIG,
            dtype=torch.bfloat16,
            load_model=_load_low_stage,
            before_load=_release_high_stage,
        )

        # Tell the pipeline about the switch point (registered into self.config)
        # Typical values for Wan2.2 14B I2V high/low splits are around 0.875
        pipe.register_to_config(boundary_ratio=float(boundary_ratio))

        logger.info(
            "Configured dual-stage Wan pipeline: high=%s, low=%s, boundary_ratio=%s (from base %s)",
            Path(high_path).name, Path(low_path).name, boundary_ratio, Path(base).name
        )
        return pipe

    def generate(self, request, image, *, on_progress=None, should_cancel=None):
        """Run image->video. Returns list of frames (PIL or numpy arrays from VAE decode)."""
        _require_wan()
        import torch

        # Wan 2.2 image-to-video ALWAYS runs a two-stage high-noise + low-noise
        # transformer pair. There is no single-model path — both must be set
        # (this holds even when using LoRAs: you still need one high and one low).
        if not (getattr(request, "uses_dual_transformers", None) and request.uses_dual_transformers()):
            raise WanUnavailable(
                "Wan 2.2 image-to-video needs BOTH a high-noise and a low-noise model. "
                "Select a High noise model and a Low noise model — Wan 2.2 always uses a "
                "two-stage high/low transformer pair (required even when using LoRAs)."
            )

        pipe = self._ensure(
            high_noise_model_id=request.high_noise_model_id,
            low_noise_model_id=request.low_noise_model_id,
            boundary_ratio=getattr(request, "boundary_ratio", None),
            vae_id=getattr(request, "vae_id", None),
            high_noise_lora_id=getattr(request, "high_noise_lora_id", None),
            high_noise_lora_scale=float(getattr(request, "high_noise_lora_scale", 1.0) or 1.0),
            low_noise_lora_id=getattr(request, "low_noise_lora_id", None),
            low_noise_lora_scale=float(getattr(request, "low_noise_lora_scale", 1.0) or 1.0),
            components_base=getattr(request, "components_base", None),
            offload=request.offload,
            flow_shift=request.flow_shift,
        )

        base = image.convert("RGB")
        resized, height, width = self._aspect_resize(pipe, base, request.max_area)

        generator = None
        if int(request.seed) >= 0:
            generator = torch.Generator(device="cpu").manual_seed(int(request.seed))

        steps = int(request.steps)
        high_steps = max(1, min(int(getattr(request, "high_noise_steps", steps // 2) or steps // 2), steps - 1))
        stage_boundary_ratio = _boundary_ratio_for_step_split(
            pipe.scheduler,
            total_steps=steps,
            high_steps=high_steps,
        )
        pipe.register_to_config(boundary_ratio=stage_boundary_ratio)

        from aiwf.infrastructure.torch.wan_vram import (
            decode_wan_video_latents,
            log_step_vram_trace,
            reset_cuda_peak_stats,
            wan_inference_context,
        )

        from aiwf.infrastructure.wan.sliced_sampler import describe_temporal_chunk_settings

        if request.offload == "sequential":
            _video_status(
                "WARNING: Sequential offload moves every layer over PCIe each step (~3-10x slower). "
                "On 16 GB + FP8 safetensors use Model offload instead."
            )

        _video_status(describe_temporal_chunk_settings(request.normalized_frames()))

        step_times: list[float] = []
        last_step_at = time.perf_counter()
        reset_cuda_peak_stats()

        def _callback(pipe_obj, step_index, _timestep, kwargs):
            nonlocal last_step_at
            if should_cancel and should_cancel():
                setattr(pipe_obj, "_interrupt", True)
            current = step_index + 1
            now = time.perf_counter()
            elapsed = now - last_step_at
            last_step_at = now
            step_times.append(elapsed)
            if on_progress:
                on_progress(current, steps)
            stage = "low" if current > high_steps else "high"
            if stage == "high" and current == 1:
                self._maybe_start_background_low_preload()
            _video_status(
                f"Diffusion step {current}/{steps} ({stage} stage, {elapsed:.1f}s since last step)"
            )
            log_step_vram_trace(
                step=current,
                total=steps,
                stage=stage,
                elapsed_s=elapsed,
                status_print=_video_status,
            )

            # After the very last denoising step, aggressively unload the low (and any remaining high) transformer
            # *before* the pipeline proceeds to VAE decode. This gives maximum headroom for the decode of 49 frames
            # and the return path, preventing freeze after "8/8 steps" due to VRAM pressure from the 14B models.
            if current == steps:
                _video_status("All denoising steps complete; releasing transformers before VAE decode.")
                try:
                    if self._cache_mode in ("full", "gpu_swap"):
                        self.cache.unload_from_vram("wan_high")
                        self.cache.unload_from_vram("wan_low")
                    else:
                        for tname in ("transformer", "transformer_2"):
                            module = getattr(pipe_obj, tname, None)
                            if module is not None:
                                try:
                                    module.to("cpu")
                                except Exception:
                                    pass
                    _video_status("Transformers released post-steps; VAE decode should have more VRAM.")
                except Exception as exc:
                    logger.debug("Post-steps transformer release non-fatal: %s", exc)

            return kwargs

        # Match Comfy's two KSampler Advanced stages: high runs the first
        # high_steps, returns latents/noise, then low continues from there.
        call_kwargs = dict(
            image=resized,
            prompt=request.prompt or "",
            negative_prompt=(request.negative_prompt or None),
            height=height,
            width=width,
            num_frames=request.normalized_frames(),
            guidance_scale=float(request.guidance_scale),
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="latent",
        )
        _video_status(
            f"Running video diffusion: {steps} steps ({high_steps} high / {steps - high_steps} low), "
            f"{request.normalized_frames()} frames."
        )
        _video_status(
            "Invoking WanImageToVideoPipeline (UMT5 encode + VAE image cond + dual transformer denoising). "
            "GPU will ramp up during actual steps. Watch for step callbacks. "
            "Prep can take 30-180s on 16GB for 14B."
        )
        for method in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, method)()
            except Exception:
                pass
        self._maybe_start_background_low_preload()
        try:
            with wan_inference_context():
                output = pipe(**call_kwargs)
        except Exception:
            logger.exception("Wan pipeline execution failed")
            raise
        if step_times:
            avg = sum(step_times[1:]) / max(len(step_times) - 1, 1) if len(step_times) > 1 else step_times[0]
            _video_status(f"Average step time (after warmup): {avg:.1f}s")


        frames_tensor = output.frames[0]  # [T, H, W, 3] (uint8 or float)
        frames: list
        if isinstance(frames_tensor, list):
            frames = frames_tensor
        else:
            frames = [frames_tensor[i] for i in range(frames_tensor.shape[0])]


        # Final aggressive unload of all heavy Wan components right after the pipeline call
        # (including any VAE decode). This gives headroom and follows "unload as soon as gen drops"
        # so high (and low after use) are fully and completely offloaded before returning frames/save.
        # Uses the cache to evict to pinned CPU (no del, no disk reload on next gen).
        try:
            _video_status("Post-pipe: releasing heavy Wan transformers for headroom.")
            if self._cache_mode in ("full", "gpu_swap"):
                self.cache.unload_from_vram("wan_high")
                self.cache.unload_from_vram("wan_low")
            else:
                for tname in ("transformer", "transformer_2"):
                    module = getattr(pipe, tname, None)
                    if module is not None:
                        try:
                            module.to("cpu")
                        except Exception:
                            pass
        except Exception as exc:
            logger.debug("Post-pipe transformer release non-fatal: %s", exc)

        return frames, width, height
