"""Wan 2.2 image-to-video backend (diffusers WanImageToVideoPipeline).

Heavy imports (torch/diffusers) are lazy so the app loads fine without the Wan
stack installed or the model downloaded. Tuned for consumer GPUs via CPU
offloading + VAE tiling — slow on 8 GB, but it runs.

Supports:
- Full diffusers layouts (model_index.json + subfolders) via from_pretrained
- Standalone .safetensors or .gguf files for the transformer weights (ComfyUI
  diffusion_models style or GGUF quants). These are loaded by reusing a
  co-located or default full diffusers folder (e.g. Wan2.2-TI2V-5B-Diffusers)
  for the text encoder / tokenizer / VAE / scheduler, then injecting the
  custom transformer weights (with key rename to support both "diffusers-key"
  and "original/Comfy-key" files).
"""
from __future__ import annotations

import logging
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

            if (
                input.is_cuda
                and self.weight.is_cuda
                and hasattr(torch, "_scaled_mm")
                and input.shape[-1] % 16 == 0
                and self.weight.shape[0] % 16 == 0
            ):
                original_shape = input.shape[:-1]
                x = input.reshape(-1, input.shape[-1])
                scale_a = torch.ones((), device=x.device, dtype=torch.float32)
                x8 = x.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
                y = torch._scaled_mm(
                    x8,
                    self.weight.t(),
                    scale_a=scale_a,
                    scale_b=self.weight_scale.to(device=x.device, dtype=torch.float32),
                    out_dtype=input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
                )
                if self.bias is not None:
                    y = y + self.bias.to(device=y.device, dtype=y.dtype)
                return y.reshape(*original_shape, self.out_features)

            weight = self.weight.float() * self.weight_scale.float()
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


def _load_gguf_state_dict(path: Path) -> dict:
    """Load a GGUF file (any quantization) into a torch state_dict (fp32 tensors). Requires `gguf` package."""
    import gguf
    import numpy as np
    import torch

    reader = gguf.GGUFReader(str(path))
    sd: dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        name = tensor.name
        arr = gguf.dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(np.asarray(arr))
        # Keep as float32 here; dtype cast happens on the model later.
        if t.dtype == torch.float16 or t.dtype == torch.bfloat16:
            t = t.float()
        sd[name] = t
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


def _find_wan_components_base(search_roots: list[Path] | None = None) -> str | None:
    """Find a full diffusers Wan layout we can use for text_encoder / vae / tokenizer / scheduler when loading single-file weights."""
    roots = search_roots or [Path("models/wan/Diffusers"), Path("models/wan")]

    for root in roots:
        preferred = root / "Wan2.2-TI2V-5B-Diffusers"
        if _is_wan_components_base(preferred):
            return str(preferred.resolve())

    # Backward-compatible fallbacks for older local layouts.
    preferred = [
        Path("models/wan/Diffusers/Wan2.2-TI2V-5B-Diffusers"),
        Path("models/wan/Wan2.2-TI2V-5B-Diffusers"),
        Path(r"C:\Users\Shawn\Desktop\AIWF-Studio\models\wan\Diffusers\Wan2.2-TI2V-5B-Diffusers"),
        Path(r"C:\Users\Shawn\Desktop\AIWF-Studio\models\wan\Wan2.2-TI2V-5B-Diffusers"),
    ]
    for p in preferred:
        rp = p.resolve() if not p.is_absolute() else p
        if _is_wan_components_base(rp):
            return str(rp)

    for wan_root in roots:
        if wan_root.exists():
            candidates = [wan_root, *[child for child in wan_root.rglob("*") if child.is_dir()]]
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


def _empty_wan_transformer(config):
    """Create a Wan transformer shell without random parameter allocation."""
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    with init_empty_weights():
        return WanTransformer3DModel.from_config(config)


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


class WanI2VBackend:
    """Loads a Wan image-to-video pipeline once and reuses it across renders."""

    def __init__(self) -> None:
        self._pipe = None
        self._key = None

    def available(self) -> bool:
        return wan_supported()

    def unload(self) -> None:
        self._pipe = None
        self._key = None

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

        _require_wan()
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        logger.info(
            "Loading Wan I2V pipeline (dual high/low): high=%s low=%s boundary=%s vae=%s (offload=%s)",
            high_noise_model_id, low_noise_model_id, boundary_ratio, vae_id, offload,
        )
        _video_status(
            f"Loading video pipeline with local base components and dual transformers ({Path(high_noise_model_id).name} / {Path(low_noise_model_id).name})."
        )
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
            flow_shift=flow_shift,
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=float(flow_shift))

        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            fast_fp8_pair = (
                _torch_native_fp8_available()
                and _is_native_comfy_fp8_transformer(high_noise_model_id)
                and _is_native_comfy_fp8_transformer(low_noise_model_id)
            )
            if fast_fp8_pair:
                _video_status(
                    "Using fast FP8 placement: keeping the active Wan transformer on GPU while offloading text encoder/VAE."
                )
                original_seq = pipe.model_cpu_offload_seq
                original_exclude = list(getattr(pipe, "_exclude_from_cpu_offload", []) or [])
                pipe.model_cpu_offload_seq = "text_encoder->image_encoder->vae"
                pipe._exclude_from_cpu_offload = sorted(
                    set(original_exclude).union({"transformer", "transformer_2"})
                )
                try:
                    pipe.enable_model_cpu_offload()
                finally:
                    pipe.model_cpu_offload_seq = original_seq
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        for method in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, method)()
            except Exception:
                pass

        self._pipe = pipe
        self._key = key
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
                "Dual high/low Wan models require a full diffusers base layout (text_encoder/vae etc.) "
                "in models/wan/. The existing Wan2.2-TI2V-5B-Diffusers can be used as the component provider."
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

        # Helper to get raw (renamed) state for a high or low non-FP8 file
        def _get_sd(pth: str) -> dict:
            pp = Path(pth)
            _video_status(f"Loading transformer weights: {pp.name}")
            if pp.suffix.lower() == ".gguf":
                try:
                    raw = _load_gguf_state_dict(pp)
                except Exception as exc:
                    raise WanUnavailable(f"GGUF load failed for {pp.name}. Install `gguf`.") from exc
            else:
                from safetensors.torch import load_file
                raw = load_file(str(pp))
                if _safetensors_uses_comfy_fp8_quant(pp):
                    raise WanUnavailable(
                        f"Internal loader error: {pp.name} is Comfy FP8 and should use the native FP8 path."
                    )
            return _apply_wan_transformer_key_renames(raw)

        # Inject high into the primary transformer (overwrites the base one in-place)
        high_pp = Path(high_path)
        if high_pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(high_pp):
            _video_status(f"Loading native Comfy FP8 high-noise transformer: {high_pp.name}")
            miss_h, unex_h = _load_comfy_fp8_transformer_weights(
                pipe.transformer, high_pp, torch_dtype=torch.bfloat16
            )
        else:
            high_sd = _get_sd(high_path)
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

        # Low-noise is loaded lazily at the high->low boundary so machines that
        # cannot hold both 14B stages at once can still run the staged schedule.
        low_pp = Path(low_path)

        def _release_high_stage():
            import gc
            import torch

            if pipe.transformer is not None:
                _video_status("High-noise stage complete; releasing high-noise transformer before loading low-noise.")
                pipe.transformer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def _load_low_stage():
            _video_status("Preparing empty low-noise transformer stage.")
            low_trans = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
            if low_pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(low_pp):
                _video_status(f"Loading native Comfy FP8 low-noise transformer: {low_pp.name}")
                miss_l, unex_l = _load_comfy_fp8_transformer_weights(low_trans, low_pp, torch_dtype=torch.bfloat16)
            else:
                low_sd = _get_sd(low_path)
                _video_status(f"Applying low-noise transformer: {Path(low_path).name}")
                miss_l, unex_l = low_trans.load_state_dict(low_sd, strict=False, assign=True)
                del low_sd
            if miss_l or unex_l:
                logger.warning(
                    "Low-noise weights (%s) vs base: %d missing %d unexpected",
                    Path(low_path).name,
                    len(miss_l),
                    len(unex_l),
                )
            if low_lora_path:
                _video_status(f"Applying low-noise LoRA: {Path(low_lora_path).name}")
            _apply_transformer_lora(
                low_trans,
                low_lora_path,
                adapter_name="wan_low_lora",
                weight=low_lora_scale,
            )
            return low_trans

        _video_status("Low-noise transformer will load after the high-noise stage releases.")
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
        """Run image->video. Returns a list of PIL frames."""
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

        def _callback(pipe_obj, step_index, _timestep, kwargs):
            if should_cancel and should_cancel():
                setattr(pipe_obj, "_interrupt", True)
            if on_progress:
                on_progress(step_index + 1, steps)
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
        )
        _video_status(
            f"Running video diffusion: {steps} steps ({high_steps} high / {steps - high_steps} low), "
            f"{request.normalized_frames()} frames."
        )
        try:
            output = pipe(**call_kwargs)
        except Exception:
            logger.exception("Wan pipeline execution failed")
            raise
        _video_status("Video diffusion complete; preparing output.")
        return output.frames[0], width, height
