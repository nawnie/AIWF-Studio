from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
from diffusers.quantizers.bitsandbytes import replace_with_bnb_linear
from safetensors import safe_open

from aiwf.infrastructure.quant.bnb_nf4_format import normalize_bnb_4bit_compute_dtype

logger = logging.getLogger(__name__)

_BNB_SIDEcars = ("bitsandbytes__nf4", "bitsandbytes__fp4")


def _read_config(config_dir: str | Path) -> dict:
    config_path = Path(config_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _quant_state_keys(keys: set[str], weight_key: str) -> list[str]:
    return [
        key
        for key in keys
        if key.startswith(f"{weight_key}.")
        and (
            ".quant_state." in key
            or key.endswith(".absmax")
            or key.endswith(".quant_map")
            or key.endswith(".nested_absmax")
            or key.endswith(".nested_quant_map")
            or key.endswith(".nested_offset")
        )
    ]


def _is_bnb_weight(keys: set[str], weight_key: str) -> bool:
    return any(f"{weight_key}.quant_state.{marker}" in keys for marker in _BNB_SIDEcars)


def _dequantize_weight(handle, keys: set[str], weight_key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import bitsandbytes as bnb

    packed = handle.get_tensor(weight_key)
    stats = {key: handle.get_tensor(key) for key in _quant_state_keys(keys, weight_key)}
    quant_state = bnb.functional.QuantState.from_dict(stats, device=device)
    tensor = bnb.functional.dequantize_4bit(packed.to(device), quant_state=quant_state)
    return tensor.to(dtype=dtype)


def _load_tensor(handle, keys: set[str], key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = (
        _dequantize_weight(handle, keys, key, device=device, dtype=dtype)
        if _is_bnb_weight(keys, key)
        else handle.get_tensor(key).to(device)
    )
    if tensor.is_floating_point():
        return tensor.to(dtype=dtype)
    return tensor


def _set_linear4bit_weight(module, value: torch.Tensor, *, device: torch.device, quant_type: str) -> None:
    import bitsandbytes as bnb

    value = value.contiguous()
    param = bnb.nn.Params4bit(
        value,
        requires_grad=False,
        compress_statistics=True,
        quant_type=quant_type,
        quant_storage=torch.uint8,
        module=module,
    ).to(device)
    module._parameters["weight"] = param


def _assign(model, name: str, value: torch.Tensor, *, device: torch.device, dtype: torch.dtype, quant_type: str) -> None:
    import bitsandbytes as bnb

    module_name, tensor_name = name.rsplit(".", 1)
    module = model.get_submodule(module_name)
    if isinstance(module, bnb.nn.Linear4bit) and tensor_name == "weight":
        _set_linear4bit_weight(module, value.to(dtype=dtype), device=device, quant_type=quant_type)
        return
    if isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
        module._parameters["bias"] = torch.nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False)
        return
    if value.is_floating_point():
        value = value.to(dtype=dtype)
    set_module_tensor_to_device(model, name, device, value=value)


def _swap_scale_shift(value: torch.Tensor) -> torch.Tensor:
    shift, scale = value.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _layer_count(keys: set[str], prefix: str) -> int:
    indexes = {int(key.split(".", 2)[1]) for key in keys if key.startswith(prefix)}
    if not indexes:
        return 0
    return max(indexes) + 1


def _clear_tensors(*values: torch.Tensor) -> None:
    for value in values:
        del value
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _missing_meta_parameters(model) -> list[str]:
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            missing.append(name)
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            missing.append(name)
    return missing


def load_flux_original_bnb_transformer(
    path: Path,
    *,
    config_dir: str | Path,
    dtype: torch.dtype,
    device: torch.device,
    quant_type: str = "nf4",
) -> FluxTransformer2DModel:
    if device.type != "cuda":
        raise RuntimeError("Flux BNB original-layout loading requires CUDA.")

    compute_dtype = normalize_bnb_4bit_compute_dtype(dtype)
    config = _read_config(config_dir)
    if isinstance(config.get("axes_dims_rope"), list):
        config["axes_dims_rope"] = tuple(config["axes_dims_rope"])

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    started = time.perf_counter()
    with init_empty_weights():
        model = FluxTransformer2DModel(**config)
    model = replace_with_bnb_linear(model, modules_to_not_convert=[], quantization_config=quant_config)
    model.eval()

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        num_layers = _layer_count(keys, "double_blocks.")
        num_single_layers = _layer_count(keys, "single_blocks.")

        def load(key: str) -> torch.Tensor:
            return _load_tensor(handle, keys, key, device=device, dtype=compute_dtype)

        def assign(target: str, value: torch.Tensor) -> None:
            _assign(model, target, value, device=device, dtype=compute_dtype, quant_type=quant_type)

        linears = (
            ("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer"),
            ("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer"),
            ("time_text_embed.text_embedder.linear_1", "vector_in.in_layer"),
            ("time_text_embed.text_embedder.linear_2", "vector_in.out_layer"),
            ("context_embedder", "txt_in"),
            ("x_embedder", "img_in"),
        )
        if any("guidance_in." in key for key in keys):
            linears = (
                *linears,
                ("time_text_embed.guidance_embedder.linear_1", "guidance_in.in_layer"),
                ("time_text_embed.guidance_embedder.linear_2", "guidance_in.out_layer"),
            )
        for target, source in linears:
            assign(f"{target}.weight", load(f"{source}.weight"))
            assign(f"{target}.bias", load(f"{source}.bias"))

        for i in range(num_layers):
            block = f"transformer_blocks.{i}"
            source = f"double_blocks.{i}"
            assign(f"{block}.norm1.linear.weight", load(f"{source}.img_mod.lin.weight"))
            assign(f"{block}.norm1.linear.bias", load(f"{source}.img_mod.lin.bias"))
            assign(f"{block}.norm1_context.linear.weight", load(f"{source}.txt_mod.lin.weight"))
            assign(f"{block}.norm1_context.linear.bias", load(f"{source}.txt_mod.lin.bias"))

            sample_q, sample_k, sample_v = torch.chunk(load(f"{source}.img_attn.qkv.weight"), 3, dim=0)
            context_q, context_k, context_v = torch.chunk(load(f"{source}.txt_attn.qkv.weight"), 3, dim=0)
            sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(load(f"{source}.img_attn.qkv.bias"), 3, dim=0)
            context_q_bias, context_k_bias, context_v_bias = torch.chunk(load(f"{source}.txt_attn.qkv.bias"), 3, dim=0)
            assign(f"{block}.attn.to_q.weight", sample_q)
            assign(f"{block}.attn.to_k.weight", sample_k)
            assign(f"{block}.attn.to_v.weight", sample_v)
            assign(f"{block}.attn.add_q_proj.weight", context_q)
            assign(f"{block}.attn.add_k_proj.weight", context_k)
            assign(f"{block}.attn.add_v_proj.weight", context_v)
            assign(f"{block}.attn.to_q.bias", sample_q_bias)
            assign(f"{block}.attn.to_k.bias", sample_k_bias)
            assign(f"{block}.attn.to_v.bias", sample_v_bias)
            assign(f"{block}.attn.add_q_proj.bias", context_q_bias)
            assign(f"{block}.attn.add_k_proj.bias", context_k_bias)
            assign(f"{block}.attn.add_v_proj.bias", context_v_bias)
            _clear_tensors(
                sample_q,
                sample_k,
                sample_v,
                context_q,
                context_k,
                context_v,
                sample_q_bias,
                sample_k_bias,
                sample_v_bias,
                context_q_bias,
                context_k_bias,
                context_v_bias,
            )

            assign(f"{block}.attn.norm_q.weight", load(f"{source}.img_attn.norm.query_norm.scale"))
            assign(f"{block}.attn.norm_k.weight", load(f"{source}.img_attn.norm.key_norm.scale"))
            assign(f"{block}.attn.norm_added_q.weight", load(f"{source}.txt_attn.norm.query_norm.scale"))
            assign(f"{block}.attn.norm_added_k.weight", load(f"{source}.txt_attn.norm.key_norm.scale"))

            for target, source_name in (
                ("ff.net.0.proj", "img_mlp.0"),
                ("ff.net.2", "img_mlp.2"),
                ("ff_context.net.0.proj", "txt_mlp.0"),
                ("ff_context.net.2", "txt_mlp.2"),
                ("attn.to_out.0", "img_attn.proj"),
                ("attn.to_add_out", "txt_attn.proj"),
            ):
                assign(f"{block}.{target}.weight", load(f"{source}.{source_name}.weight"))
                assign(f"{block}.{target}.bias", load(f"{source}.{source_name}.bias"))

        inner_dim = int(config["attention_head_dim"]) * int(config["num_attention_heads"])
        mlp_hidden_dim = inner_dim * 4
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        for i in range(num_single_layers):
            block = f"single_transformer_blocks.{i}"
            source = f"single_blocks.{i}"
            assign(f"{block}.norm.linear.weight", load(f"{source}.modulation.lin.weight"))
            assign(f"{block}.norm.linear.bias", load(f"{source}.modulation.lin.bias"))

            q, k, v, mlp = torch.split(load(f"{source}.linear1.weight"), split_size, dim=0)
            q_bias, k_bias, v_bias, mlp_bias = torch.split(load(f"{source}.linear1.bias"), split_size, dim=0)
            assign(f"{block}.attn.to_q.weight", q)
            assign(f"{block}.attn.to_k.weight", k)
            assign(f"{block}.attn.to_v.weight", v)
            assign(f"{block}.proj_mlp.weight", mlp)
            assign(f"{block}.attn.to_q.bias", q_bias)
            assign(f"{block}.attn.to_k.bias", k_bias)
            assign(f"{block}.attn.to_v.bias", v_bias)
            assign(f"{block}.proj_mlp.bias", mlp_bias)
            _clear_tensors(q, k, v, mlp, q_bias, k_bias, v_bias, mlp_bias)

            assign(f"{block}.attn.norm_q.weight", load(f"{source}.norm.query_norm.scale"))
            assign(f"{block}.attn.norm_k.weight", load(f"{source}.norm.key_norm.scale"))
            assign(f"{block}.proj_out.weight", load(f"{source}.linear2.weight"))
            assign(f"{block}.proj_out.bias", load(f"{source}.linear2.bias"))

        assign("proj_out.weight", load("final_layer.linear.weight"))
        assign("proj_out.bias", load("final_layer.linear.bias"))
        assign("norm_out.linear.weight", _swap_scale_shift(load("final_layer.adaLN_modulation.1.weight")))
        assign("norm_out.linear.bias", _swap_scale_shift(load("final_layer.adaLN_modulation.1.bias")))

    missing = _missing_meta_parameters(model)
    if missing:
        raise RuntimeError(f"Flux BNB loader left {len(missing)} tensor(s) on meta: {missing[:8]}")

    logger.info(
        "Loaded packed Flux BNB %s transformer from %s in %.1fs",
        quant_type.upper(),
        path.name,
        time.perf_counter() - started,
    )
    model._aiwf_bnb_original_layout = True
    model._aiwf_force_diffusers_attention_backend = "native"
    return model
