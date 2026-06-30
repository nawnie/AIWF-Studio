"""AIWF-native dual-stage (high/low) Wan I2V denoise loop.

This module reimplements -- rather than calls -- the diffusers
``WanImageToVideoPipeline.__call__`` denoising orchestration: explicit
prompt/image encoding, scheduler timestep setup, per-step high/low
transformer boundary switching with classifier-free guidance, scheduler
stepping, and VAE decode/postprocess. It exists so AIWF owns the actual
generation loop instead of treating the diffusers pipeline call as a
black box. The detailed standalone-runtime notes are local-only project
planning material.

Important: the high/low stage swap itself (background preload, VRAM/CPU
cache eviction, disk-sequential staging) is NOT reimplemented here -- that
machinery already lives behind ``pipe.transformer_2``'s lazy-loading proxy
(``_LazyWanTransformer`` in aiwf/infrastructure/wan/pipeline.py) and fires
automatically the first time this loop touches ``transformer_2.cache_context``
or calls it directly, exactly as it does inside diffusers' own loop. This
loop only needs to reach for ``pipe.transformer`` / ``pipe.transformer_2``
at the correct boundary timestep -- normal attribute access does the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import contextlib
import time


@dataclass
class NativeWanDenoiseOutput:
    """Duck-typed replacement for diffusers' ``WanPipelineOutput``.

    Downstream code in ``pipeline.py`` only ever does
    ``output.frames if hasattr(output, "frames") else output``, so this
    minimal shape is sufficient and keeps this module free of a diffusers
    import dependency.
    """

    frames: Any


class NativeWanDenoiseCancelled(RuntimeError):
    pass


def _trace_native_denoise(event: str, message: str, **fields: Any) -> None:
    """Best-effort diagnostics for a real native denoise run.

    Always feeds the in-process diagnostics ring (``trace_safe``) and, when
    ``AIWF_WAN_DENOISE_DIAG`` is truthy, also logs at INFO so a real
    720x720/81-frame run can be traced from the console.
    """
    import logging

    log = logging.getLogger(__name__)
    try:
        from aiwf.dev.diagnostics import trace_safe

        trace_safe(event, message, **fields)
    except Exception:
        log.debug("native denoise trace failed", exc_info=True)
    try:
        import os

        if os.environ.get("AIWF_WAN_DENOISE_DIAG", "").strip().lower() in {"1", "true", "yes", "on"}:
            log.info("%s: %s", message, fields)
    except Exception:
        pass


def _denoise_diag_enabled() -> bool:
    try:
        import os

        return os.environ.get("AIWF_WAN_DENOISE_DIAG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def _strict_attention_enabled() -> bool:
    try:
        import os

        return os.environ.get("AIWF_WAN_STRICT_ATTENTION", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def _strict_sdpa_context():
    if not _strict_attention_enabled():
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        raise RuntimeError("AIWF_WAN_STRICT_ATTENTION=1 but torch.nn.attention is unavailable.") from exc

    backends = []
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"):
        backend = getattr(SDPBackend, name, None)
        if backend is not None:
            backends.append(backend)
    if not backends:
        raise RuntimeError("AIWF_WAN_STRICT_ATTENTION=1 but no fused SDPA backends are available.")
    return sdpa_kernel(backends)


def _cuda_memory_fields() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_gb": round(float(torch.cuda.memory_allocated()) / 1024**3, 3),
            "cuda_reserved_gb": round(float(torch.cuda.memory_reserved()) / 1024**3, 3),
            "cuda_peak_allocated_gb": round(float(torch.cuda.max_memory_allocated()) / 1024**3, 3),
        }
    except Exception:
        return {}


def _sync_cuda_for_diag() -> None:
    if not _denoise_diag_enabled():
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _module_has_offload_hook(module: Any) -> bool:
    if module is None:
        return False
    if getattr(module, "_hf_hook", None) is not None:
        return True
    if getattr(module, "_diffusers_hook", None) is not None:
        return True
    if getattr(module, "_aiwf_group_offload", False):
        return True
    try:
        for child in module.modules():
            if (
                getattr(child, "_hf_hook", None) is not None
                or getattr(child, "_diffusers_hook", None) is not None
                or getattr(child, "_aiwf_group_offload", False)
            ):
                return True
    except Exception:
        pass
    return False


def _first_module_tensor_device(module: Any) -> Any:
    if module is None:
        return None
    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(module, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            for tensor in iterator():
                device = getattr(tensor, "device", None)
                if device is not None:
                    return device
        except Exception:
            continue
    return None


@contextlib.contextmanager
def _component_on_device(module: Any, device: Any, *, label: str):
    """Temporarily place plain CPU components on the execution device."""
    if module is None or _module_has_offload_hook(module):
        yield
        return

    try:
        import torch
    except Exception:
        yield
        return

    target = torch.device(device)
    original = _first_module_tensor_device(module)
    moved = original is not None and original != target
    if moved:
        started = time.perf_counter()
        module.to(target)
        _sync_cuda_for_diag()
        _trace_native_denoise(
            "wan.native_component_onload",
            "Moved Wan component to execution device",
            component=label,
            device=str(target),
            seconds=round(max(0.0, time.perf_counter() - started), 3),
            **_cuda_memory_fields(),
        )
    try:
        yield
    finally:
        if moved:
            started = time.perf_counter()
            try:
                module.to(original)
                if target.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                _trace_native_denoise(
                    "wan.native_component_offload",
                    "Returned Wan component to original device",
                    component=label,
                    device=str(original),
                    seconds=round(max(0.0, time.perf_counter() - started), 3),
                    **_cuda_memory_fields(),
                )


def _cache_key_for_stage(stage: str) -> str | None:
    if stage == "high":
        return "wan_high"
    if stage == "low":
        return "wan_low"
    return None


def _evict_active_stage_before_prompt(pipe: Any) -> None:
    """Free cached Wan expert VRAM before UMT5 prompt encoding.

    Balanced/model placement keeps only one high/low expert in VRAM at a time.
    If a previous run or an early load left an expert resident, UMT5 can OOM
    while Accelerate moves the text encoder to CUDA. Evicting here gives prompt
    encoding first claim on VRAM; the selected expert is reloaded before the
    first denoise forward.
    """
    cache = getattr(pipe, "_aiwf_stage_cache", None)
    active = str(getattr(cache, "active_in_vram", "") or "")
    if not cache or active not in {"wan_high", "wan_low"}:
        return
    try:
        cache.unload_from_vram(active)
        _trace_native_denoise(
            "wan.native_prompt_headroom",
            "Evicted cached Wan stage before prompt encoding",
            stage=active,
            **_cuda_memory_fields(),
        )
    except Exception:
        pass


def _ensure_cached_stage_on_device(pipe: Any, stage: str) -> None:
    cache = getattr(pipe, "_aiwf_stage_cache", None)
    key = _cache_key_for_stage(stage)
    if not cache or key is None:
        return
    try:
        if key in getattr(cache, "cpu_cache", {}) and getattr(cache, "active_in_vram", None) != key:
            cache.load_to_vram(key)
            _trace_native_denoise(
                "wan.native_stage_cache_load",
                "Loaded cached Wan stage before denoise forward",
                stage=key,
                **_cuda_memory_fields(),
            )
    except Exception as exc:
        raise RuntimeError(f"Could not load cached Wan stage {key} to the denoise device.") from exc


def _fp8_metrics_for(model: Any) -> dict[str, Any]:
    if not _denoise_diag_enabled():
        return {}
    try:
        from aiwf.infrastructure.quant.fp8_linear import collect_fp8_linear_metrics

        loaded = getattr(model, "_loaded_model", None)
        if loaded is not None:
            return collect_fp8_linear_metrics(model, loaded)
        return collect_fp8_linear_metrics(model)
    except Exception:
        return {}


def _metric_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> int:
    try:
        return int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
    except Exception:
        return 0


def _ensure_stage_ready(model: Any, *, device: Any, stage: str, step: int, pipe: Any = None) -> tuple[Any, float]:
    """Materialize lazy high/low stage proxies on the target device before cache_context.

    ``_LazyWanTransformer.cache_context()`` cannot see ``hidden_states.device``.
    If the real low model is first moved to CUDA inside ``forward()``, it enters
    the real transformer's cache context while still CPU-backed, which makes the
    stage boundary hard to reason about and can leave cleanup/device placement
    outside our timings. Force the lazy stage onto the denoise device first.
    """
    started = time.perf_counter()
    if pipe is not None:
        _ensure_cached_stage_on_device(pipe, stage)
    loaded = model
    ensure_loaded = getattr(model, "_ensure_loaded", None)
    if callable(ensure_loaded):
        try:
            ensured = ensure_loaded(device=device)
        except TypeError:
            ensured = ensure_loaded()
        if ensured is not None:
            loaded = ensured
        if loaded is model and getattr(model, "loaded", True) is not False:
            try:
                if hasattr(loaded, "to"):
                    loaded.to(device)
            except Exception:
                pass
    elapsed = max(0.0, time.perf_counter() - started)
    if elapsed > 0.25 or _denoise_diag_enabled():
        _sync_cuda_for_diag()
        _trace_native_denoise(
            "wan.native_stage_ready",
            "Wan native denoise stage ready",
            stage=stage,
            step=int(step),
            stage_ready_seconds=round(float(elapsed), 3),
            device=str(device),
            **_cuda_memory_fields(),
        )
    return loaded, elapsed


def run_native_wan_denoise(pipe: Any, **kwargs: Any) -> "NativeWanDenoiseOutput":
    """Public entry point for AIWF's native Wan denoise loop.

    Wraps the implementation in ``torch.inference_mode()`` to mirror
    diffusers' inference-only call contract while avoiding autograd view/version
    overhead. AIWF deliberately bypasses ``pipe.__call__`` to own the loop, but
    doing so also bypasses diffusers' decorator. Without this guard the loop
    builds a full autograd graph: ``prepare_latents`` runs ``vae.encode`` to
    produce the conditioning tensor, that tensor feeds the transformer every
    step, and activations are retained for a backward pass that never happens.
    Also records peak VRAM for the run when CUDA is available.
    """
    import torch

    reset_peak = False
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            reset_peak = True
    except Exception:
        reset_peak = False

    with torch.inference_mode():
        output = _run_native_wan_denoise_impl(pipe, **kwargs)

    if reset_peak:
        try:
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            _trace_native_denoise(
                "wan.native_denoise_peak",
                "Wan native denoise peak VRAM",
                peak_vram_gb=round(float(peak_gb), 3),
            )
        except Exception:
            pass
    return output


def _run_native_wan_denoise_impl(
    pipe: Any,
    *,
    image: Any,
    prompt: str | Sequence[str] | None = None,
    negative_prompt: str | Sequence[str] | None = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_2: float | None = None,
    num_videos_per_prompt: int = 1,
    generator: Any = None,
    latents: Any = None,
    prompt_embeds: Any = None,
    negative_prompt_embeds: Any = None,
    image_embeds: Any = None,
    last_image: Any = None,
    output_type: str = "np",
    attention_kwargs: dict[str, Any] | None = None,
    callback_on_step_end: Callable[..., Any] | None = None,
    callback_on_step_end_tensor_inputs: Sequence[str] = ("latents",),
    aiwf_on_phase_progress: Callable[[str], Any] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    max_sequence_length: int = 512,
    **_extra: Any,
) -> NativeWanDenoiseOutput:
    """Run AIWF's own dual-stage (high/low) Wan I2V denoise loop.

    Mirrors ``WanImageToVideoPipeline.__call__``'s orchestration exactly
    (same tensor shapes/dtypes, same boundary-ratio switch, same CFG math)
    but performs every step explicitly instead of delegating to diffusers.
    Accepts a superset of the keyword arguments AIWF already builds for the
    diffusers call (``call_kwargs`` in ``WanI2VBackend.generate``), so it is
    a drop-in replacement at that call site. Unrecognized kwargs (e.g. a
    future ``image_guidance_scale`` the installed diffusers build doesn't
    support either) are accepted and ignored via ``**_extra``.
    """
    import torch

    transformer = getattr(pipe, "transformer", None)
    transformer_2 = getattr(pipe, "transformer_2", None)
    if transformer is None and transformer_2 is None:
        raise RuntimeError("Wan pipeline has neither transformer nor transformer_2 loaded.")

    def _emit_phase(message: str) -> None:
        if aiwf_on_phase_progress is None:
            return
        try:
            aiwf_on_phase_progress(message)
        except Exception:
            pass

    def _cancel_requested() -> bool:
        if getattr(pipe, "interrupt", False):
            return True
        if should_cancel is None:
            return False
        try:
            if should_cancel():
                pipe._interrupt = True
                return True
        except Exception:
            _trace_native_denoise(
                "wan.native_cancel_check_failed",
                "Wan native cancel check failed",
            )
        return bool(getattr(pipe, "interrupt", False))

    def _raise_if_cancelled(phase: str) -> None:
        if not _cancel_requested():
            return
        _trace_native_denoise(
            "wan.native_denoise_cancelled",
            "Wan native denoise cancelled",
            phase=phase,
        )
        try:
            pipe._current_timestep = None
        except Exception:
            pass
        raise NativeWanDenoiseCancelled("Wan native denoise cancelled by user.")

    device = getattr(pipe, "_aiwf_execution_device", None) or pipe._execution_device

    # --- frame-count / spatial alignment (mirrors diffusers __call__) ---
    vae_scale_factor_temporal = getattr(pipe, "vae_scale_factor_temporal", 4)
    if num_frames % vae_scale_factor_temporal != 1:
        num_frames = num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    patch_size = (
        transformer.config.patch_size if transformer is not None else transformer_2.config.patch_size
    )
    vae_scale_factor_spatial = getattr(pipe, "vae_scale_factor_spatial", 8)
    h_multiple_of = vae_scale_factor_spatial * patch_size[1]
    w_multiple_of = vae_scale_factor_spatial * patch_size[2]
    height = height // h_multiple_of * h_multiple_of
    width = width // w_multiple_of * w_multiple_of

    if getattr(pipe.config, "boundary_ratio", None) is not None and guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale

    pipe._guidance_scale = guidance_scale
    pipe._guidance_scale_2 = guidance_scale_2
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False
    _raise_if_cancelled("start")

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, (list, tuple)):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    _evict_active_stage_before_prompt(pipe)
    _raise_if_cancelled("before_prompt_encode")
    with _component_on_device(getattr(pipe, "text_encoder", None), device, label="text_encoder"):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
    _raise_if_cancelled("after_prompt_encode")

    transformer_dtype = transformer.dtype if transformer is not None else transformer_2.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # Only the Wan 2.1 I2V transformer accepts image_embeds; Wan 2.2's
    # dual-stage 14B transformers have image_dim=None and skip this.
    if transformer is not None and transformer.config.image_dim is not None:
        if image_embeds is None:
            _raise_if_cancelled("before_image_encode")
            with _component_on_device(getattr(pipe, "image_encoder", None), device, label="image_encoder"):
                if last_image is None:
                    image_embeds = pipe.encode_image(image, device)
                else:
                    image_embeds = pipe.encode_image([image, last_image], device)
            _raise_if_cancelled("after_image_encode")
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    # Parity with diffusers __call__ (sets self._num_timesteps); some
    # callback_on_step_end implementations read pipe.num_timesteps.
    pipe._num_timesteps = len(timesteps)

    num_channels_latents = pipe.vae.config.z_dim
    image_tensor = pipe.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=torch.float32
    )
    last_image_tensor = None
    if last_image is not None:
        last_image_tensor = pipe.video_processor.preprocess(last_image, height=height, width=width).to(
            device, dtype=torch.float32
        )

    _raise_if_cancelled("before_latents")
    with _component_on_device(getattr(pipe, "vae", None), device, label="vae_encode"):
        latents_outputs = pipe.prepare_latents(
            image_tensor,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image_tensor,
        )
    _raise_if_cancelled("after_latents")
    expand_timesteps = bool(getattr(pipe.config, "expand_timesteps", False))
    first_frame_mask = None
    if expand_timesteps:
        latents, condition, first_frame_mask = latents_outputs
    else:
        latents, condition = latents_outputs

    boundary_ratio = getattr(pipe.config, "boundary_ratio", None)
    boundary_timestep = (
        boundary_ratio * pipe.scheduler.config.num_train_timesteps if boundary_ratio is not None else None
    )

    latent_model_input_cache = None
    latent_channels = int(latents.shape[1])
    use_preallocated_model_input = (
        not expand_timesteps
        and getattr(latents, "ndim", 0) == 5
        and getattr(condition, "ndim", 0) == 5
    )
    if use_preallocated_model_input:
        latent_model_input_cache = torch.empty(
            (
                latents.shape[0],
                latents.shape[1] + condition.shape[1],
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=transformer_dtype,
        )
        latent_model_input_cache[:, latent_channels:].copy_(condition)

    # --- pre-loop diagnostics: resolved geometry, dtypes and stage split ---
    # This is the single most useful thing to log before a real run: it
    # surfaces wrong latent/condition shapes, dtype mismatches, an off
    # boundary (wrong high/low split), and whether CFG is active -- all
    # before a single (expensive) transformer forward fires.
    try:
        if boundary_timestep is None:
            _high_steps_diag = len(timesteps)
        else:
            _high_steps_diag = int(sum(1 for _t in timesteps if float(_t) >= boundary_timestep))
        _trace_native_denoise(
            "wan.native_denoise_setup",
            "Wan native denoise setup",
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            num_inference_steps=int(len(timesteps)),
            latent_shape=[int(v) for v in getattr(latents, "shape", ())],
            condition_shape=[int(v) for v in getattr(condition, "shape", ())],
            transformer_dtype=str(transformer_dtype).replace("torch.", ""),
            boundary_timestep=None if boundary_timestep is None else round(float(boundary_timestep), 3),
            high_steps=_high_steps_diag,
            low_steps=int(len(timesteps) - _high_steps_diag),
            expand_timesteps=bool(expand_timesteps),
            classifier_free_guidance=bool(getattr(pipe, "do_classifier_free_guidance", False)),
            device=str(device),
            grad_enabled=bool(torch.is_grad_enabled()),
        )
    except Exception:
        pass

    _emit_phase("Denoising video; first GGUF step can take several minutes")
    _raise_if_cancelled("before_denoise")

    for i, t in enumerate(timesteps):
        step_started = time.perf_counter()
        _raise_if_cancelled("step_start")

        pipe._current_timestep = t

        if boundary_timestep is None or t >= boundary_timestep:
            # Wan 2.1, or the high-noise stage of Wan 2.2's dual-stage pair.
            current_model = transformer
            current_guidance_scale = guidance_scale
            stage_name = "high"
        else:
            # Low-noise stage of Wan 2.2. Accessing transformer_2 here is
            # what triggers the lazy stage-swap proxy (background preload
            # completion, high-stage release, low-stage load) -- nothing
            # else in this loop needs to know that swap is happening.
            current_model = transformer_2
            current_guidance_scale = guidance_scale_2
            stage_name = "low"

        if expand_timesteps:
            latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
            latent_model_input = latent_model_input.to(transformer_dtype)
            temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            if use_preallocated_model_input:
                latent_model_input_cache[:, :latent_channels].copy_(latents)
                latent_model_input = latent_model_input_cache
            else:
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

        current_model, stage_ready_seconds = _ensure_stage_ready(
            current_model,
            device=latent_model_input.device,
            stage=stage_name,
            step=i + 1,
            pipe=pipe,
        )
        _raise_if_cancelled(f"{stage_name}_stage_ready")
        fp8_before = _fp8_metrics_for(current_model)

        _raise_if_cancelled(f"{stage_name}_before_cond_forward")
        with current_model.cache_context("cond"), _strict_sdpa_context():
            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
        _raise_if_cancelled(f"{stage_name}_after_cond_forward")

        if pipe.do_classifier_free_guidance:
            _raise_if_cancelled(f"{stage_name}_before_uncond_forward")
            with current_model.cache_context("uncond"), _strict_sdpa_context():
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)
            _raise_if_cancelled(f"{stage_name}_after_uncond_forward")

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        _raise_if_cancelled(f"{stage_name}_after_scheduler_step")
        _sync_cuda_for_diag()
        fp8_after = _fp8_metrics_for(current_model)
        if _denoise_diag_enabled():
            _trace_native_denoise(
                "wan.native_denoise_step_detail",
                "Wan native denoise step detail",
                stage=stage_name,
                step=int(i + 1),
                total_steps=int(len(timesteps)),
                timestep=round(float(t), 3),
                stage_ready_seconds=round(float(stage_ready_seconds), 3),
                step_seconds=round(float(max(0.0, time.perf_counter() - step_started)), 3),
                fp8_linear_layers=int(fp8_after.get("fp8_linear_layers", 0) or 0),
                fp8_fast_mm_calls_delta=_metric_delta(fp8_after, fp8_before, "fp8_fast_mm_calls"),
                fp8_fallback_calls_delta=_metric_delta(fp8_after, fp8_before, "fp8_fallback_calls"),
                fp8_fallback_layers=int(fp8_after.get("fp8_fallback_layers", 0) or 0),
                **_cuda_memory_fields(),
            )

        if callback_on_step_end is not None:
            local_vars = {
                "latents": latents,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            }
            callback_kwargs = {
                k: local_vars[k] for k in callback_on_step_end_tensor_inputs if k in local_vars
            }
            callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs) or {}
            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
            _raise_if_cancelled("after_step_callback")

    pipe._current_timestep = None
    _raise_if_cancelled("after_denoise")

    if expand_timesteps:
        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

    if output_type != "latent":
        _raise_if_cancelled("before_decode")
        _emit_phase("Denoise complete; decoding video frames")
        _emit_phase("Decoding video frames")
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        with _component_on_device(getattr(pipe, "vae", None), latents.device, label="vae_decode"):
            video = pipe.vae.decode(latents, return_dict=False)[0]
            _emit_phase("Post-processing video frames")
            video = pipe.video_processor.postprocess_video(video, output_type=output_type)
        _raise_if_cancelled("after_decode")
    else:
        _emit_phase("Denoise complete; returning latents")
        video = latents

    pipe.maybe_free_model_hooks()

    return NativeWanDenoiseOutput(frames=video)
