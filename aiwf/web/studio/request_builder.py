from __future__ import annotations

import gradio as gr

from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.models import normalize_schedule_id_for_sampler
from aiwf.core.tags import parse_tags
from aiwf.infrastructure.diffusers.mask import inpaint_session_background, resolve_inpaint_mask
from aiwf.services.controlnet import ControlNetService
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.controlnet_stack import StudioControlNetSlot, build_controlnet_stack
from aiwf.web.studio.helpers import generation_style_fields, mode_from_label
from aiwf.web.studio.session import StudioSession


def build_generation_request(
    *,
    catalogs: StudioCatalogs,
    session: StudioSession,
    mode_label: str,
    editing_mask: bool,
    prompt_text: str,
    negative_text: str,
    ckpt_title: str,
    sampler_label: str,
    scheduler_label: str,
    step_count,
    cfg_scale,
    clip_skip_value,
    w,
    h,
    bs,
    bc,
    seed_value,
    vae_id,
    hires_enabled,
    hires_scale,
    hires_steps,
    hires_denoise,
    img2img_denoise,
    inpaint_denoise_value,
    mask_blur_value,
    inpaint_area_value,
    inpaint_padding_value,
    masked_content_value,
    source_image,
    editor_value,
    ckpt_map: dict,
    tags_text: str,
    use_file: bool,
    prompt_file_path: str | None,
    dynamic_seed,
    style_name: str | None,
    style_template_prompt: str | None,
    style_template_negative: str | None,
    cn_enable: bool,
    cn_model_id: str | None,
    cn_module: str | None,
    cn_image,
    cn_weight,
    cn_guidance_start,
    cn_guidance_end,
    cn_threshold_a,
    cn_threshold_b,
    inpaint_source_choice: str,
    cn2_enable: bool = False,
    cn2_model_id: str | None = None,
    cn2_module: str | None = None,
    cn2_image=None,
    cn2_weight=1.0,
    cn2_guidance_start=0.0,
    cn2_guidance_end=1.0,
    cn2_threshold_a=100.0,
    cn2_threshold_b=200.0,
    cn3_enable: bool = False,
    cn3_model_id: str | None = None,
    cn3_module: str | None = None,
    cn3_image=None,
    cn3_weight=1.0,
    cn3_guidance_start=0.0,
    cn3_guidance_end=1.0,
    cn3_threshold_a=100.0,
    cn3_threshold_b=200.0,
    controlnet: ControlNetService | None = None,
    checkpoint_architecture: str | None = None,
):
    if not ckpt_map or not ckpt_title:
        raise gr.Error("No checkpoint available. Refresh models.")

    if use_file and not prompt_file_path and not (prompt_text or "").strip():
        raise gr.Error("Select a prompt file or enter a prompt.")

    mode = mode_from_label(mode_label)
    ckpt_id = ckpt_map.get(ckpt_title)
    tags = parse_tags(tags_text or "")
    style_fields = generation_style_fields(style_name, style_template_prompt, style_template_negative)
    sampler_id = catalogs.sampler_map.get(sampler_label, "euler_a")
    scheduler_id = normalize_schedule_id_for_sampler(
        sampler_id,
        catalogs.schedule_map.get(scheduler_label, "automatic"),
    )
    before_image = None
    init_images = None
    mask_images = None

    if mode == "txt2img":
        request = GenerationRequest(
            mode=GenerationMode.TXT2IMG,
            prompt=prompt_text,
            negative_prompt=negative_text,
            prompt_file=prompt_file_path,
            use_prompt_file=bool(use_file),
            prompt_seed=dynamic_seed,
            **style_fields,
            tags=tags,
            steps=int(step_count),
            cfg_scale=float(cfg_scale),
            width=int(w),
            height=int(h),
            seed=int(seed_value),
            sampler=sampler_id,
            scheduler=scheduler_id,
            batch_size=int(bs),
            batch_count=int(bc),
            clip_skip=int(clip_skip_value),
            enable_hr=bool(hires_enabled),
            hr_scale=float(hires_scale),
            hr_steps=int(hires_steps),
            hr_denoising_strength=float(hires_denoise),
            checkpoint_id=ckpt_id,
            vae_id=vae_id,
        )
    elif mode == "img2img":
        if source_image is None:
            raise gr.Error("Upload an image first.")
        before_image = source_image.copy()
        init_images = [source_image]
        request = GenerationRequest(
            mode=GenerationMode.IMG2IMG,
            prompt=prompt_text,
            negative_prompt=negative_text,
            prompt_file=prompt_file_path,
            use_prompt_file=bool(use_file),
            prompt_seed=dynamic_seed,
            **style_fields,
            tags=tags,
            steps=int(step_count),
            cfg_scale=float(cfg_scale),
            seed=int(seed_value),
            sampler=sampler_id,
            scheduler=scheduler_id,
            denoising_strength=float(img2img_denoise),
            clip_skip=int(clip_skip_value),
            checkpoint_id=ckpt_id,
        )
    else:
        background = inpaint_session_background(
            inpaint_source_choice,
            source_image,
            editor_value,
            session.inpaint_session,
        )
        if background is None:
            raise gr.Error("Upload an image and paint a mask.")

        mask = resolve_inpaint_mask(
            editor_value,
            session.inpaint_session,
            session.sam_mask,
            background.size,
            editing_mask=bool(editing_mask),
        )
        if mask is None or mask.getbbox() is None:
            raise gr.Error(
                "No mask found. Paint over the area, use Segment, or click **Paint mask** to restore the last mask."
            )

        session.inpaint.mask = mask.copy()
        if session.inpaint.original is None:
            session.inpaint.original = background.copy()

        before_image = background.copy()
        init_images = [background]
        mask_images = [mask]
        request = GenerationRequest(
            mode=GenerationMode.INPAINT,
            prompt=prompt_text,
            negative_prompt=negative_text,
            prompt_file=prompt_file_path,
            use_prompt_file=bool(use_file),
            prompt_seed=dynamic_seed,
            **style_fields,
            tags=tags,
            steps=int(step_count),
            cfg_scale=float(cfg_scale),
            seed=int(seed_value),
            sampler=sampler_id,
            scheduler=scheduler_id,
            denoising_strength=float(inpaint_denoise_value),
            mask_blur=int(mask_blur_value),
            inpaint_only_masked=(inpaint_area_value == "Only masked"),
            inpaint_masked_padding=int(inpaint_padding_value),
            inpaint_mask_content=str(masked_content_value or "original"),
            clip_skip=int(clip_skip_value),
            checkpoint_id=ckpt_id,
        )

    control_images = None
    try:
        units, control_images_list = build_controlnet_stack(
            slots=[
                StudioControlNetSlot(
                    "ControlNet unit 1",
                    bool(cn_enable),
                    cn_model_id,
                    cn_module,
                    cn_image,
                    float(cn_weight),
                    float(cn_guidance_start),
                    float(cn_guidance_end),
                    float(cn_threshold_a),
                    float(cn_threshold_b),
                ),
                StudioControlNetSlot(
                    "ControlNet unit 2",
                    bool(cn2_enable),
                    cn2_model_id,
                    cn2_module,
                    cn2_image,
                    float(cn2_weight),
                    float(cn2_guidance_start),
                    float(cn2_guidance_end),
                    float(cn2_threshold_a),
                    float(cn2_threshold_b),
                ),
                StudioControlNetSlot(
                    "ControlNet unit 3",
                    bool(cn3_enable),
                    cn3_model_id,
                    cn3_module,
                    cn3_image,
                    float(cn3_weight),
                    float(cn3_guidance_start),
                    float(cn3_guidance_end),
                    float(cn3_threshold_a),
                    float(cn3_threshold_b),
                ),
            ],
            mode=mode,
            controlnet=controlnet,
            checkpoint_architecture=checkpoint_architecture,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if units:
        request = request.model_copy(update={"controlnet_units": units})
        control_images = control_images_list

    return request, init_images, mask_images, before_image, mode, control_images
