from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.models import normalize_schedule_id_for_sampler
from aiwf.core.tags import parse_tags
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.request_builder import resolve_checkpoint_id


def run_reactor(
    ctx: AppContext,
    service,
    catalogs: StudioCatalogs,
    workspace_result,
    stored_result,
    source_image,
    source_idx,
    target_idx,
    do_restore,
    restorer_id,
    visibility,
    cf_weight,
    do_blend,
    blend_denoise,
    prompt_text,
    negative_text,
    ckpt_title,
    sampler_label,
    step_count,
    cfg_scale,
    clip_skip_value,
    seed_value,
    vae_id,
    style_name,
    ckpt_map,
    tags_text,
    use_file,
    prompt_file_path,
    model_id,
    gender_source,
    gender_target,
    mask_face,
) -> tuple:
    target = stored_result or workspace_result
    if target is None:
        raise gr.Error("Generate an image first, then run ReActor on the result.")
    if source_image is None:
        raise gr.Error("Upload a source face image.")

    options = FaceSwapOptions(
        source_face_index=max(0, int(source_idx or 0)),
        target_face_index=int(target_idx if target_idx is not None else -1),
        model_id=model_id or "inswapper_128",
        gender_source=int(gender_source or 0),
        gender_target=int(gender_target or 0),
        mask_face=bool(mask_face),
        restore_face=bool(do_restore),
        restorer_id=restorer_id,
        restore_visibility=float(visibility),
        codeformer_weight=float(cf_weight),
    )

    restore_fn = None
    if do_restore and restorer_id:

        def restore_fn(image):
            return ctx.enhance.restore(
                image,
                RestoreOptions(
                    model_id=restorer_id,
                    visibility=float(visibility),
                    codeformer_weight=float(cf_weight),
                ),
            )

    try:
        swapped = ctx.faceswap.swap(target, source_image, options, restore_fn=restore_fn)
    except FaceSwapUnavailable as exc:
        raise gr.Error(str(exc))

    result_image = swapped
    status_parts = ["**ReActor complete.**"]

    if do_blend:
        try:
            ckpt_id = resolve_checkpoint_id(ckpt_title, ckpt_map)
        except gr.Error as exc:
            raise gr.Error(f"No checkpoint available for seam blend. {exc}") from exc
        sampler_id = catalogs.sampler_map.get(sampler_label, "euler_a")
        blend_request = GenerationRequest(
            mode=GenerationMode.IMG2IMG,
            prompt=prompt_text,
            negative_prompt=negative_text,
            prompt_file=prompt_file_path,
            use_prompt_file=bool(use_file),
            style_name=style_name or None,
            tags=parse_tags(tags_text or ""),
            steps=int(step_count),
            cfg_scale=float(cfg_scale),
            seed=int(seed_value),
            sampler=sampler_id,
            scheduler=normalize_schedule_id_for_sampler(sampler_id, "automatic"),
            denoising_strength=float(blend_denoise),
            clip_skip=int(clip_skip_value),
            checkpoint_id=ckpt_id,
            vae_id=vae_id,
        )
        job = service.submit(blend_request, init_images=[swapped])
        if job.result is None or not job.result.images:
            raise gr.Error(job.error or "Seam blend img2img failed.")
        result_image = job.result.images[0]
        status_parts.append(f"Seam blend at **{float(blend_denoise):.2f}** denoise.")

    return (
        gr.update(value=result_image, visible=True),
        result_image,
        " ".join(status_parts),
        gr.update(visible=True, value="Compare"),
        gr.update(visible=False),
    )
