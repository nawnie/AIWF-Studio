from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import JobState
from aiwf.services.generation import GenerationService
from aiwf.web.components.results import format_generation_outputs
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.helpers import format_tag_summary, mode_from_label
from aiwf.web.studio.mode_ui import apply_mode_ui
from aiwf.web.studio.request_builder import build_generation_request
from aiwf.web.studio.session import StudioSession


def _resolve_checkpoint_architecture(
    service: GenerationService,
    *,
    ckpt_title: str | None,
    ckpt_map: dict | None,
) -> str | None:
    if not ckpt_map or not ckpt_title:
        return None
    ckpt_id = ckpt_map.get(ckpt_title)
    if not ckpt_id:
        return None
    return service.resolve_checkpoint(ckpt_id).architecture


class GenerationRunner:
    """Orchestrates Studio generate / continuous / streaming yields."""

    def __init__(self, ctx: AppContext, service: GenerationService, catalogs: StudioCatalogs, session: StudioSession):
        self._ctx = ctx
        self._service = service
        self._catalogs = catalogs
        self._session = session

    def progress_outputs(self, mode_label: str, message: str, preview_image=None) -> tuple:
        mode_ui = apply_mode_ui(self._ctx, mode_label, False, hide_empty=True)
        if preview_image is not None:
            workspace_update = gr.update(value=preview_image, visible=True)
        else:
            workspace_update = gr.update(visible=True)
        status_text = message if message.startswith("**") else f"**{message}**"
        return (
            workspace_update,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(visible=False, value=""),
            "",
            status_text,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            False,
            False,
            *mode_ui,
        )

    def finished_outputs(
        self,
        mode_label: str,
        job,
        before_image,
        *,
        continuous_on: bool,
    ) -> tuple:
        mode_ui = apply_mode_ui(self._ctx, mode_label, False, hide_empty=True)
        can_compare = before_image is not None

        if job.result is None:
            self._session.loop_active = False
            if job.state == JobState.CANCELLED:
                status_text = "**Stopped** — generation cancelled"
            else:
                err = job.error or job.state.value
                status_text = f"**Error** — {err}"
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=""),
                "",
                status_text,
                -1,
                None,
                before_image,
                gr.update(),
                gr.update(value=False),
                False,
                False,
                *apply_mode_ui(self._ctx, mode_label, False),
            )

        infotext = job.result.infotexts[0] if job.result.infotexts else ""
        primary, images, infotext, job_status = format_generation_outputs(
            job.result.images,
            infotext,
            job.state.value,
        )
        gallery_update = gr.update(value=images, visible=len(images) > 1, columns=min(2, len(images)))
        new_seed = job.result.seeds[0] if job.result.seeds else -1
        if new_seed >= 0:
            done_status = f"**Done** — seed **{new_seed}**"
        elif job_status.startswith("**"):
            done_status = job_status
        else:
            done_status = f"**Done** — {job_status}"
        applied_tags = job.request.tags
        if applied_tags:
            self._ctx.tags.remember_tags(applied_tags, save=self._ctx.save_settings)
        tag_line = format_tag_summary(applied_tags)
        quick_tag_update = gr.update(choices=self._ctx.tags.recent_tag_choices())
        toggle_update = gr.update(value=continuous_on) if continuous_on else gr.update()
        return (
            gr.update(value=primary, visible=True),
            gr.update(visible=False),
            gr.update(visible=can_compare, value="Compare"),
            gallery_update,
            gr.update(value=tag_line, visible=bool(tag_line)),
            infotext,
            done_status,
            new_seed,
            primary,
            before_image,
            quick_tag_update,
            toggle_update,
            False,
            False,
            *mode_ui,
        )

    def run_once(self, mode_label: str, all_inputs: tuple, *, keep_continuous_toggle: bool) -> Iterator[tuple]:
        request, init_images, mask_images, before_image, mode, control_images = build_generation_request(
            catalogs=self._catalogs,
            session=self._session,
            mode_label=all_inputs[0],
            editing_mask=all_inputs[1],
            prompt_text=all_inputs[2],
            negative_text=all_inputs[3],
            ckpt_title=all_inputs[4],
            sampler_label=all_inputs[5],
            scheduler_label=all_inputs[6],
            step_count=all_inputs[7],
            cfg_scale=all_inputs[8],
            clip_skip_value=all_inputs[9],
            w=all_inputs[10],
            h=all_inputs[11],
            bs=all_inputs[12],
            bc=all_inputs[13],
            seed_value=all_inputs[14],
            vae_id=all_inputs[15],
            hires_enabled=all_inputs[16],
            hires_scale=all_inputs[17],
            hires_steps=all_inputs[18],
            hires_denoise=all_inputs[19],
            hires_upscaler=all_inputs[20],
            img2img_denoise=all_inputs[21],
            inpaint_denoise_value=all_inputs[22],
            mask_blur_value=all_inputs[23],
            seam_erode_value=all_inputs[24],
            inpaint_area_value=all_inputs[25],
            inpaint_padding_value=all_inputs[26],
            masked_content_value=all_inputs[27],
            source_image=all_inputs[28],
            editor_value=all_inputs[29],
            ckpt_map=all_inputs[30],
            tags_text=all_inputs[31],
            use_file=all_inputs[32],
            prompt_file_path=all_inputs[33],
            dynamic_seed=all_inputs[34],
            style_name=all_inputs[35],
            style_template_prompt=all_inputs[36],
            style_template_negative=all_inputs[37],
            cn_enable=all_inputs[38],
            cn_model_id=all_inputs[39],
            cn_module=all_inputs[40],
            cn_image=all_inputs[41],
            cn_weight=all_inputs[42],
            cn_guidance_start=all_inputs[43],
            cn_guidance_end=all_inputs[44],
            cn_threshold_a=all_inputs[45],
            cn_threshold_b=all_inputs[46],
            cn2_enable=all_inputs[47],
            cn2_model_id=all_inputs[48],
            cn2_module=all_inputs[49],
            cn2_image=all_inputs[50],
            cn2_weight=all_inputs[51],
            cn2_guidance_start=all_inputs[52],
            cn2_guidance_end=all_inputs[53],
            cn2_threshold_a=all_inputs[54],
            cn2_threshold_b=all_inputs[55],
            cn3_enable=all_inputs[56],
            cn3_model_id=all_inputs[57],
            cn3_module=all_inputs[58],
            cn3_image=all_inputs[59],
            cn3_weight=all_inputs[60],
            cn3_guidance_start=all_inputs[61],
            cn3_guidance_end=all_inputs[62],
            cn3_threshold_a=all_inputs[63],
            cn3_threshold_b=all_inputs[64],
            inpaint_source_choice=all_inputs[65],
            controlnet=self._ctx.controlnet,
            checkpoint_architecture=_resolve_checkpoint_architecture(
                self._service,
                ckpt_title=all_inputs[4],
                ckpt_map=all_inputs[30],
            ),
        )
        yield self.progress_outputs(mode_label, "Queued")
        for event in self._service.submit_streaming(
            request,
            init_images=init_images,
            mask_images=mask_images,
            control_images=control_images,
        ):
            if event[0] == "progress":
                _, _step, _total, message, preview = event
                yield self.progress_outputs(mode_label, message, preview)
            else:
                _, job = event
                yield self.finished_outputs(
                    mode_label,
                    job,
                    before_image if mode != "txt2img" else None,
                    continuous_on=keep_continuous_toggle,
                )

    def run(self, *args: Any) -> Iterator[tuple]:
        (
            mode_label,
            editing_mask,
            prompt_text,
            negative_text,
            ckpt_title,
            sampler_label,
            scheduler_label,
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
            hires_upscaler,
            img2img_denoise,
            inpaint_denoise_value,
            mask_blur_value,
            seam_erode_value,
            inpaint_area_value,
            inpaint_padding_value,
            masked_content_value,
            source_image,
            editor_value,
            ckpt_map,
            tags_text,
            use_file,
            prompt_file_path,
            style_name,
            style_template_prompt,
            style_template_negative,
            cn_enable,
            cn_model,
            cn_module,
            cn_image,
            cn_weight,
            cn_guidance_start,
            cn_guidance_end,
            cn_threshold_a,
            cn_threshold_b,
            cn2_enable,
            cn2_model,
            cn2_module,
            cn2_image,
            cn2_weight,
            cn2_guidance_start,
            cn2_guidance_end,
            cn2_threshold_a,
            cn2_threshold_b,
            cn3_enable,
            cn3_model,
            cn3_module,
            cn3_image,
            cn3_weight,
            cn3_guidance_start,
            cn3_guidance_end,
            cn3_threshold_a,
            cn3_threshold_b,
            inpaint_source,
            continuous_enabled,
            cooldown_wait,
        ) = args

        self._session.loop_active = True
        self._ctx.settings.generation_cooldown_seconds = float(cooldown_wait or 0)
        self._ctx.save_settings()

        try:
            run_number = 0
            while self._session.loop_active:
                run_number += 1
                if continuous_enabled and run_number > 1:
                    yield self.progress_outputs(mode_label, f"Run {run_number}")

                base_seed = int(seed_value)
                if base_seed < 0:
                    dynamic_seed = None
                elif continuous_enabled:
                    dynamic_seed = base_seed + run_number - 1
                else:
                    dynamic_seed = base_seed

                request_inputs = (
                    mode_label,
                    editing_mask,
                    prompt_text,
                    negative_text,
                    ckpt_title,
                    sampler_label,
                    scheduler_label,
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
                    hires_upscaler,
                    img2img_denoise,
                    inpaint_denoise_value,
                    mask_blur_value,
                    seam_erode_value,
                    inpaint_area_value,
                    inpaint_padding_value,
                    masked_content_value,
                    source_image,
                    editor_value,
                    ckpt_map,
                    tags_text,
                    use_file,
                    prompt_file_path,
                    dynamic_seed,
                    style_name,
                    style_template_prompt,
                    style_template_negative,
                    cn_enable,
                    cn_model,
                    cn_module,
                    cn_image,
                    cn_weight,
                    cn_guidance_start,
                    cn_guidance_end,
                    cn_threshold_a,
                    cn_threshold_b,
                    cn2_enable,
                    cn2_model,
                    cn2_module,
                    cn2_image,
                    cn2_weight,
                    cn2_guidance_start,
                    cn2_guidance_end,
                    cn2_threshold_a,
                    cn2_threshold_b,
                    cn3_enable,
                    cn3_model,
                    cn3_module,
                    cn3_image,
                    cn3_weight,
                    cn3_guidance_start,
                    cn3_guidance_end,
                    cn3_threshold_a,
                    cn3_threshold_b,
                    inpaint_source,
                )

                for update in self.run_once(
                    mode_label,
                    request_inputs,
                    keep_continuous_toggle=continuous_enabled and self._session.loop_active,
                ):
                    yield update

                if not self._session.loop_active:
                    break
                if not continuous_enabled:
                    break

                wait_s = max(0, int(cooldown_wait or 0))
                for remaining in range(wait_s, 0, -1):
                    if not self._session.loop_active:
                        break
                    yield self.progress_outputs(mode_label, f"Cooling — next run in {remaining}s")
                    time.sleep(1)
        finally:
            self._session.loop_active = False
