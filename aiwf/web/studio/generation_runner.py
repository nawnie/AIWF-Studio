from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import JobState
from aiwf.dev.diagnostics import (
    trace_exception_safe,
    trace_job_record_state,
    trace_studio_generate,
    trace_studio_request_built,
)
from aiwf.services.generation import GenerationService
from aiwf.web.components.results import format_generation_outputs
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.helpers import format_tag_summary
from aiwf.web.studio.mode_ui import apply_mode_ui
from aiwf.web.studio.request_builder import build_generation_request
from aiwf.web.studio.session import StudioSession
from aiwf.web.studio.summaries import result_summary_markdown


ImagePostprocess = Callable[[Any], Any]
GENERATION_RUNNER_INPUT_COUNT = 67


class GenerationRunner:
    """Owns Studio generate, streaming, and continuous-loop orchestration."""

    def __init__(self, ctx: AppContext, service: GenerationService, catalogs: StudioCatalogs, session: StudioSession):
        self._ctx = ctx
        self._service = service
        self._catalogs = catalogs
        self._session = session

    def _checkpoint_architecture(self, checkpoint_id: str) -> str:
        return self._service.resolve_checkpoint(checkpoint_id).architecture

    def progress_outputs(self, mode_label: str, message: str, preview_image=None, hold_image=None) -> tuple:
        mode_ui = apply_mode_ui(self._ctx, mode_label, False, hide_empty=True)
        if preview_image is not None:
            workspace_update = gr.update(value=preview_image, visible=True)
        elif hold_image is not None:
            workspace_update = gr.update(value=hold_image, visible=True)
        else:
            workspace_update = gr.update()
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
            gr.update(),
            False,
            False,
            *mode_ui,
        )

    def finished_outputs(self, mode_label: str, job, before_image, *, continuous_on: bool) -> tuple:
        mode_ui = apply_mode_ui(self._ctx, mode_label, False, hide_empty=True)
        can_compare = before_image is not None

        if job.result is None:
            self._session.loop_active = False
            if job.state == JobState.CANCELLED:
                status_text = "**Stopped** -- generation cancelled"
            else:
                err = job.error or job.state.value
                status_text = f"**Error** -- {err}"
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=""),
                "",
                status_text,
                -1,
                [],
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
            result_summary_markdown(job, new_seed, job_status),
            new_seed,
            list(job.result.seeds),
            primary,
            before_image,
            quick_tag_update,
            toggle_update,
            False,
            False,
            *mode_ui,
        )

    def build_request(self, inputs: tuple):
        request, init_images, mask_images, before_image, mode, control_images = build_generation_request(
            catalogs=self._catalogs,
            session=self._session,
            mode_label=inputs[0],
            editing_mask=inputs[1],
            prompt_text=inputs[2],
            negative_text=inputs[3],
            ckpt_title=inputs[4],
            sampler_label=inputs[5],
            scheduler_label=inputs[6],
            step_count=inputs[7],
            cfg_scale=inputs[8],
            clip_skip_value=inputs[9],
            w=inputs[10],
            h=inputs[11],
            bs=inputs[12],
            bc=inputs[13],
            seed_value=inputs[14],
            vae_id=inputs[15],
            hires_enabled=inputs[16],
            hires_scale=inputs[17],
            hires_steps=inputs[18],
            hires_denoise=inputs[19],
            hires_upscaler=inputs[20],
            img2img_denoise=inputs[21],
            inpaint_denoise_value=inputs[22],
            mask_blur_value=inputs[23],
            seam_erode_value=inputs[24],
            inpaint_area_value=inputs[25],
            inpaint_padding_value=inputs[26],
            masked_content_value=inputs[27],
            source_image=inputs[28],
            editor_value=inputs[29],
            ckpt_map=inputs[30],
            tags_text=inputs[31],
            use_file=inputs[32],
            prompt_file_path=inputs[33],
            dynamic_seed=inputs[34],
            style_name=inputs[35],
            style_template_prompt=inputs[36],
            style_template_negative=inputs[37],
            cn_enable=inputs[38],
            cn_model_id=inputs[39],
            cn_module=inputs[40],
            cn_image=inputs[41],
            cn_weight=inputs[42],
            cn_guidance_start=inputs[43],
            cn_guidance_end=inputs[44],
            cn_threshold_a=inputs[45],
            cn_threshold_b=inputs[46],
            cn2_enable=inputs[47],
            cn2_model_id=inputs[48],
            cn2_module=inputs[49],
            cn2_image=inputs[50],
            cn2_weight=inputs[51],
            cn2_guidance_start=inputs[52],
            cn2_guidance_end=inputs[53],
            cn2_threshold_a=inputs[54],
            cn2_threshold_b=inputs[55],
            cn3_enable=inputs[56],
            cn3_model_id=inputs[57],
            cn3_module=inputs[58],
            cn3_image=inputs[59],
            cn3_weight=inputs[60],
            cn3_guidance_start=inputs[61],
            cn3_guidance_end=inputs[62],
            cn3_threshold_a=inputs[63],
            cn3_threshold_b=inputs[64],
            inpaint_source_choice=inputs[65],
            controlnet=self._ctx.controlnet,
            resolve_checkpoint_architecture=self._checkpoint_architecture,
            default_hr_upscaler=self._ctx.settings.default_hr_upscaler,
        )
        trace_studio_request_built(
            mode=mode,
            width=getattr(request, "width", None),
            height=getattr(request, "height", None),
            init_count=len(init_images or []),
            mask_count=len(mask_images or []),
            control_count=len(control_images or []),
            checkpoint_id=getattr(request, "checkpoint_id", None),
        )
        return request, init_images, mask_images, before_image, mode, control_images

    def run_once(
        self,
        mode_label: str,
        inputs: tuple,
        *,
        keep_continuous_toggle: bool,
        image_postprocess: ImagePostprocess | None = None,
    ) -> Iterator[tuple]:
        try:
            request, init_images, mask_images, before_image, mode, control_images = self.build_request(inputs)
        except Exception as exc:
            trace_exception_safe("studio.request_build", exc, mode=mode_label)
            raise

        hold_image = before_image or (init_images[0] if init_images else None)
        yield self.progress_outputs(mode_label, "Queued", hold_image=hold_image)
        try:
            for event in self._service.submit_streaming(
                request,
                init_images=init_images,
                mask_images=mask_images,
                control_images=control_images,
                image_postprocess=image_postprocess,
            ):
                if event[0] == "progress":
                    _, _step, _total, message, preview = event
                    yield self.progress_outputs(
                        mode_label,
                        message,
                        preview_image=preview,
                        hold_image=hold_image if preview is None else None,
                    )
                else:
                    _, job = event
                    trace_job_record_state(job.id, job.state, job.error)
                    yield self.finished_outputs(
                        mode_label,
                        job,
                        before_image if mode != "txt2img" else None,
                        continuous_on=keep_continuous_toggle,
                    )
        except Exception as exc:
            trace_exception_safe(
                "studio.generate_stream",
                exc,
                mode=mode_label,
                checkpoint_id=request.checkpoint_id,
            )
            raise

    def run(
        self,
        *args: Any,
        image_postprocess: ImagePostprocess | None = None,
        input_count: int | None = None,
    ) -> Iterator[tuple]:
        mode_label = args[0]
        editing_mask = args[1]
        seed_value = args[14]
        source_image = args[28]
        editor_value = args[29]
        cn_enable = args[38]
        inpaint_source = args[64]
        continuous_enabled = args[65]
        cooldown_wait = args[66]

        self._session.loop_active = True
        self._ctx.settings.generation_cooldown_seconds = float(cooldown_wait or 0)
        self._ctx.save_settings()

        try:
            run_number = 0
            while self._session.loop_active:
                run_number += 1
                trace_studio_generate(
                    run_number=run_number,
                    mode_label=mode_label,
                    continuous=bool(continuous_enabled),
                    editing_mask=bool(editing_mask),
                    has_source=source_image is not None,
                    has_editor_value=editor_value is not None,
                    cn_enabled=bool(cn_enable),
                    input_count=input_count or len(args),
                )
                if continuous_enabled and run_number > 1:
                    yield self.progress_outputs(mode_label, f"Run {run_number}", hold_image=source_image)

                base_seed = int(seed_value)
                if base_seed < 0:
                    dynamic_seed = None
                elif continuous_enabled:
                    dynamic_seed = base_seed + run_number - 1
                else:
                    dynamic_seed = base_seed

                request_inputs = (*args[:34], dynamic_seed, *args[34:64], inpaint_source)
                for update in self.run_once(
                    mode_label,
                    request_inputs,
                    keep_continuous_toggle=continuous_enabled and self._session.loop_active,
                    image_postprocess=image_postprocess,
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
                    yield self.progress_outputs(mode_label, f"Cooling -- next run in {remaining}s")
                    time.sleep(1)
        finally:
            self._session.loop_active = False
