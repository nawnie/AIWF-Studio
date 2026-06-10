from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.segment import SegmentPoint, SegmentRequest
from aiwf.core.domain.segment_presets import (
    CUSTOM_SEGMENT_PRESET_ID,
    resolve_segment_text_prompt,
    segment_mask_preset_choices,
)
from aiwf.infrastructure.diffusers.mask import editor_from_mask
from aiwf.web.registry import WebRegistry


def _model_choices(models) -> list[tuple[str, str]]:
    return [(model.title, model.id) for model in models]


def register_segment(registry: WebRegistry) -> None:
    @registry.tab("Segment", order=16)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = ctx.segment
        models = service.list_models()

        with gr.Column(elem_classes=["aiwf-segment"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Segment (SAM)", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Generate inpaint masks with Segment Anything. "
                    "Pick a **mask target** from the list, use **click points** on the image, "
                    "or choose Custom for advanced prompts. Masks can feed Studio inpaint and workflows.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False, elem_classes=["aiwf-segment-workspace"]):
                with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                    source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                    sam_model = gr.Dropdown(
                        label="SAM model",
                        choices=_model_choices(models),
                        value=models[0].id if models else None,
                    )
                    mask_preset = gr.Dropdown(
                        label="What to mask",
                        choices=segment_mask_preset_choices(),
                        value="person",
                        info="Used when no click points are set",
                    )
                    custom_prompt = gr.Textbox(
                        label="Custom prompt",
                        placeholder="person.hat  (separate categories with .)",
                        visible=False,
                    )
                    box_threshold = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Detection threshold")
                    with gr.Row():
                        point_x = gr.Number(value=0, precision=0, label="Point X")
                        point_y = gr.Number(value=0, precision=0, label="Point Y")
                        point_label = gr.Radio(["Include", "Exclude"], value="Include", label="Point type")
                    mask_index = gr.Slider(0, 2, value=0, step=1, label="Mask candidate", info="SAM multimask index")
                    dilation = gr.Slider(0, 64, value=0, step=1, label="Mask dilation")
                    refresh_models = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                    run_segment = gr.Button("Generate mask", variant="primary", elem_classes=["aiwf-generate-btn"])

                with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                    mask_output = gr.Image(label="Mask", type="pil", interactive=False)
                    preview_output = gr.Image(label="Preview", type="pil", interactive=False)
                    mask_gallery = gr.Gallery(label="SAM candidates", columns=3, object_fit="contain")
                    status = gr.Markdown("**Ready**", elem_classes=["aiwf-status-bar"])
                    editor_export = gr.JSON(label="ImageEditor payload (for workflows)", visible=False)

        def _refresh():
            refreshed = service.refresh_models()
            return gr.update(choices=_model_choices(refreshed), value=refreshed[0].id if refreshed else None)

        refresh_models.click(_refresh, outputs=[sam_model], show_progress=False)

        def _on_preset_change(preset_id):
            return gr.update(visible=preset_id == CUSTOM_SEGMENT_PRESET_ID)

        mask_preset.change(
            _on_preset_change,
            inputs=[mask_preset],
            outputs=[custom_prompt],
            show_progress=False,
        )

        def _segment(image, model_id, preset_id, custom_text, threshold, px, py, p_label, m_index, dilate):
            if image is None:
                raise gr.Error("Upload an image first.")
            if not model_id and not service.list_models():
                raise gr.Error(f"No SAM models in {service.sam_dir()}")

            has_points = int(px or 0) > 0 or int(py or 0) > 0
            text_prompt = resolve_segment_text_prompt(preset_id, custom_text)
            if not has_points and not text_prompt:
                raise gr.Error("Choose what to mask, set click points, or enter a custom prompt.")

            request = SegmentRequest(
                text_prompt=text_prompt,
                box_threshold=float(threshold),
                points=[
                    SegmentPoint(x=int(px), y=int(py), label=0 if p_label == "Exclude" else 1)
                ]
                if (int(px or 0) > 0 or int(py or 0) > 0)
                else [],
                mask_index=int(m_index),
                dilation=int(dilate),
            )

            mask, preview, candidates, message = service.segment(image, request, model_id=model_id)
            gallery = [mask, *candidates]
            editor = editor_from_mask(image, mask)
            return mask, preview, gallery, f"**{message}**", editor

        run_segment.click(
            _segment,
            inputs=[
                source,
                sam_model,
                mask_preset,
                custom_prompt,
                box_threshold,
                point_x,
                point_y,
                point_label,
                mask_index,
                dilation,
            ],
            outputs=[mask_output, preview_output, mask_gallery, status, editor_export],
            show_progress="minimal",
        )