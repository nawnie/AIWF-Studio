from __future__ import annotations

from uuid import UUID

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import GenerationMode, JobState
from aiwf.web.components.results import result_image
from aiwf.web.registry import WebRegistry


def register_history(registry: WebRegistry) -> None:
    @registry.tab("History", order=70)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = ctx.generation

        def _collect(limit: int = 60):
            """Flatten recent completed jobs into (gallery_images, entries)."""
            gallery: list = []
            entries: list[dict] = []
            for job in service.recent_jobs(limit):
                if job.state != JobState.COMPLETED or job.result is None:
                    continue
                images = job.result.images
                infotexts = job.result.infotexts
                seeds = job.result.seeds
                for index, image in enumerate(images):
                    gallery.append(image)
                    entries.append(
                        {
                            "job_id": str(job.id),
                            "mode": job.request.mode.value,
                            "prompt": job.request.prompt,
                            "infotext": infotexts[index] if index < len(infotexts) else "",
                            "seed": seeds[index] if index < len(seeds) else job.request.seed,
                        }
                    )
            return gallery, entries

        initial_gallery, initial_entries = _collect()

        with gr.Column(elem_classes=["aiwf-history"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("History", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Recent generations from this session, newest first. Select an image to see its "
                    "parameters, copy the infotext, or re-run a text-to-image job.",
                    elem_classes=["aiwf-page-intro"],
                )

            with gr.Column(elem_classes=["aiwf-panel"]):
                with gr.Row(elem_classes=["aiwf-history-toolbar"]):
                    refresh_btn = gr.Button(
                        "Refresh", variant="primary", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"]
                    )
                    rerun_btn = gr.Button(
                        "Re-run selected", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"]
                    )
                    status = gr.Markdown(
                        f"**{len(initial_gallery)}** result(s)." if initial_gallery else "No generations yet.",
                        elem_classes=["aiwf-status-bar", "aiwf-history-status"],
                    )
                gallery = gr.Gallery(
                    value=initial_gallery,
                    columns=4,
                    object_fit="contain",
                    allow_preview=True,
                    height=440,
                    show_label=False,
                    elem_classes=["aiwf-results-gallery", "aiwf-history-gallery"],
                )

            with gr.Row(equal_height=False, elem_classes=["aiwf-history-detail"]):
                with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                    gr.Markdown("Parameters", elem_classes=["aiwf-section-label"])
                    detail = gr.Markdown("_Select an image to see its parameters._")
                    infotext = gr.Textbox(label="Infotext", lines=5, buttons=["copy"])
                with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                    gr.Markdown("Re-run", elem_classes=["aiwf-section-label"])
                    rerun_output = result_image(label="Re-run result")

            entries_state = gr.State(initial_entries)
            selected_state = gr.State(-1)

        def _refresh():
            images, entries = _collect()
            note = "No generations yet." if not images else f"**{len(images)}** result(s)."
            return (
                gr.update(value=images),
                entries,
                note,
                -1,
                "_Select an image to see its parameters._",
                "",
            )

        refresh_btn.click(
            _refresh,
            outputs=[gallery, entries_state, status, selected_state, detail, infotext],
            show_progress=False,
        )

        def _select(entries, evt: gr.SelectData):
            index = evt.index
            if not entries or index is None or index >= len(entries):
                return "_Select an image to see its parameters._", "", -1
            entry = entries[index]
            detail_md = (
                f"**Mode** {entry['mode']} · **Seed** {entry['seed']}  \n"
                f"**Prompt** {entry['prompt'] or '_(none)_'}"
            )
            return detail_md, entry["infotext"], index

        gallery.select(
            _select,
            inputs=[entries_state],
            outputs=[detail, infotext, selected_state],
            show_progress=False,
        )

        def _rerun(entries, selected):
            if not entries or selected is None or selected < 0 or selected >= len(entries):
                raise gr.Error("Select an image first.")
            entry = entries[selected]
            job = service.get_job(UUID(entry["job_id"]))
            if job is None:
                raise gr.Error("That job is no longer in history — try Refresh.")
            if job.request.mode != GenerationMode.TXT2IMG:
                raise gr.Error(
                    "Only text-to-image jobs can be re-run here (img2img/inpaint need their source "
                    "image). Use the infotext with PNG Info → Send instead."
                )
            finished = service.submit(job.request)
            if finished.result is None or not finished.result.images:
                return None, f"**Re-run failed** — {finished.error or 'no image'}"
            return finished.result.images[0], "**Re-run complete.**"

        rerun_btn.click(
            _rerun,
            inputs=[entries_state, selected_state],
            outputs=[rerun_output, status],
            show_progress="minimal",
        )

        if tab is not None:
            tab.select(
                _refresh,
                outputs=[gallery, entries_state, status, selected_state, detail, infotext],
                show_progress=False,
            )
