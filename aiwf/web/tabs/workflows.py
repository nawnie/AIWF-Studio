from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.workflow import WorkflowDefinition
from aiwf.web.registry import WebRegistry


def _format_step_log(run) -> str:
    if not run or not run.steps:
        return ""
    lines = []
    for step in run.steps:
        path = f" → `{step.image_path}`" if step.image_path else ""
        lines.append(f"**{step.label}** ({step.step_type.value}){path}")
        if step.infotext:
            lines.append(f"> {step.infotext}")
    return "\n\n".join(lines)


def register_workflows(registry: WebRegistry) -> None:
    @registry.tab("Workflows", order=18)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = ctx.workflows
        choices = service.list_choices()
        default_key = choices[0][1] if choices else None
        default_workflow = service.load(default_key)
        default_json = service.to_json(default_workflow) if default_workflow else "{}"

        with gr.Column(elem_classes=["aiwf-workflows"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Workflows", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Chain generation and enhance steps into reusable pipelines. "
                    "Workflows are stored as JSON — edit parameters, save, and re-run anytime. "
                    "This tab is still WIP and has not had a full reliability pass yet.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(
                    f"**Saved workflows** → `{service.workflows_dir()}`",
                    elem_classes=["aiwf-page-path"],
                )

            with gr.Row(equal_height=False, elem_classes=["aiwf-workflow-workspace"]):
                with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                    workflow_select = gr.Dropdown(
                        label="Load workflow",
                        choices=choices,
                        value=default_key,
                        allow_custom_value=False,
                    )
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                        new_btn = gr.Button("New blank", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                    workflow_json = gr.Code(
                        label="Workflow JSON",
                        language="json",
                        value=default_json,
                        lines=22,
                        elem_classes=["aiwf-workflow-json"],
                    )
                    with gr.Row():
                        save_btn = gr.Button("Save to disk", variant="secondary")
                        delete_btn = gr.Button("Delete saved", elem_classes=["aiwf-btn-ghost"])
                    seed_image = gr.Image(
                        label="Seed image (optional — required when the first step is not txt2img)",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )
                    run_btn = gr.Button("Run workflow", variant="primary", elem_classes=["aiwf-generate-btn"])

                with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                    final_image = gr.Image(label="Final result", type="pil", interactive=False)
                    step_gallery = gr.Gallery(
                        label="Step outputs",
                        columns=3,
                        object_fit="contain",
                        allow_preview=True,
                        elem_classes=["aiwf-workflow-gallery"],
                    )
                    status = gr.Markdown("**Ready** — load or edit a workflow, then Run.", elem_classes=["aiwf-status-bar"])
                    step_log = gr.Markdown("", elem_classes=["aiwf-workflow-log"])

        def _blank_workflow() -> str:
            blank = WorkflowDefinition(name="My workflow", description="", steps=[])
            return service.to_json(blank)

        def _refresh_choices(select_key: str | None):
            updated = service.list_choices()
            value = select_key if select_key and any(key == select_key for _, key in updated) else (
                updated[0][1] if updated else None
            )
            return gr.update(choices=updated, value=value)

        def _load_selected(key: str | None):
            workflow = service.load(key)
            if workflow is None:
                return "{}", "**No workflow selected.**", []
            return service.to_json(workflow), f"**Loaded:** {workflow.name}", []

        refresh_btn.click(
            _refresh_choices,
            inputs=[workflow_select],
            outputs=[workflow_select],
            show_progress=False,
        ).then(_load_selected, inputs=[workflow_select], outputs=[workflow_json, status, step_gallery])

        new_btn.click(
            lambda: (_blank_workflow(), "**New blank workflow** — add steps in JSON.", []),
            outputs=[workflow_json, status, step_gallery],
            show_progress=False,
        )

        workflow_select.change(
            _load_selected,
            inputs=[workflow_select],
            outputs=[workflow_json, status, step_gallery],
            show_progress=False,
        )

        def _save_workflow(raw: str):
            try:
                workflow = service.from_json(raw)
            except Exception as exc:
                raise gr.Error(f"Invalid workflow JSON: {exc}") from exc
            path = service.save(workflow)
            updated = service.list_choices()
            key = path.stem
            return (
                gr.update(choices=updated, value=key),
                f"**Saved** `{path.name}`",
            )

        save_btn.click(
            _save_workflow,
            inputs=[workflow_json],
            outputs=[workflow_select, status],
            show_progress=False,
        )

        def _delete_workflow(key: str | None):
            if not key or key.startswith("builtin:"):
                raise gr.Error("Built-in workflows cannot be deleted.")
            if not service.delete(key):
                raise gr.Error("Workflow file not found.")
            updated = service.list_choices()
            new_key = updated[0][1] if updated else None
            workflow = service.load(new_key)
            return (
                gr.update(choices=updated, value=new_key),
                service.to_json(workflow) if workflow else "{}",
                f"**Deleted** `{key}.json`",
                [],
                None,
            )

        delete_btn.click(
            _delete_workflow,
            inputs=[workflow_select],
            outputs=[workflow_select, workflow_json, status, step_gallery, final_image],
            show_progress=False,
        )

        def _run_workflow(raw: str, seed):
            try:
                workflow = service.from_json(raw)
            except Exception as exc:
                raise gr.Error(f"Invalid workflow JSON: {exc}") from exc

            progress_state = {"message": "Starting…"}

            def on_progress(step: int, total: int, message: str) -> None:
                progress_state["message"] = f"Step {step}/{total}: {message}"

            try:
                run, images = service.run(workflow, seed_image=seed, on_progress=on_progress)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

            gallery = images
            final = images[-1] if images else None
            status_line = f"**Done** — {run.summary}"
            if run.final_image_path:
                status_line += f"  \nSaved to `{run.final_image_path}`"
            return final, gallery, status_line, _format_step_log(run)

        run_btn.click(
            _run_workflow,
            inputs=[workflow_json, seed_image],
            outputs=[final_image, step_gallery, status, step_log],
            show_progress="minimal",
        )
