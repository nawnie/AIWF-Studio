from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.web.registry import WebRegistry


def _format_tag_cloud(counts: list[tuple[str, int]], limit: int = 16) -> str:
    if not counts:
        return "_No tagged images yet — add hashtags when generating in Studio._"
    chips = [f"`#{tag}` · {count}" for tag, count in counts[:limit]]
    more = len(counts) - limit
    suffix = f"  \n_+{more} more tags_" if more > 0 else ""
    return "**Browse by tag**  \n" + "  \n".join(chips) + suffix


def register_library(registry: WebRegistry) -> None:
    @registry.tab("Library", order=20)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        tag_service = ctx.tags

        with gr.Column(elem_classes=["aiwf-library"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Library", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Search saved outputs by hashtag. Tags are embedded in PNG metadata from Studio generations. "
                    "Click **Refresh** to scan your output folder.",
                    elem_classes=["aiwf-page-intro"],
                )

            with gr.Row(elem_classes=["aiwf-panel"]):
                search = gr.Textbox(
                    label="Search tags",
                    placeholder="#portrait  #client-work",
                    scale=3,
                    elem_classes=["aiwf-tags-input"],
                )
                filter_tag = gr.Dropdown(
                    label="Filter",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    scale=1,
                )
                refresh_btn = gr.Button("Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"], scale=1)

            tag_cloud = gr.Markdown(_format_tag_cloud([]), elem_classes=["aiwf-tag-cloud"])
            result_count = gr.Markdown("", elem_classes=["aiwf-library-count"])

            gallery = gr.Gallery(
                label="Results",
                columns=4,
                object_fit="contain",
                allow_preview=True,
                elem_classes=["aiwf-library-gallery"],
            )

            with gr.Accordion("Image details", open=False, elem_classes=["aiwf-meta-accordion"]):
                selected_tags = gr.Markdown("", elem_classes=["aiwf-tag-summary"])
                details = gr.Textbox(
                    label="Infotext",
                    lines=6,
                    show_label=False,
                    elem_classes=["aiwf-gen-info"],
                )

        library_outputs = [gallery, tag_cloud, result_count, filter_tag, selected_tags, details]

        def load_library(query: str | None):
            entries = tag_service.scan_library()
            filtered = tag_service.filter_entries(entries, query)
            images = [str(entry.path) for entry in filtered]
            counts = tag_service.collect_tag_counts(entries)
            count_line = f"**{len(filtered)}** of **{len(entries)}** images"
            if (query or "").strip():
                count_line += f" matching `{(query or '').strip()}`"
            tag_choices = [tag for tag, _count in counts]
            return (
                images,
                _format_tag_cloud(counts),
                count_line,
                gr.update(choices=tag_choices, value=None),
                "",
                "",
            )

        def show_details(evt: gr.SelectData, query: str | None):
            entries = tag_service.filter_entries(tag_service.scan_library(), query)
            index = evt.index
            if index is None or index >= len(entries):
                return "", ""
            entry = entries[index]
            if entry.tags:
                tag_line = "**Tags** " + " · ".join(f"`#{tag}`" for tag in entry.tags)
            else:
                tag_line = "_No tags on this image_"
            return tag_line, entry.infotext or "_No generation metadata found_"

        def pick_filter_tag(tag: str | None):
            if not tag:
                return ""
            return f"#{tag}"

        refresh_btn.click(load_library, inputs=[search], outputs=library_outputs, show_progress=False)
        search.submit(load_library, inputs=[search], outputs=library_outputs, show_progress=False)
        search.change(load_library, inputs=[search], outputs=library_outputs, show_progress=False)
        filter_tag.change(pick_filter_tag, inputs=[filter_tag], outputs=[search]).then(
            load_library,
            inputs=[search],
            outputs=library_outputs,
            show_progress=False,
        )
        gallery.select(show_details, inputs=[search], outputs=[selected_tags, details], show_progress=False)

        if tab is not None:
            tab.select(load_library, inputs=[search], outputs=library_outputs, show_progress=False)
