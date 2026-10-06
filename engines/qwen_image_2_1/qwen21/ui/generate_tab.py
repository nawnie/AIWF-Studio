"""Generate and Edit tabs. One widget class, two modes; edit adds references, annotation and canvas controls."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSplitter, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QSize

from .. import presets as P
from ..prompts import load_pe_system_prompt
from ..workflow_builder import (ControlNetSpec, GenerationSpec, SpecError, build_prompt, describe_prompt,
                                export_prompt, has_errors, validate_prompt)
from .widgets import AnnotationCanvas, ImageView, JobWorker, LoraStack, ModelPanel, ReferenceSlots, SamplerPanel


class GenerationTab(QWidget):
    """mode='t2i' -> Generate tab; mode='edit' -> Edit tab."""

    log = Signal(str)

    def __init__(self, ctx, mode: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.mode = mode
        self.worker: JobWorker | None = None
        self.last_prompt: dict[str, Any] | None = None
        self.last_spec: GenerationSpec | None = None
        self._build_ui()
        self.ctx.models_changed.connect(self._refresh_lists)
        self._refresh_lists()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        # prompt
        prompt_box = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_box)
        if self.mode == "edit":
            preset_row = QHBoxLayout()
            self.edit_preset = QComboBox()
            self.edit_preset.addItem("Edit prompt presets…", "")
            for label, text in P.EDIT_PROMPT_PRESETS.items():
                self.edit_preset.addItem(label, text)
            self.edit_preset.currentIndexChanged.connect(self._apply_edit_preset)
            preset_row.addWidget(self.edit_preset, 1)
            for n in range(1, 4):
                btn = QPushButton(f"<image{n}>")
                btn.setToolTip("Insert a reference token at the cursor")
                btn.clicked.connect(lambda _=False, n=n: self.prompt.insertPlainText(f"<image{n}>"))
                preset_row.addWidget(btn)
            prompt_layout.addLayout(preset_row)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "Describe the image. For transparency add: 'This is an RGBA image with transparency…'"
            if self.mode == "t2i" else
            "Edit instruction. image_1 is edited; refer to others as <image2>, <image3>… Paint on image_1 to mark a region.")
        self.prompt.setMinimumHeight(90)
        prompt_layout.addWidget(self.prompt)
        hint_row = QHBoxLayout()
        self.rgba_btn = QPushButton("Add RGBA/transparent hint")
        self.rgba_btn.clicked.connect(lambda: self.prompt.appendPlainText(P.RGBA_PROMPT_HINT))
        hint_row.addWidget(self.rgba_btn)
        self.pe_enable = QCheckBox("Prompt enhancer (TextGenerate)")
        self.pe_enable.setToolTip("Rewrites the prompt with the official Qwen3.5-9B prompt-enhancer model before encoding. "
                                  "Loads a second 9.5 GB model; slower, often better composition.")
        self.pe_thinking = QCheckBox("thinking")
        hint_row.addWidget(self.pe_enable)
        hint_row.addWidget(self.pe_thinking)
        hint_row.addStretch(1)
        prompt_layout.addLayout(hint_row)
        left_layout.addWidget(prompt_box)

        # references (edit)
        if self.mode == "edit":
            self.refs = ReferenceSlots()
            self.refs.changed.connect(self._refs_changed)
            left_layout.addWidget(self.refs)
            ref_opts = QGroupBox("Reference handling")
            ref_form = QFormLayout(ref_opts)
            self.ref_resolution = QComboBox()
            for value, label in ((0, "0 — keep each reference at its own size"), (512, "512 px budget"),
                                 (768, "768 px budget"), (1024, "1024 px budget (default)"),
                                 (1536, "1536 px budget"), (2048, "2048 px budget (native 2K, slow)")):
                self.ref_resolution.addItem(label, value)
            self.ref_resolution.setCurrentIndex(3)
            self.ref_resolution.setToolTip("References are resized to about N×N pixels keeping aspect. "
                                           "Canvas cost follows image_1's size: 12 MP at 0 is ~20x slower than 1 MP.")
            ref_form.addRow("Reference resolution", self.ref_resolution)
            self.canvas_mode = QComboBox()
            self.canvas_mode.addItem("Match image_1 (recommended for edits)", "match_ref")
            self.canvas_mode.addItem("Custom canvas size", "custom")
            ref_form.addRow("Output canvas", self.canvas_mode)
            left_layout.addWidget(ref_opts)

        # resolution
        res_box = QGroupBox("Resolution" if self.mode == "t2i" else "Custom canvas (when selected above)")
        res_form = QFormLayout(res_box)
        self.aspect = QComboBox()
        for key in P.ONE_MP:
            self.aspect.addItem(key)
        self.megapixels = QComboBox()
        self.megapixels.addItem("~1 MP (fast, template default)", "1mp")
        self.megapixels.addItem("Native 2K (model card sizes)", "2k")
        self.width = QSpinBox()
        self.height = QSpinBox()
        for spin in (self.width, self.height):
            spin.setRange(256, 4096)
            spin.setSingleStep(32)
            spin.setValue(1024)
        self.aspect.currentIndexChanged.connect(self._apply_resolution_preset)
        self.megapixels.currentIndexChanged.connect(self._apply_resolution_preset)
        res_row = QHBoxLayout()
        res_row.addWidget(self.aspect)
        res_row.addWidget(self.megapixels)
        res_form.addRow("Preset", res_row)
        wh_row = QHBoxLayout()
        wh_row.addWidget(QLabel("W"))
        wh_row.addWidget(self.width)
        wh_row.addWidget(QLabel("H"))
        wh_row.addWidget(self.height)
        self.batch = QSpinBox()
        self.batch.setRange(1, 4)
        wh_row.addWidget(QLabel("Batch"))
        wh_row.addWidget(self.batch)
        res_form.addRow(wh_row)
        left_layout.addWidget(res_box)

        self.models = ModelPanel()
        self.models.refresh.clicked.connect(self.ctx.refresh_models)
        left_layout.addWidget(self.models)
        self.loras = LoraStack()
        left_layout.addWidget(self.loras)
        self.sampler = SamplerPanel()
        left_layout.addWidget(self.sampler)

        # controlnet
        cn_box = QGroupBox("Fun ControlNet Union (optional)")
        cn_box.setCheckable(True)
        cn_box.setChecked(False)
        self.cn_box = cn_box
        cn_form = QFormLayout(cn_box)
        self.cn_patch = QComboBox()
        self.cn_patch.setEditable(True)
        cn_form.addRow("Model patch", self.cn_patch)
        self.cn_image = QLineEdit()
        cn_pick = QPushButton("…")
        cn_pick.clicked.connect(lambda: self._pick_file(self.cn_image))
        cn_row = QHBoxLayout()
        cn_row.addWidget(self.cn_image)
        cn_row.addWidget(cn_pick)
        cn_form.addRow("Control map (Canny/Depth/HED/Lineart/MLSD/Pose/Scribble/Gray)", cn_row)
        self.cn_strength = QDoubleSpinBox()
        self.cn_strength.setRange(0.0, 2.0)
        self.cn_strength.setSingleStep(0.05)
        self.cn_strength.setValue(1.0)
        self.cn_start = QDoubleSpinBox()
        self.cn_start.setRange(0.0, 1.0)
        self.cn_start.setSingleStep(0.05)
        self.cn_end = QDoubleSpinBox()
        self.cn_end.setRange(0.0, 1.0)
        self.cn_end.setSingleStep(0.05)
        self.cn_end.setValue(1.0)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Strength"))
        srow.addWidget(self.cn_strength)
        srow.addWidget(QLabel("Start"))
        srow.addWidget(self.cn_start)
        srow.addWidget(QLabel("End"))
        srow.addWidget(self.cn_end)
        cn_form.addRow(srow)
        if self.mode == "edit":
            self.cn_inpaint = QCheckBox("Inpaint: keep image_1 outside the painted mask (uses your strokes as the mask)")
            cn_form.addRow(self.cn_inpaint)
        left_layout.addWidget(cn_box)

        # run
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Generate" if self.mode == "t2i" else "Edit")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.validate_btn = QPushButton("Validate only")
        self.validate_btn.clicked.connect(lambda: self.validate(show=True))
        self.export_btn = QPushButton("Export workflow JSON…")
        self.export_btn.clicked.connect(self.export_workflow)
        run_row.addWidget(self.run_btn, 2)
        run_row.addWidget(self.cancel_btn)
        run_row.addWidget(self.validate_btn)
        run_row.addWidget(self.export_btn)
        left_layout.addLayout(run_row)
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(460)

        # right: preview / canvas / result
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = QLabel("idle")
        self.status.setStyleSheet("color:#9aa5b1;")
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.status)
        self.views = QTabWidget()
        self.result_view = ImageView()
        self.result_view.caption = "Result appears here (alpha shown over a checkerboard)."
        self.views.addTab(self.result_view, "Result")
        if self.mode == "edit":
            canvas_wrap = QWidget()
            canvas_layout = QVBoxLayout(canvas_wrap)
            tools = QHBoxLayout()
            self.brush_color = QComboBox()
            self.brush_color.addItems(list(AnnotationCanvas.COLORS))
            self.brush_size = QSpinBox()
            self.brush_size.setRange(2, 200)
            self.brush_size.setValue(24)
            clear_btn = QPushButton("Clear marks")
            tools.addWidget(QLabel("Mark color"))
            tools.addWidget(self.brush_color)
            tools.addWidget(QLabel("Size"))
            tools.addWidget(self.brush_size)
            tools.addWidget(clear_btn)
            tools.addStretch(1)
            self.canvas = AnnotationCanvas()
            self.brush_color.currentTextChanged.connect(lambda c: setattr(self.canvas, "color_name", c))
            self.brush_size.valueChanged.connect(lambda v: setattr(self.canvas, "brush_size", v))
            clear_btn.clicked.connect(self.canvas.clear_strokes)
            canvas_layout.addLayout(tools)
            canvas_layout.addWidget(self.canvas, 1)
            canvas_layout.addWidget(QLabel("Marks are burned into image_1 (the model reads 'the red area'). "
                                           "With Fun ControlNet inpaint they also become the mask."))
            self.views.addTab(canvas_wrap, "Mark image_1")
        self.preview_view = ImageView()
        self.preview_view.caption = "Live sampler preview"
        self.views.addTab(self.preview_view, "Live preview")
        self.workflow_text = QPlainTextEdit()
        self.workflow_text.setReadOnly(True)
        self.workflow_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.views.addTab(self.workflow_text, "Workflow (API nodes)")
        right_layout.addWidget(self.views, 1)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(88, 88))
        self.gallery.setFixedHeight(120)
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.itemClicked.connect(lambda item: self.result_view.set_path(item.data(Qt.ItemDataRole.UserRole)))
        self.gallery.itemDoubleClicked.connect(self._send_to_edit)
        right_layout.addWidget(self.gallery)
        self.rewritten = QPlainTextEdit()
        self.rewritten.setReadOnly(True)
        self.rewritten.setPlaceholderText("Prompt-enhancer output appears here after a run.")
        self.rewritten.setMaximumHeight(70)
        right_layout.addWidget(self.rewritten)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._apply_resolution_preset()

    # ------------------------------------------------------------ helpers
    def _pick_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            target.setText(path)

    def _apply_edit_preset(self) -> None:
        text = self.edit_preset.currentData()
        if text:
            self.prompt.setPlainText(text)
        self.edit_preset.setCurrentIndex(0)

    def _apply_resolution_preset(self) -> None:
        table = P.NATIVE_2K if self.megapixels.currentData() == "2k" else P.ONE_MP
        w, h = table[self.aspect.currentText()]
        self.width.setValue(w)
        self.height.setValue(h)

    def _refs_changed(self) -> None:
        paths = self.refs.paths()
        self.canvas.load(paths[0] if paths else None)

    def _refresh_lists(self) -> None:
        m = self.ctx.models
        self.models.set_lists(m.get("diffusion_models", []), m.get("text_encoders", []), m.get("vae", []))
        self.loras.set_files(m.get("loras", []))
        self.sampler.set_server_options(self.ctx.samplers, self.ctx.schedulers, m.get("loras", []))
        current = self.cn_patch.currentText() or self.ctx.settings.controlnet_patch
        self.cn_patch.clear()
        self.cn_patch.addItems(m.get("model_patches", []))
        self.cn_patch.setEditText(current)

    def _send_to_edit(self, item: QListWidgetItem) -> None:
        self.ctx.send_to_edit.emit(item.data(Qt.ItemDataRole.UserRole))

    def load_reference(self, path: str) -> None:
        if self.mode == "edit":
            self.refs.add_path(path)

    # ------------------------------------------------------------ spec
    def build_spec(self, uploaded: dict[str, str] | None = None) -> GenerationSpec:
        uploaded = uploaded or {}
        dit, te, vae, te_device = self.models.values()
        spec = GenerationSpec(
            mode=self.mode,
            prompt=self.prompt.toPlainText().strip(),
            negative_prompt=self.sampler.negative.toPlainText().strip(),
            width=int(self.width.value()), height=int(self.height.value()), batch_size=int(self.batch.value()),
            seed=int(self.sampler.seed.value()), steps=int(self.sampler.steps.value()),
            cfg=float(self.sampler.cfg.value()), sampler=self.sampler.sampler.currentText().strip(),
            scheduler=self.sampler.scheduler.currentText().strip(),
            dit=dit, text_encoder=te, vae=vae, text_encoder_device=te_device,
            loras=self.loras.entries(), accelerator=self.sampler.accelerator_key(),
            accelerator_lora_name=self.sampler.accel_file.currentText().strip() or None,
            kv_cache_device=self.sampler.kv_device.currentText(), kv_cache_dtype=self.sampler.kv_dtype.currentText(),
            prompt_enhancer=self.pe_enable.isChecked(), pe_thinking=self.pe_thinking.isChecked(),
            fix_guidance=self.sampler.fix_guidance.isChecked(),
            use_save_image_advanced=self.ctx.has_node("SaveImageAdvanced"),
        )
        if spec.prompt_enhancer:
            kind = "t2i" if self.mode == "t2i" else "i2i"
            spec.pe_model = self.ctx.settings.pe_t2i if kind == "t2i" else self.ctx.settings.pe_i2i
            spec.pe_system_prompt = load_pe_system_prompt(kind) or None
        if self.mode == "edit":
            spec.canvas_mode = self.canvas_mode.currentData()
            spec.ref_resolution = int(self.ref_resolution.currentData())
            spec.references = [uploaded.get(f"ref_{i}", Path(p).name) for i, p in enumerate(self.refs.paths(), start=1)]
        if self.cn_box.isChecked():
            cn = ControlNetSpec(patch=self.cn_patch.currentText().strip(), strength=float(self.cn_strength.value()),
                                start_percent=float(self.cn_start.value()), end_percent=float(self.cn_end.value()))
            if self.cn_image.text().strip():
                cn.control_image = uploaded.get("cn_image", Path(self.cn_image.text()).name)
            if self.mode == "edit" and self.cn_inpaint.isChecked():
                cn.inpaint_image = uploaded.get("ref_1_clean", "image_1.png")
                cn.mask_image = uploaded.get("mask", "mask.png")
            spec.controlnet = cn
        return spec

    def _uploads(self) -> dict[str, str]:
        """Local files to upload before running: references (image_1 annotated), control map, mask."""
        uploads: dict[str, str] = {}
        if self.mode == "edit":
            paths = self.refs.paths()
            for i, p in enumerate(paths, start=1):
                uploads[f"ref_{i}"] = p
            if paths and self.canvas.has_strokes:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                scratch = self.ctx.scratch_dir
                uploads["ref_1"] = str(self.canvas.export_annotated(scratch / f"image1_marked_{stamp}.png"))
                if self.cn_box.isChecked() and self.cn_inpaint.isChecked():
                    uploads["mask"] = str(self.canvas.export_mask(scratch / f"mask_{stamp}.png"))
                    uploads["ref_1_clean"] = paths[0]
            elif self.cn_box.isChecked() and self.cn_inpaint.isChecked():
                raise SpecError("Inpaint needs painted strokes on image_1 (Mark image_1 tab) to build the mask")
        if self.cn_box.isChecked() and self.cn_image.text().strip():
            uploads["cn_image"] = self.cn_image.text().strip()
        return uploads

    # ------------------------------------------------------------ actions
    def validate(self, show: bool = False) -> bool:
        try:
            spec = self.build_spec()
            prompt = build_prompt(spec)
        except SpecError as exc:
            QMessageBox.warning(self, "Spec problem", str(exc))
            return False
        issues = validate_prompt(prompt, self.ctx.object_info)
        self.workflow_text.setPlainText(describe_prompt(prompt) + "\n\n" + "\n".join(str(i) for i in issues))
        if has_errors(issues):
            QMessageBox.warning(self, "Workflow validation failed",
                                "\n".join(str(i) for i in issues if i.level == "error")[:3000])
            return False
        if show:
            QMessageBox.information(self, "Validated", f"{len(prompt)} nodes, no errors."
                                    + (f"\n{len(issues)} warning(s), see Workflow tab." if issues else ""))
        return True

    def export_workflow(self) -> None:
        try:
            spec = self.build_spec()
        except SpecError as exc:
            QMessageBox.warning(self, "Spec problem", str(exc))
            return
        default = Path(self.ctx.settings.output_dir) / f"qwen21_{self.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export ComfyUI API workflow", str(default), "JSON (*.json)")
        if path:
            export_prompt(spec, path)
            self.status.setText(f"exported {path} (+ .spec.json)")

    def run(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.ctx.ensure_connected(self):
            return
        self.sampler.next_seed()
        try:
            uploads = self._uploads()
            spec = self.build_spec()
            build_prompt(spec)  # early structural check with placeholder names
        except SpecError as exc:
            QMessageBox.warning(self, "Spec problem", str(exc))
            return
        self.last_spec = spec
        out_dir = Path(self.ctx.settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def build(names: dict[str, str]) -> dict[str, Any]:
            spec_final = self.build_spec(names)
            prompt = build_prompt(spec_final)
            issues = validate_prompt(prompt, self.ctx.object_info)
            if has_errors(issues):
                raise SpecError("validation failed:\n" + "\n".join(str(i) for i in issues if i.level == "error"))
            self.last_prompt = prompt
            return prompt

        self.worker = JobWorker(self.ctx.client, build, uploads, out_dir, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.status.setText)
        self.worker.preview.connect(self._on_preview)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("starting…")
        self.worker.start()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("cancelling…")

    def _on_progress(self, value: int, maximum: int, node: str) -> None:
        if maximum:
            self.progress.setValue(int(value * 100 / maximum))
        self.status.setText(f"step {value}/{maximum}" + (f" · node {node}" if node else ""))

    def _on_preview(self, data: bytes) -> None:
        self.preview_view.set_bytes(data)

    def _on_done(self, result) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)
        saved = self.worker.saved if self.worker else []
        self.status.setText(f"done in {result.elapsed_s:.1f} s · {len(saved)} file(s) → {self.ctx.settings.output_dir}")
        for path in saved:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            pix = QPixmap(str(path))
            if not pix.isNull():
                item.setIcon(pix.scaled(88, 88, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.gallery.insertItem(0, item)
        if saved:
            self.result_view.set_path(saved[0])
            self.views.setCurrentIndex(0)
            self._write_receipt(saved)
        texts = [t for lst in result.text_outputs.values() for t in lst]
        if texts:
            self.rewritten.setPlainText("\n".join(texts))
        if self.last_prompt is not None:
            self.workflow_text.setPlainText(describe_prompt(self.last_prompt))
        self.ctx.record_output(saved)

    def _write_receipt(self, saved: list[Path]) -> None:
        if self.last_spec is None or self.last_prompt is None:
            return
        receipt = {"spec": self.last_spec.to_dict(), "outputs": [str(p) for p in saved],
                   "prompt": self.last_prompt, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        (saved[0].with_suffix(".receipt.json")).write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("failed")
        if message != "cancelled":
            QMessageBox.critical(self, "ComfyUI job failed", message[:4000])
        else:
            self.status.setText("cancelled")
