"""Train tab: dataset tools, ai-toolkit / DiffSynth config generation, subprocess runner with live log."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..training import aitoolkit, dataset as ds, diffsynth
from ..training.runner import Progress, TrainingRunner


class _Bridge(QObject):
    line = Signal(str)
    progress = Signal(object)
    exited = Signal(int)


class TrainTab(QWidget):
    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.runner: TrainingRunner | None = None
        self.bridge = _Bridge()
        self.bridge.line.connect(self._append_log)
        self.bridge.progress.connect(self._on_progress)
        self.bridge.exited.connect(self._on_exit)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _dir_row(self, line: QLineEdit, title: str) -> QHBoxLayout:
        btn = QPushButton("…")
        btn.clicked.connect(lambda: self._pick_dir(line, title))
        row = QHBoxLayout()
        row.addWidget(line)
        row.addWidget(btn)
        return row

    def _pick_dir(self, line: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title, line.text() or str(Path.home()))
        if path:
            line.setText(path)

    def _build_ui(self) -> None:
        left = QWidget()
        form_layout = QVBoxLayout(left)

        trainer_box = QGroupBox("Trainer")
        tform = QFormLayout(trainer_box)
        self.trainer = QComboBox()
        self.trainer.addItem("ai-toolkit (ostris) — recommended on 16 GB: int8 ConvRot + layer offload", "aitoolkit")
        self.trainer.addItem("DiffSynth-Studio (official Qwen recipe, bf16, needs offload on 16 GB)", "diffsynth")
        tform.addRow("Backend", self.trainer)
        self.vram_preset = QComboBox()
        self.vram_preset.addItem("16 GB preset (rank 16, 1024, offload everything)", "16")
        self.vram_preset.addItem("24 GB preset (rank 32, 768+1024, no offload)", "24")
        self.vram_preset.currentIndexChanged.connect(self._apply_vram_preset)
        tform.addRow("VRAM preset", self.vram_preset)
        form_layout.addWidget(trainer_box)

        data_box = QGroupBox("Dataset")
        dform = QFormLayout(data_box)
        self.dataset_dir = QLineEdit()
        dform.addRow("Images + .txt captions", self._dir_row(self.dataset_dir, "Dataset folder"))
        self.control_dirs = [QLineEdit() for _ in range(3)]
        for i, line in enumerate(self.control_dirs, start=1):
            line.setPlaceholderText(f"optional: reference/control folder {i} (same basenames) → edit LoRA")
            dform.addRow(f"Reference folder {i}", self._dir_row(line, f"Reference folder {i}"))
        self.trigger = QLineEdit()
        self.trigger.setPlaceholderText("e.g. ohwx_shawn — written into captions, not relied on at runtime")
        dform.addRow("Trigger word", self.trigger)
        self.default_caption = QLineEdit()
        self.default_caption.setPlaceholderText("caption used for images without a .txt (optional)")
        dform.addRow("Default caption", self.default_caption)
        self.rgba = QCheckBox("Dataset has transparent PNGs — train RGBA output")
        dform.addRow(self.rgba)
        tools = QHBoxLayout()
        scan = QPushButton("Scan dataset")
        scan.clicked.connect(self.scan)
        fill = QPushButton("Write missing captions")
        fill.clicked.connect(self.fill_captions)
        prepend = QPushButton("Prepend trigger to captions")
        prepend.clicked.connect(self.prepend_trigger)
        tools.addWidget(scan)
        tools.addWidget(fill)
        tools.addWidget(prepend)
        dform.addRow(tools)
        form_layout.addWidget(data_box)

        hp_box = QGroupBox("LoRA hyperparameters")
        hform = QFormLayout(hp_box)
        self.name = QLineEdit("qwen21_lora_v1")
        hform.addRow("Name", self.name)
        self.rank = QSpinBox()
        self.rank.setRange(1, 256)
        self.rank.setValue(16)
        self.alpha = QSpinBox()
        self.alpha.setRange(1, 256)
        self.alpha.setValue(16)
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Rank"))
        rrow.addWidget(self.rank)
        rrow.addWidget(QLabel("Alpha"))
        rrow.addWidget(self.alpha)
        hform.addRow(rrow)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1e-2)
        self.lr.setSingleStep(1e-5)
        self.lr.setValue(1e-4)
        self.steps = QSpinBox()
        self.steps.setRange(50, 20000)
        self.steps.setValue(2000)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("LR"))
        srow.addWidget(self.lr)
        srow.addWidget(QLabel("Steps"))
        srow.addWidget(self.steps)
        hform.addRow(srow)
        self.resolution = QComboBox()
        self.resolution.addItem("1024", "1024")
        self.resolution.addItem("768", "768")
        self.resolution.addItem("512 + 1024 (buckets)", "512,1024")
        self.resolution.addItem("768 + 1024 (buckets)", "768,1024")
        hform.addRow("Resolution buckets", self.resolution)
        self.timestep = QComboBox()
        self.timestep.addItems(["shift", "sigmoid", "weighted", "linear"])
        hform.addRow("Timestep sampling", self.timestep)
        self.save_every = QSpinBox()
        self.save_every.setRange(50, 5000)
        self.save_every.setValue(250)
        hform.addRow("Save every N steps", self.save_every)
        form_layout.addWidget(hp_box)

        mem_box = QGroupBox("Memory (ai-toolkit)")
        mform = QFormLayout(mem_box)
        self.qtype = QComboBox()
        self.qtype.addItems(["convrot8", "float8", "uint4", "none"])
        self.qtype.setToolTip("convrot8 = int8 rotation quantization (same family as the ComfyUI int8_convrot files).")
        mform.addRow("DiT quant", self.qtype)
        self.qtype_te = QComboBox()
        self.qtype_te.addItems(["convrot8", "float8", "none"])
        mform.addRow("Text encoder quant", self.qtype_te)
        self.low_vram = QCheckBox("low_vram")
        self.low_vram.setChecked(True)
        self.offload = QCheckBox("layer offloading (slower, fits 12-16 GB)")
        self.offload.setChecked(True)
        mform.addRow(self.low_vram)
        mform.addRow(self.offload)
        self.offload_pct = QDoubleSpinBox()
        self.offload_pct.setRange(0.0, 1.0)
        self.offload_pct.setSingleStep(0.1)
        self.offload_pct.setValue(1.0)
        mform.addRow("Transformer offload fraction", self.offload_pct)
        self.cache = QCheckBox("cache latents + text embeddings (unloads the encoder)")
        self.cache.setChecked(True)
        mform.addRow(self.cache)
        self.sampling = QCheckBox("sample during training (costs VRAM/time)")
        mform.addRow(self.sampling)
        self.sample_prompt = QLineEdit("[trigger] portrait photo, soft window light, 85mm")
        mform.addRow("Sample prompt", self.sample_prompt)
        form_layout.addWidget(mem_box)

        self.name_or_path = QLineEdit(aitoolkit.DEFAULT_NAME_OR_PATH)
        self.name_or_path.setToolTip("HF repo id (Comfy-Org repack), a local folder with text_encoder/ + vae/, "
                                     "or the single qwen_image_2.1_bf16.safetensors file from your ComfyUI models folder.")
        mform.addRow("Base weights (name_or_path)", self.name_or_path)

        act = QHBoxLayout()
        self.write_btn = QPushButton("Write config")
        self.write_btn.clicked.connect(self.write_config)
        self.start_btn = QPushButton("Start training")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        act.addWidget(self.write_btn)
        act.addWidget(self.start_btn, 2)
        act.addWidget(self.stop_btn)
        form_layout.addLayout(act)
        form_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(460)

        right = QWidget()
        rlayout = QVBoxLayout(right)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = QLabel("idle")
        self.status.setStyleSheet("color:#9aa5b1;")
        rlayout.addWidget(self.progress)
        rlayout.addWidget(self.status)
        self.config_view = QPlainTextEdit()
        self.config_view.setReadOnly(True)
        self.config_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.config_view.setPlaceholderText("Generated trainer config / command appears here.")
        rlayout.addWidget(self.config_view, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        rlayout.addWidget(self.log, 2)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ------------------------------------------------------------ dataset tools
    def _control_dirs(self) -> list[str]:
        return [l.text().strip() for l in self.control_dirs if l.text().strip()]

    def scan(self) -> None:
        report = ds.scan_dataset(self.dataset_dir.text().strip(), self._control_dirs())
        if report.rgba_images and not self.rgba.isChecked():
            report.warnings.append("transparent PNGs found: consider enabling RGBA training")
        self.config_view.setPlainText(report.summary())

    def fill_captions(self) -> None:
        folder = self.dataset_dir.text().strip()
        caption = self.default_caption.text().strip()
        if not folder or not caption:
            QMessageBox.information(self, "Captions", "Set the dataset folder and a default caption first.")
            return
        n = ds.ensure_captions(folder, caption, self.trigger.text().strip())
        self.status.setText(f"wrote {n} caption file(s)")

    def prepend_trigger(self) -> None:
        folder, trigger = self.dataset_dir.text().strip(), self.trigger.text().strip()
        if not folder or not trigger:
            QMessageBox.information(self, "Trigger", "Set the dataset folder and a trigger word first.")
            return
        n = ds.prepend_trigger(folder, trigger)
        self.status.setText(f"prepended trigger to {n} caption(s)")

    def _apply_vram_preset(self) -> None:
        is16 = self.vram_preset.currentData() == "16"
        self.rank.setValue(16 if is16 else 32)
        self.alpha.setValue(16 if is16 else 32)
        self.offload.setChecked(is16)
        self.resolution.setCurrentIndex(0 if is16 else 3)

    # ------------------------------------------------------------ specs
    def train_spec(self) -> aitoolkit.TrainSpec:
        spec = aitoolkit.TrainSpec(
            name=self.name.text().strip(), dataset_dir=self.dataset_dir.text().strip(),
            control_dirs=self._control_dirs(), training_folder=str(Path(self.ctx.settings.training_output_dir)),
            name_or_path=self.name_or_path.text().strip() or aitoolkit.DEFAULT_NAME_OR_PATH,
            default_caption=self.default_caption.text().strip(), rank=int(self.rank.value()),
            alpha=int(self.alpha.value()), learning_rate=float(self.lr.value()), steps=int(self.steps.value()),
            resolutions=[int(r) for r in self.resolution.currentData().split(",")],
            timestep_type=self.timestep.currentText(), save_every=int(self.save_every.value()),
            quantize=self.qtype.currentText() != "none", qtype=self.qtype.currentText(),
            quantize_te=self.qtype_te.currentText() != "none", qtype_te=self.qtype_te.currentText(),
            low_vram=self.low_vram.isChecked(), layer_offloading=self.offload.isChecked(),
            layer_offloading_transformer_percent=float(self.offload_pct.value()),
            cache_latents_to_disk=self.cache.isChecked(), cache_text_embeddings=self.cache.isChecked(),
            rgba=self.rgba.isChecked(), sampling=self.sampling.isChecked(),
            sample_prompts=[self.sample_prompt.text().strip()] if self.sample_prompt.text().strip() else [],
        )
        return spec

    def diffsynth_spec(self) -> diffsynth.DiffSynthSpec:
        folder = self.dataset_dir.text().strip()
        edit = bool(self._control_dirs())
        return diffsynth.DiffSynthSpec(
            name=self.name.text().strip(), dataset_dir=folder, edit_mode=edit,
            metadata_path=str(Path(folder) / ("metadata.json" if edit else "metadata.csv")),
            output_path=str(Path(self.ctx.settings.training_output_dir) / self.name.text().strip()),
            learning_rate=float(self.lr.value()), lora_rank=int(self.rank.value()),
            num_epochs=max(1, int(self.steps.value()) // 400), save_steps=int(self.save_every.value()),
        )

    def write_config(self) -> Path | None:
        name = self.name.text().strip() or "qwen21_lora"
        out_dir = Path(self.ctx.settings.training_output_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.trainer.currentData() == "aitoolkit":
            spec = self.train_spec()
            problems = aitoolkit.validate_spec(spec)
            if problems:
                QMessageBox.warning(self, "Config problems", "\n".join(problems))
                return None
            path = aitoolkit.write_config(spec, out_dir / f"{name}.aitoolkit.yaml")
            self.config_view.setPlainText(path.read_text(encoding="utf-8"))
            return path
        spec = self.diffsynth_spec()
        try:
            meta = ds.write_diffsynth_metadata(spec.dataset_dir, self._control_dirs() or None,
                                               self.default_caption.text().strip())
        except ValueError as exc:
            QMessageBox.warning(self, "Dataset problem", str(exc))
            return None
        spec.metadata_path = str(meta)
        cmd = diffsynth.build_command(spec, self.ctx.settings.diffsynth_python_exe(), self.ctx.settings.diffsynth_dir)
        path = out_dir / f"{name}.diffsynth.cmd.txt"
        path.write_text(diffsynth.command_to_shell(cmd), encoding="utf-8")
        self.config_view.setPlainText(f"metadata: {meta}\n\n" + diffsynth.command_to_shell(cmd))
        return path

    # ------------------------------------------------------------ run
    def start(self) -> None:
        if self.runner is not None and self.runner.running:
            return
        path = self.write_config()
        if path is None:
            return
        settings = self.ctx.settings
        if self.trainer.currentData() == "aitoolkit":
            python = settings.ai_toolkit_python_exe()
            if not python.is_file() or not (Path(settings.ai_toolkit_dir) / "run.py").is_file():
                QMessageBox.warning(self, "ai-toolkit missing",
                                    f"Expected {python} and run.py in {settings.ai_toolkit_dir}.\n"
                                    "Run scripts/bootstrap_qwen21.ps1 -WithAiToolkit or set the paths in Settings.")
                return
            cmd = aitoolkit.build_command(python, settings.ai_toolkit_dir, path)
            cwd = Path(settings.ai_toolkit_dir)
        else:
            python = settings.diffsynth_python_exe()
            if not python.is_file():
                QMessageBox.warning(self, "DiffSynth missing", f"Expected {python}. Set the path in Settings.")
                return
            cmd = diffsynth.build_command(self.diffsynth_spec(), python, settings.diffsynth_dir)
            cwd = Path(settings.diffsynth_dir)
        log_path = path.with_suffix(".log")
        self.log.clear()
        self._append_log("$ " + " ".join(cmd))
        self.runner = TrainingRunner(cmd, cwd, on_line=self.bridge.line.emit, on_progress=self.bridge.progress.emit,
                                     on_exit=self.bridge.exited.emit, log_path=log_path)
        try:
            self.runner.start()
        except OSError as exc:
            QMessageBox.critical(self, "Could not start trainer", str(exc))
            self.runner = None
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("training… (log mirrored to " + str(log_path) + ")")
        self.progress.setValue(0)

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()
            self.status.setText("stopping (waiting for the trainer to finish its checkpoint)…")

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_progress(self, prog: Progress) -> None:
        if prog.total:
            self.progress.setValue(int(prog.step * 100 / prog.total))
        bits = [f"step {prog.step}/{prog.total}" if prog.total else ""]
        if prog.loss is not None:
            bits.append(f"loss {prog.loss:.4f}")
        if prog.epoch is not None:
            bits.append(f"epoch {prog.epoch}")
        self.status.setText(" · ".join(b for b in bits if b))

    def _on_exit(self, rc: int) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if rc == 0:
            self.progress.setValue(100)
            spec = self.train_spec()
            out = aitoolkit.expected_output_lora(spec, self.ctx.settings.ai_toolkit_dir)
            self.status.setText(f"finished. LoRA: {out}  (copy it to ComfyUI/models/loras and refresh the Generate tab)")
        else:
            self.status.setText(f"trainer exited with code {rc} — see log")
