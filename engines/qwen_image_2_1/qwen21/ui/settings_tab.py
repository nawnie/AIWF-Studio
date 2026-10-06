"""Settings tab: ComfyUI connection/launch, model availability, trainer paths, Pro-tab bridge."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from .. import presets as P
from ..comfy_client import ComfyError


class SettingsTab(QWidget):
    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        s = ctx.settings
        inner = QWidget()
        layout = QVBoxLayout(inner)

        comfy = QGroupBox("ComfyUI")
        cform = QFormLayout(comfy)
        self.comfy_url = QLineEdit(s.comfy_url)
        cform.addRow("Server URL", self.comfy_url)
        test = QPushButton("Test connection")
        test.clicked.connect(self.test_connection)
        cform.addRow(test)
        self.launch_cmd = QLineEdit(s.comfy_launch_command)
        self.launch_cmd.setPlaceholderText(r"F:\ComfyUI\venv\Scripts\python.exe main.py --port 8188 --disable-auto-launch")
        cform.addRow("Launch command", self.launch_cmd)
        self.launch_cwd = QLineEdit(s.comfy_launch_cwd)
        self.launch_cwd.setPlaceholderText(r"F:\ComfyUI")
        cform.addRow("Launch working dir", self.launch_cwd)
        launch = QPushButton("Launch ComfyUI (UTF-8 env, new console)")
        launch.clicked.connect(self.launch_comfy)
        cform.addRow(launch)
        self.models_dir = QLineEdit(s.comfy_models_dir)
        self.models_dir.setPlaceholderText(r"F:\ComfyUI\models  (optional, for on-disk file checks)")
        cform.addRow("ComfyUI models folder", self.models_dir)
        self.output_dir = QLineEdit(s.output_dir)
        cform.addRow("Download outputs to", self.output_dir)
        layout.addWidget(comfy)

        files = QGroupBox("Model files for Qwen-Image 2.1")
        flayout = QVBoxLayout(files)
        check = QPushButton("Check which files the server sees")
        check.clicked.connect(self.check_files)
        flayout.addWidget(check)
        self.files_view = QPlainTextEdit()
        self.files_view.setReadOnly(True)
        self.files_view.setMinimumHeight(220)
        self.files_view.setPlainText(self._file_table())
        flayout.addWidget(self.files_view)
        quant = QLabel(P.QUANT_GUIDANCE_16GB)
        quant.setWordWrap(True)
        quant.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color:#c8d0d8;")
        flayout.addWidget(quant)
        layout.addWidget(files)

        pe = QGroupBox("Prompt enhancer / ControlNet files")
        pform = QFormLayout(pe)
        self.pe_t2i = QLineEdit(s.pe_t2i)
        self.pe_i2i = QLineEdit(s.pe_i2i)
        self.cn_patch = QLineEdit(s.controlnet_patch)
        pform.addRow("PE text-to-image (text_encoders/)", self.pe_t2i)
        pform.addRow("PE image-edit (text_encoders/)", self.pe_i2i)
        pform.addRow("Fun ControlNet (model_patches/)", self.cn_patch)
        layout.addWidget(pe)

        train = QGroupBox("Trainers")
        tform = QFormLayout(train)
        self.ai_dir = QLineEdit(s.ai_toolkit_dir)
        self.ai_py = QLineEdit(s.ai_toolkit_python)
        self.ai_py.setPlaceholderText("default: <ai-toolkit>/venv/Scripts/python.exe")
        self.ds_dir = QLineEdit(s.diffsynth_dir)
        self.ds_py = QLineEdit(s.diffsynth_python)
        self.ds_py.setPlaceholderText("default: <DiffSynth-Studio>/venv/Scripts/python.exe")
        self.train_out = QLineEdit(s.training_output_dir)
        tform.addRow("ai-toolkit folder", self._dir_row(self.ai_dir))
        tform.addRow("ai-toolkit python", self.ai_py)
        tform.addRow("DiffSynth-Studio folder", self._dir_row(self.ds_dir))
        tform.addRow("DiffSynth python", self.ds_py)
        tform.addRow("Training output folder", self.train_out)
        layout.addWidget(train)

        bridge = QGroupBox("AIWF Studio Pro bridge")
        bform = QFormLayout(bridge)
        self.web_enabled = QCheckBox("Serve /api/health + output gallery for the Pro 'Qwen Image Editor' tab")
        self.web_enabled.setChecked(s.web_status_enabled)
        self.web_port = QSpinBox()
        self.web_port.setRange(1024, 65535)
        self.web_port.setValue(s.web_status_port)
        bform.addRow(self.web_enabled)
        bform.addRow("Port", self.web_port)
        layout.addWidget(bridge)

        save = QPushButton("Save settings")
        save.setMinimumHeight(34)
        save.clicked.connect(self.save)
        layout.addWidget(save)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _dir_row(self, line: QLineEdit) -> QHBoxLayout:
        btn = QPushButton("…")

        def pick() -> None:
            path = QFileDialog.getExistingDirectory(self, "Folder", line.text() or str(Path.home()))
            if path:
                line.setText(path)

        btn.clicked.connect(pick)
        row = QHBoxLayout()
        row.addWidget(line)
        row.addWidget(btn)
        return row

    @staticmethod
    def _file_table(present: set[str] | None = None) -> str:
        lines = [f"{'file':<66} {'folder':<16} {'GB':>6}  16GB  status"]
        for f in P.ALL_MODEL_FILES:
            status = "" if present is None else ("present" if f.filename in present else "MISSING")
            lines.append(f"{f.filename:<66} {f.folder:<16} {f.size_gb:>6.2f}  {'*' if f.recommended_16gb else ' '}     {status}")
        lines.append("")
        lines.append(f"Download from https://huggingface.co/{P.COMFY_ORG_REPO}/tree/main into the ComfyUI models folders above.")
        return "\n".join(lines)

    def save(self) -> None:
        s = self.ctx.settings
        s.comfy_url = self.comfy_url.text().strip() or s.comfy_url
        s.comfy_launch_command = self.launch_cmd.text().strip()
        s.comfy_launch_cwd = self.launch_cwd.text().strip()
        s.comfy_models_dir = self.models_dir.text().strip()
        s.output_dir = self.output_dir.text().strip() or s.output_dir
        s.pe_t2i, s.pe_i2i, s.controlnet_patch = self.pe_t2i.text().strip(), self.pe_i2i.text().strip(), self.cn_patch.text().strip()
        s.ai_toolkit_dir, s.ai_toolkit_python = self.ai_dir.text().strip(), self.ai_py.text().strip()
        s.diffsynth_dir, s.diffsynth_python = self.ds_dir.text().strip(), self.ds_py.text().strip()
        s.training_output_dir = self.train_out.text().strip() or s.training_output_dir
        s.web_status_enabled, s.web_status_port = self.web_enabled.isChecked(), int(self.web_port.value())
        s.save()
        self.ctx.apply_settings()
        self.ctx.statusbar_message("settings saved")

    def test_connection(self) -> None:
        self.ctx.settings.comfy_url = self.comfy_url.text().strip()
        self.ctx.apply_settings()
        try:
            summary = self.ctx.client.server_summary()
        except ComfyError as exc:
            QMessageBox.warning(self, "ComfyUI", str(exc))
            return
        version = self.ctx.client.comfyui_version()
        missing = [n for n in ("TextEncodeQwenImage21", "QwenImage21Cache", "ZImageFunControlnet", "ManualSigmas",
                               "SaveImageAdvanced", "TextGenerate") if not self.ctx.has_node(n)]
        text = summary + (f"\n\nMissing nodes: {', '.join(missing)} (update ComfyUI; 2.1 needs >= {P.MIN_COMFYUI_VERSION})"
                          if missing else f"\n\nAll Qwen 2.1 nodes present (ComfyUI {version}).")
        QMessageBox.information(self, "ComfyUI", text)
        self.ctx.refresh_models()

    def check_files(self) -> None:
        if not self.ctx.ensure_connected(self):
            return
        present: set[str] = set()
        for folder in ("diffusion_models", "text_encoders", "vae", "model_patches"):
            present.update(Path(n).name for n in self.ctx.client.list_models(folder))
        self.files_view.setPlainText(self._file_table(present))

    def launch_comfy(self) -> None:
        cmd, cwd = self.launch_cmd.text().strip(), self.launch_cwd.text().strip()
        if not cmd:
            QMessageBox.information(self, "Launch ComfyUI", "Fill in the launch command first "
                                    "(your F: install's python.exe + main.py, not the bundled desktop app).")
            return
        args = shlex.split(cmd, posix=False) if "\\" in cmd else shlex.split(cmd)
        try:
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen(args, cwd=cwd or None, env=self.ctx.settings.comfy_launch_env(), creationflags=flags)
        except OSError as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))
            return
        self.ctx.statusbar_message("ComfyUI launching with PYTHONUTF8=1; connect in ~30-60 s")
