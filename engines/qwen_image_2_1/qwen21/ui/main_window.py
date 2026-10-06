"""Main window + shared application context (settings, ComfyUI client, model lists, status bridge)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget, QWidget

from ..comfy_client import ComfyClient, ComfyError
from ..settings import ENGINE_DIR, Settings
from ..web_status import StatusState, WebStatusServer


class AppContext(QObject):
    models_changed = Signal()
    send_to_edit = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.client = ComfyClient(settings.comfy_url)
        self.object_info: dict[str, Any] = {}
        self.models: dict[str, list[str]] = {}
        self.samplers: list[str] = []
        self.schedulers: list[str] = []
        self.connected = False
        self.last_outputs: list[str] = []
        self.scratch_dir = ENGINE_DIR / ".scratch"
        self.scratch_dir.mkdir(exist_ok=True)
        self.web = WebStatusServer(StatusState(Path(settings.output_dir), self.status_payload),
                                   port=settings.web_status_port)
        self._statusbar: QLabel | None = None

    # ---------------------------------------------------------------- server
    def apply_settings(self) -> None:
        self.client = ComfyClient(self.settings.comfy_url)
        self.web.state.output_dir = Path(self.settings.output_dir)
        if self.settings.web_status_enabled and not self.web.running:
            self.web.port = self.settings.web_status_port
            self.web.start()
        elif not self.settings.web_status_enabled and self.web.running:
            self.web.stop()

    def check_connection(self) -> bool:
        was = self.connected
        self.connected = self.client.is_up()
        if self.connected and (not was or not self.object_info):
            self.refresh_models()
        return self.connected

    def refresh_models(self) -> None:
        try:
            self.object_info = self.client.object_info()
            self.models = {folder: self.client.list_models(folder)
                           for folder in ("diffusion_models", "text_encoders", "vae", "loras", "model_patches")}
            ks = self.object_info.get("KSampler", {}).get("input", {}).get("required", {})
            self.samplers = list(ks.get("sampler_name", [[]])[0]) if ks else []
            self.schedulers = list(ks.get("scheduler", [[]])[0]) if ks else []
            self.connected = True
        except ComfyError:
            self.connected = False
        self.models_changed.emit()

    def has_node(self, class_type: str) -> bool:
        return class_type in self.object_info if self.object_info else True

    def ensure_connected(self, parent: QWidget | None) -> bool:
        if self.check_connection():
            return True
        QMessageBox.warning(parent, "ComfyUI offline",
                            f"No ComfyUI at {self.settings.comfy_url}.\nStart it (Settings → Launch ComfyUI) and retry.")
        return False

    # ---------------------------------------------------------------- misc
    def record_output(self, paths: list[Path]) -> None:
        self.last_outputs = [str(p) for p in paths][-10:] + self.last_outputs[:40]

    def status_payload(self) -> dict[str, Any]:
        return {"comfy": self.settings.comfy_url, "comfy_connected": self.connected,
                "last_output": self.last_outputs[0] if self.last_outputs else None, "time": int(time.time())}

    def statusbar_message(self, text: str) -> None:
        if self._statusbar is not None:
            self._statusbar.setText(text)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or Settings.load()
        self.ctx = AppContext(self.settings)
        self.setWindowTitle("Qwen Image 2.1 Studio — AIWF (ComfyUI backend)")
        self.resize(1480, 940)

        from .generate_tab import GenerationTab
        from .settings_tab import SettingsTab
        from .train_tab import TrainTab

        self.tabs = QTabWidget()
        self.generate_tab = GenerationTab(self.ctx, "t2i")
        self.edit_tab = GenerationTab(self.ctx, "edit")
        self.train_tab = TrainTab(self.ctx)
        self.settings_tab = SettingsTab(self.ctx)
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.addTab(self.edit_tab, "Edit (references, RGBA, ControlNet)")
        self.tabs.addTab(self.train_tab, "Train LoRA")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.setCentralWidget(self.tabs)

        self.conn_label = QLabel("checking ComfyUI…")
        self.msg_label = QLabel("")
        self.statusBar().addWidget(self.conn_label)
        self.statusBar().addPermanentWidget(self.msg_label)
        self.ctx._statusbar = self.msg_label
        self.ctx.send_to_edit.connect(self._send_to_edit)

        self.ctx.apply_settings()
        self.timer = QTimer(self)
        self.timer.setInterval(10_000)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        QTimer.singleShot(200, self._poll)

    def _poll(self) -> None:
        up = self.ctx.check_connection()
        if up:
            try:
                version = self.ctx.client.comfyui_version()
            except ComfyError:
                version = "?"
            self.conn_label.setText(f"● ComfyUI {version} at {self.settings.comfy_url}")
            self.conn_label.setStyleSheet("color:#43d17a;")
        else:
            self.conn_label.setText(f"○ ComfyUI offline ({self.settings.comfy_url})")
            self.conn_label.setStyleSheet("color:#e06c75;")

    def _send_to_edit(self, path: str) -> None:
        self.edit_tab.load_reference(path)
        self.tabs.setCurrentWidget(self.edit_tab)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.ctx.web.stop()
        super().closeEvent(event)
