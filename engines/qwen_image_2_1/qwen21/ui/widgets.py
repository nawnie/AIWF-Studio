"""Shared Qt widgets: image view with alpha checkerboard, annotation canvas, reference slots,
model / LoRA / sampler panels, and the background job worker."""
from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QPushButton, QSizePolicy, QSpinBox, QTableWidget, QVBoxLayout, QWidget)

from .. import presets as P
from ..comfy_client import ComfyClient, ComfyError, RunResult
from ..workflow_builder import LoraEntry

SAMPLERS_FALLBACK = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "res_multistep", "seeds_2",
                     "lcm", "uni_pc"]
SCHEDULERS_FALLBACK = ["simple", "normal", "sgm_uniform", "beta", "karras", "exponential", "linear_quadratic", "kl_optimal"]


def checkerboard_brush(size: int = 12) -> QBrush:
    tile = QPixmap(size * 2, size * 2)
    tile.fill(QColor(70, 70, 70))
    painter = QPainter(tile)
    painter.fillRect(0, 0, size, size, QColor(100, 100, 100))
    painter.fillRect(size, size, size, size, QColor(100, 100, 100))
    painter.end()
    return QBrush(tile)


def fit_rect(image_size: QSize, area: QRect) -> QRect:
    if image_size.isEmpty() or area.isEmpty():
        return QRect()
    scaled = image_size.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
    x = area.x() + (area.width() - scaled.width()) // 2
    y = area.y() + (area.height() - scaled.height()) // 2
    return QRect(x, y, scaled.width(), scaled.height())


class ImageView(QWidget):
    """Shows one image scaled to fit, alpha over a checkerboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._brush = checkerboard_brush()
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.caption = ""

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_bytes(self, data: bytes) -> None:
        self.set_image(QImage.fromData(data))

    def set_path(self, path: str | Path) -> None:
        self.set_image(QImage(str(path)))

    def clear(self) -> None:
        self.set_image(QImage())

    def image(self) -> QImage:
        return self._image

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 26, 28))
        if self._image.isNull():
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.caption or "No image")
            return
        target = fit_rect(self._image.size(), self.rect().adjusted(4, 4, -4, -4))
        painter.fillRect(target, self._brush)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)


class AnnotationCanvas(QWidget):
    """Paint circles/strokes on the image being edited (Qwen 2.1 reads painted marks),
    and export either the flattened annotated image or a white-on-black mask."""

    changed = Signal()
    COLORS = {"red": QColor(255, 32, 32), "green": QColor(32, 220, 32), "blue": QColor(48, 96, 255),
              "yellow": QColor(255, 220, 0), "white": QColor(255, 255, 255)}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._overlay = QImage()
        self._brush = checkerboard_brush()
        self._last: QPoint | None = None
        self.color_name = "red"
        self.brush_size = 24
        self.has_strokes = False
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # -- state ---------------------------------------------------------------
    def load(self, path: str | Path | None) -> None:
        self._image = QImage(str(path)) if path else QImage()
        self._overlay = QImage(self._image.size(), QImage.Format.Format_ARGB32) if not self._image.isNull() else QImage()
        if not self._overlay.isNull():
            self._overlay.fill(Qt.GlobalColor.transparent)
        self.has_strokes = False
        self.update()
        self.changed.emit()

    def clear_strokes(self) -> None:
        if not self._overlay.isNull():
            self._overlay.fill(Qt.GlobalColor.transparent)
        self.has_strokes = False
        self.update()
        self.changed.emit()

    def export_annotated(self, path: str | Path) -> Path:
        """Original image with the strokes burned in (what the model sees as image_1)."""
        out = self._image.convertToFormat(QImage.Format.Format_ARGB32)
        painter = QPainter(out)
        painter.drawImage(0, 0, self._overlay)
        painter.end()
        out.save(str(path), "PNG")
        return Path(path)

    def export_mask(self, path: str | Path) -> Path:
        """Mask PNG: white where painted, black elsewhere (used with the Fun ControlNet inpaint input)."""
        mask = QImage(self._image.size(), QImage.Format.Format_RGB32)
        mask.fill(Qt.GlobalColor.black)
        painter = QPainter(mask)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        white = QImage(self._overlay.size(), QImage.Format.Format_ARGB32)
        white.fill(Qt.GlobalColor.white)
        white.setAlphaChannel(self._overlay.convertToFormat(QImage.Format.Format_Alpha8))
        painter.drawImage(0, 0, white)
        painter.end()
        mask.save(str(path), "PNG")
        return Path(path)

    # -- painting ------------------------------------------------------------
    def _target_rect(self) -> QRect:
        return fit_rect(self._image.size(), self.rect().adjusted(4, 4, -4, -4))

    def _to_image(self, pos: QPoint) -> QPoint | None:
        target = self._target_rect()
        if target.isEmpty() or not target.contains(pos):
            return None
        sx = self._image.width() / target.width()
        sy = self._image.height() / target.height()
        return QPoint(int((pos.x() - target.x()) * sx), int((pos.y() - target.y()) * sy))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._image.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return
        self._last = self._to_image(event.position().toPoint())
        if self._last is not None:
            self._draw(self._last, self._last)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._last is None:
            return
        point = self._to_image(event.position().toPoint())
        if point is not None:
            self._draw(self._last, point)
            self._last = point

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        self._last = None

    def _draw(self, a: QPoint, b: QPoint) -> None:
        scale = self._image.width() / max(1, self._target_rect().width())
        painter = QPainter(self._overlay)
        pen = QPen(self.COLORS[self.color_name], max(1.0, self.brush_size * scale), Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(a, b)
        painter.end()
        self.has_strokes = True
        self.update()
        self.changed.emit()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 26, 28))
        if self._image.isNull():
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "image_1 appears here.\nPaint a circle or area to tell the model where to edit.")
            return
        target = self._target_rect()
        painter.fillRect(target, self._brush)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)
        painter.drawImage(target, self._overlay)


class ReferenceSlots(QGroupBox):
    """Up to 10 reference images. image_1 is the picture being edited."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None, max_refs: int = 10) -> None:
        super().__init__(f"Reference images (image_1 … image_{max_refs})", parent)
        self.max_refs = max_refs
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(96, 96))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setMinimumHeight(130)
        self.list.setAcceptDrops(True)
        self.list.model().rowsMoved.connect(lambda *_: self._renumber())
        add = QPushButton("Add…")
        remove = QPushButton("Remove")
        clear = QPushButton("Clear")
        add.clicked.connect(self.add_files)
        remove.clicked.connect(self.remove_selected)
        clear.clicked.connect(self.clear)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        self.hint = QLabel("Drag to reorder. Refer to them in the prompt as <image1>, <image2>, …")
        self.hint.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addWidget(self.hint)

    def paths(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Reference images", "", "Images (*.png *.jpg *.jpeg *.webp)")
        for f in files:
            self.add_path(f)

    def add_path(self, path: str) -> None:
        if self.list.count() >= self.max_refs:
            return
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        pix = QPixmap(path)
        if not pix.isNull():
            item.setIcon(pix.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.list.addItem(item)
        self._renumber()

    def remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self._renumber()

    def clear(self) -> None:
        self.list.clear()
        self._renumber()

    def _renumber(self) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setText(f"image_{i + 1}  {Path(item.data(Qt.ItemDataRole.UserRole)).name}")
        self.changed.emit()


class ModelPanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Model files (ComfyUI models folder)", parent)
        self.dit = QComboBox()
        self.te = QComboBox()
        self.vae = QComboBox()
        self.te_device = QComboBox()
        self.te_device.addItems(["default", "cpu"])
        self.te_device.setToolTip("cpu keeps the text encoder in system RAM. Slower encode, more VRAM for 2K / many references.")
        for combo in (self.dit, self.te, self.vae):
            combo.setEditable(True)
        self.recommend = QPushButton("Use 16 GB recommended")
        self.recommend.clicked.connect(self.apply_recommended)
        self.refresh = QPushButton("Refresh from server")
        form = QFormLayout(self)
        form.addRow("Diffusion model", self.dit)
        form.addRow("Text encoder", self.te)
        form.addRow("Text encoder device", self.te_device)
        form.addRow("VAE", self.vae)
        row = QHBoxLayout()
        row.addWidget(self.recommend)
        row.addWidget(self.refresh)
        form.addRow(row)
        self.apply_recommended()

    def set_lists(self, dits: list[str], tes: list[str], vaes: list[str]) -> None:
        for combo, items in ((self.dit, dits), (self.te, tes), (self.vae, vaes)):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if current:
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setEditText(current)
            combo.blockSignals(False)

    def apply_recommended(self) -> None:
        for combo, key in ((self.dit, "dit"), (self.te, "text_encoder"), (self.vae, "vae")):
            value = P.RECOMMENDED_16GB[key]
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.setEditText(value)
        self.te_device.setCurrentText("default")

    def values(self) -> tuple[str, str, str, str]:
        return (self.dit.currentText().strip(), self.te.currentText().strip(), self.vae.currentText().strip(),
                self.te_device.currentText())


class LoraStack(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("LoRA stack (LoraLoaderModelOnly, in order)", parent)
        self.files: list[str] = []
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["LoRA file", "Strength"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 320)
        self.table.setMinimumHeight(90)
        add = QPushButton("Add LoRA")
        remove = QPushButton("Remove selected")
        add.clicked.connect(lambda: self.add_row())
        remove.clicked.connect(self.remove_selected)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def set_files(self, files: list[str]) -> None:
        self.files = list(files)
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            if isinstance(combo, QComboBox):
                current = combo.currentText()
                combo.clear()
                combo.addItems(self.files)
                combo.setEditText(current)

    def add_row(self, name: str = "", strength: float = 1.0) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self.files)
        if name:
            combo.setEditText(name)
        spin = QDoubleSpinBox()
        spin.setRange(-2.0, 3.0)
        spin.setSingleStep(0.05)
        spin.setValue(strength)
        self.table.setCellWidget(row, 0, combo)
        self.table.setCellWidget(row, 1, spin)

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for row in rows:
            self.table.removeRow(row)

    def entries(self) -> list[LoraEntry]:
        out = []
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            spin = self.table.cellWidget(row, 1)
            name = combo.currentText().strip() if isinstance(combo, QComboBox) else ""
            if name:
                out.append(LoraEntry(name, float(spin.value()) if isinstance(spin, QDoubleSpinBox) else 1.0))
        return out


class SamplerPanel(QGroupBox):
    """Steps / CFG / seed / sampler / scheduler + presets, accelerator LoRA, KV cache, guidance stack."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Sampling", parent)
        self.preset = QComboBox()
        for key, preset in P.SAMPLER_PRESETS.items():
            self.preset.addItem(preset.label, key)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.steps = QSpinBox()
        self.steps.setRange(1, 150)
        self.steps.setValue(25)
        self.cfg = QDoubleSpinBox()
        self.cfg.setRange(1.0, 20.0)
        self.cfg.setSingleStep(0.5)
        self.cfg.setValue(1.0)
        self.cfg.setToolTip("1.0 = official default (negative prompt ignored). 2 for dense prompts/typography. 5+ degrades.")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.random_seed = QCheckBox("random")
        self.random_seed.setChecked(True)
        self.sampler = QComboBox()
        self.sampler.setEditable(True)
        self.sampler.addItems(SAMPLERS_FALLBACK)
        self.scheduler = QComboBox()
        self.scheduler.setEditable(True)
        self.scheduler.addItems(SCHEDULERS_FALLBACK)
        self.accelerator = QComboBox()
        for key, acc in P.ACCELERATORS.items():
            self.accelerator.addItem(acc.label, key)
        self.accelerator.currentIndexChanged.connect(self._accelerator_changed)
        self.accel_file = QComboBox()
        self.accel_file.setEditable(True)
        self.accel_file.setToolTip("LoRA filename under models/loras for the selected accelerator.")
        self.accel_note = QLabel("")
        self.accel_note.setWordWrap(True)
        self.accel_note.setStyleSheet("color: #9aa5b1;")
        self.kv_device = QComboBox()
        self.kv_device.addItems(["auto", "gpu", "cpu", "off"])
        self.kv_dtype = QComboBox()
        self.kv_dtype.addItems(["default", "int8", "int4"])
        self.kv_device.setToolTip("Qwen Image 2.1 Cache: where the prefix KV cache lives. cpu costs little speed and frees VRAM for edits.")
        self.fix_guidance = QCheckBox("APG + FreSca guidance stack (Fix-LoRA recipe)")
        self.negative_preset = QComboBox()
        for label, text in P.NEGATIVE_PRESETS.items():
            self.negative_preset.addItem(label, text)
        self.negative = QPlainTextEdit()
        self.negative.setPlaceholderText("Negative prompt (only used when CFG > 1)")
        self.negative.setMaximumHeight(56)
        self.negative_preset.currentIndexChanged.connect(
            lambda: self.negative.setPlainText(self.negative_preset.currentData() or ""))

        form = QFormLayout(self)
        form.addRow("Preset", self.preset)
        row = QHBoxLayout()
        row.addWidget(QLabel("Steps"))
        row.addWidget(self.steps)
        row.addWidget(QLabel("CFG"))
        row.addWidget(self.cfg)
        row.addWidget(QLabel("Seed"))
        row.addWidget(self.seed)
        row.addWidget(self.random_seed)
        form.addRow(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Sampler"))
        row2.addWidget(self.sampler)
        row2.addWidget(QLabel("Scheduler"))
        row2.addWidget(self.scheduler)
        form.addRow(row2)
        form.addRow("Accelerator", self.accelerator)
        form.addRow("Accelerator LoRA file", self.accel_file)
        form.addRow(self.accel_note)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("KV cache device"))
        row3.addWidget(self.kv_device)
        row3.addWidget(QLabel("dtype"))
        row3.addWidget(self.kv_dtype)
        form.addRow(row3)
        form.addRow(self.fix_guidance)
        form.addRow("Negative preset", self.negative_preset)
        form.addRow(self.negative)
        self._accelerator_changed()

    def set_server_options(self, samplers: list[str], schedulers: list[str], lora_files: list[str]) -> None:
        for combo, items in ((self.sampler, samplers), (self.scheduler, schedulers)):
            if items:
                current = combo.currentText()
                combo.clear()
                combo.addItems(items)
                combo.setEditText(current)
        current = self.accel_file.currentText()
        self.accel_file.clear()
        self.accel_file.addItems(lora_files)
        self.accel_file.setEditText(current)

    def _apply_preset(self) -> None:
        preset = P.SAMPLER_PRESETS[self.preset.currentData()]
        self.steps.setValue(preset.steps)
        self.cfg.setValue(preset.cfg)
        self.sampler.setEditText(preset.sampler)
        self.scheduler.setEditText(preset.scheduler)
        self.negative.setPlainText(preset.negative)
        self.fix_guidance.setChecked(preset.fix_guidance)

    def _accelerator_changed(self) -> None:
        acc = P.ACCELERATORS[self.accelerator.currentData()]
        if acc.key == "none":
            self.accel_note.setText("Full-step base model. 25 steps ≈ 20 s per 1024² image on a 16 GB card with int8 files.")
            self.accel_file.setEnabled(False)
            return
        self.accel_file.setEnabled(True)
        self.accel_file.setEditText(acc.filename.split("/")[-1])
        self.steps.setValue(acc.steps)
        self.cfg.setValue(acc.cfg)
        self.sampler.setEditText(acc.sampler)
        self.scheduler.setEditText(acc.scheduler)
        mode = "ManualSigmas + SamplerCustom" if acc.mode == "sigmas" else "KSampler"
        self.accel_note.setText(f"[{acc.status}] {acc.note}\n{mode} · {acc.size_gb:.2f} GB · {acc.download_url}")

    def accelerator_key(self) -> str:
        return self.accelerator.currentData()

    def next_seed(self) -> int:
        if self.random_seed.isChecked():
            self.seed.setValue(random.randint(0, 2**31 - 1))
        return int(self.seed.value())


class JobWorker(QThread):
    """Uploads inputs, runs the prompt on ComfyUI, downloads outputs."""

    progress = Signal(int, int, str)
    status = Signal(str)
    preview = Signal(bytes)
    finished_ok = Signal(object)  # RunResult
    failed = Signal(str)

    def __init__(self, client: ComfyClient, build: Callable[[dict[str, str]], dict[str, Any]],
                 uploads: dict[str, str], output_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.build = build  # receives {local_path: uploaded_name} -> API prompt
        self.uploads = uploads  # key -> local path
        self.output_dir = output_dir
        self.cancel_event = threading.Event()
        self.saved: list[Path] = []

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            names: dict[str, str] = {}
            for key, local in self.uploads.items():
                self.status.emit(f"uploading {Path(local).name}")
                names[key] = self.client.upload_image(local)
            prompt = self.build(names)
            result: RunResult = self.client.run(
                prompt,
                on_progress=lambda v, m, n: self.progress.emit(v, m, n),
                on_status=lambda s: self.status.emit(s),
                on_preview=lambda b: self.preview.emit(b),
                cancel=self.cancel_event,
            )
            if result.error:
                self.failed.emit(result.error)
                return
            for img in result.images:
                if img.type != "output":
                    continue
                self.status.emit(f"downloading {img.filename}")
                self.saved.append(self.client.download_output(img, self.output_dir))
            self.finished_ok.emit(result)
        except ComfyError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - UI safety net
            self.failed.emit(f"{type(exc).__name__}: {exc}")
