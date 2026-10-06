"""Qwen Image 2.1 Studio — desktop app (PySide6) on top of a running ComfyUI.

    python engines/qwen_image_2_1/app.py [--comfy-url http://127.0.0.1:8188] [--no-web-status]

Generate, edit with up to 10 reference images, paint marks for local edits,
RGBA/transparent output, LoRA stacking, few-step accelerators, Fun ControlNet,
prompt enhancer, and a Train tab that drives ai-toolkit / DiffSynth-Studio.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen Image 2.1 Studio (ComfyUI backend)")
    parser.add_argument("--comfy-url", default=None, help="ComfyUI base URL (default from settings, 127.0.0.1:8188)")
    parser.add_argument("--no-web-status", action="store_true", help="do not serve the Pro-tab status page on :7865")
    parser.add_argument("--check", action="store_true", help="import the UI, print status and exit (no window)")
    args = parser.parse_args(argv)

    from qwen21.settings import Settings

    settings = Settings.load()
    if args.comfy_url:
        settings.comfy_url = args.comfy_url
    if args.no_web_status:
        settings.web_status_enabled = False

    if args.check:
        from qwen21.comfy_client import ComfyClient

        client = ComfyClient(settings.comfy_url)
        print("ComfyUI:", client.server_summary() if client.is_up() else f"offline at {settings.comfy_url}")
        try:
            import PySide6

            print("PySide6", PySide6.__version__, "OK")
        except ImportError:
            print("PySide6 missing: pip install -r engines/qwen_image_2_1/requirements.txt")
            return 1
        return 0

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    from qwen21.ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Qwen Image 2.1 Studio")
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(32, 34, 37))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(225, 228, 232))
    palette.setColor(QPalette.ColorRole.Base, QColor(24, 26, 28))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 42, 46))
    palette.setColor(QPalette.ColorRole.Text, QColor(225, 228, 232))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 48, 52))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(225, 228, 232))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(72, 128, 220))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 52, 56))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(225, 228, 232))
    app.setPalette(palette)
    window = MainWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
