from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from aiwf.core.interfaces.plugins import PluginInfo

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

logger = logging.getLogger(__name__)
TabFactory = Callable[["AppContext"], None]


def _read_manifest(plugin_root: Path) -> dict[str, Any]:
    manifest_path = plugin_root / "plugin.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", manifest_path, exc)
        return {}


@dataclass
class PluginRegistry:
    """Extension host: user folders in plugins/ hook into the app at boot.

    An extension is a folder containing ``plugin.py`` (and optionally
    ``plugin.json`` metadata). At load, the module's ``plugin.on_load(ctx)``
    or module-level ``setup(ctx)`` runs with the full AppContext, and may:

    - ``ctx.plugins.register_api(plugin_id, router)`` — a FastAPI APIRouter
      served under ``/api/ext/<plugin-id>/`` in AIWF Studio Pro.
    - ``ctx.plugins.register_tab(name, factory)`` — a Gradio Lab tab factory.
    - ``ctx.events.subscribe(EventType, handler)`` — react to app events.
    - use any service on ``ctx`` (generation, models, wan, segment, ...).

    Folders whose names start with ``_`` or ``.`` are skipped. Extensions in
    ``settings.disabled_extensions`` are listed but not imported.
    """

    tabs: list[tuple[str, TabFactory, int]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    plugins: list[PluginInfo] = field(default_factory=list)
    api_routers: list[tuple[str, Any]] = field(default_factory=list)

    def register_tab(self, name: str, factory: TabFactory, order: int = 100) -> None:
        self.tabs.append((name, factory, order))
        self.tabs.sort(key=lambda item: item[2])

    def register_api(self, plugin_id: str, router: Any) -> None:
        """Register a FastAPI router served under /api/ext/<plugin_id>/."""
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(plugin_id)).strip("-")
        if not safe_id:
            raise ValueError("register_api needs a non-empty plugin id")
        self.api_routers.append((safe_id, router))

    def discover(self, plugins_dir: Path, ctx: AppContext) -> None:
        if not plugins_dir.exists():
            return

        disabled = {
            str(item).strip().lower()
            for item in (getattr(getattr(ctx, "settings", None), "disabled_extensions", None) or [])
        }

        for plugin_root in sorted(p for p in plugins_dir.iterdir() if p.is_dir()):
            if plugin_root.name.startswith(("_", ".")):
                continue
            plugin_file = plugin_root / "plugin.py"
            if not plugin_file.exists():
                continue
            manifest = _read_manifest(plugin_root)

            if plugin_root.name.lower() in disabled:
                self.plugins.append(
                    PluginInfo(
                        id=plugin_root.name,
                        name=str(manifest.get("name") or plugin_root.name),
                        version=str(manifest.get("version") or "0.0.0"),
                        description=str(manifest.get("description") or ""),
                        path=str(plugin_root),
                        enabled=False,
                    )
                )
                logger.info("Extension disabled in settings, not loaded: %s", plugin_root.name)
                continue

            module_name = f"aiwf_plugin_{plugin_root.name}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                plugin = getattr(module, "plugin", None)
                if plugin is not None and hasattr(plugin, "on_load"):
                    plugin.on_load(ctx)
                    info = PluginInfo(
                        id=plugin_root.name,
                        name=str(manifest.get("name") or getattr(plugin, "name", plugin_root.name)),
                        version=str(manifest.get("version") or getattr(plugin, "version", "0.0.0")),
                        description=str(manifest.get("description") or getattr(plugin, "description", "")),
                        path=str(plugin_root),
                    )
                elif hasattr(module, "setup"):
                    module.setup(ctx)
                    info = PluginInfo(
                        id=plugin_root.name,
                        name=str(manifest.get("name") or plugin_root.name),
                        version=str(manifest.get("version") or "0.0.0"),
                        description=str(manifest.get("description") or ""),
                        path=str(plugin_root),
                    )
                else:
                    continue
                self.loaded.append(plugin_root.name)
                self.plugins.append(info)
                logger.info("Loaded extension: %s", plugin_root.name)
            except Exception as exc:
                self.plugins.append(
                    PluginInfo(
                        id=plugin_root.name,
                        name=str(manifest.get("name") or plugin_root.name),
                        version=str(manifest.get("version") or "0.0.0"),
                        description=str(manifest.get("description") or ""),
                        path=str(plugin_root),
                        enabled=False,
                        error=str(exc),
                    )
                )
                logger.exception("Failed to load extension %s", plugin_root.name)

    def list_plugins(self) -> list[PluginInfo]:
        return list(self.plugins)
