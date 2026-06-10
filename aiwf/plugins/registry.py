from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from aiwf.core.interfaces.plugins import PluginInfo

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

logger = logging.getLogger(__name__)
TabFactory = Callable[["AppContext"], None]


@dataclass
class PluginRegistry:
    tabs: list[tuple[str, TabFactory, int]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    plugins: list[PluginInfo] = field(default_factory=list)

    def register_tab(self, name: str, factory: TabFactory, order: int = 100) -> None:
        self.tabs.append((name, factory, order))
        self.tabs.sort(key=lambda item: item[2])

    def discover(self, plugins_dir: Path, ctx: AppContext) -> None:
        if not plugins_dir.exists():
            return

        for plugin_root in sorted(p for p in plugins_dir.iterdir() if p.is_dir()):
            plugin_file = plugin_root / "plugin.py"
            if not plugin_file.exists():
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
                        name=getattr(plugin, "name", plugin_root.name),
                        version=getattr(plugin, "version", "0.0.0"),
                        path=str(plugin_root),
                    )
                elif hasattr(module, "setup"):
                    module.setup(ctx)
                    info = PluginInfo(id=plugin_root.name, name=plugin_root.name, path=str(plugin_root))
                else:
                    continue
                self.loaded.append(plugin_root.name)
                self.plugins.append(info)
                logger.info("Loaded plugin: %s", plugin_root.name)
            except Exception as exc:
                self.plugins.append(
                    PluginInfo(
                        id=plugin_root.name,
                        name=plugin_root.name,
                        path=str(plugin_root),
                        enabled=False,
                        error=str(exc),
                    )
                )
                logger.exception("Failed to load plugin %s", plugin_root.name)

    def list_plugins(self) -> list[PluginInfo]:
        return list(self.plugins)
