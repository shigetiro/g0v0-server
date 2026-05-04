import importlib
import json
from logging import getLogger
from pathlib import Path

from app.plugins.base import EventBus, PluginLifecycle, PluginRegistry

logger = getLogger("plugin.manager")


class PluginManager:
    def __init__(self, plugins_dir: str = "plugins") -> None:
        self.plugins_dir = Path(plugins_dir)
        self.registry = PluginRegistry()
        self.lifecycle = PluginLifecycle()
        self.event_bus = EventBus()
        self._loaded_plugins: dict[str, object] = {}

    async def load_all(self) -> None:
        if not self.plugins_dir.exists():
            logger.info("Plugins directory not found: %s", self.plugins_dir)
            return

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                continue

            try:
                await self._load_plugin(plugin_dir, plugin_json)
            except Exception:
                logger.exception("Failed to load plugin: %s", plugin_dir.name)

        logger.info("Loaded %d plugins", len(self._loaded_plugins))

    async def _load_plugin(self, plugin_dir: Path, plugin_json: Path) -> None:
        with open(plugin_json) as f:
            metadata = json.load(f)

        plugin_id = metadata.get("id", plugin_dir.name)
        logger.info("Loading plugin: %s", plugin_id)

        module = importlib.import_module(f"plugins.{plugin_dir.name}")

        plugin_instance = getattr(module, "plugin", None)
        if not plugin_instance:
            logger.warning("No 'plugin' instance found in %s", plugin_dir.name)
            return

        if hasattr(plugin_instance, "register"):
            await plugin_instance.register(self.registry)

        if hasattr(plugin_instance, "start"):
            await plugin_instance.start(self.lifecycle, self.event_bus)

        self._loaded_plugins[plugin_id] = plugin_instance
        self.lifecycle.mark_started()

    async def shutdown_all(self) -> None:
        for plugin_id, plugin in self._loaded_plugins.items():
            try:
                if hasattr(plugin, "stop"):
                    await plugin.stop(self.lifecycle)
                logger.info("Stopped plugin: %s", plugin_id)
            except Exception:
                logger.exception("Failed to stop plugin: %s", plugin_id)

    @property
    def loaded_plugins(self) -> dict[str, object]:
        return dict(self._loaded_plugins)


plugin_manager = PluginManager()
