from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class PluginPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PluginInfo:
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    priority: PluginPriority = PluginPriority.MEDIUM
    tags: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    plugin_id: str
    risk_score: float
    is_suspicious: bool
    details: dict = field(default_factory=dict)


class BaseDetector(ABC):
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        ...

    @abstractmethod
    def analyze(
        self,
        score_data: dict,
        replay_data: dict | None = None,
        user_history: dict | None = None,
    ) -> list[DetectionResult]:
        ...


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}

    def register_plugin(self, info: PluginInfo) -> None:
        self._plugins[info.plugin_id] = info

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        return dict(self._plugins)

    def get(self, plugin_id: str) -> PluginInfo | None:
        return self._plugins.get(plugin_id)


class PluginLifecycle:
    def __init__(self) -> None:
        self._started = False

    def mark_started(self) -> None:
        self._started = True

    @property
    def is_started(self) -> bool:
        return self._started


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def subscribe(self, event_name: str):
        def decorator(func):
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(func)
            return func

        return decorator

    async def publish(self, event_name: str, data: dict) -> None:
        handlers = self._handlers.get(event_name, [])
        for handler in handlers:
            await handler(data)

    @property
    def handlers(self) -> dict[str, list]:
        return dict(self._handlers)
