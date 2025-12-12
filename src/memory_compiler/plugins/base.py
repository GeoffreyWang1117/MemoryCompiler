"""Base classes for the plugin system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger


class PluginType(Enum):
    """Types of plugins supported."""

    PASS = "pass"  # Optimization pass
    EXTRACTOR = "extractor"  # IR extraction
    GENERATOR = "generator"  # Text generation
    STORAGE = "storage"  # Storage backend
    HOOK = "hook"  # Pipeline hook


@dataclass
class PluginMetadata:
    """Metadata about a plugin.

    Attributes:
        name: Plugin name.
        version: Plugin version string.
        description: Brief description.
        author: Plugin author.
        plugin_type: Type of plugin.
        dependencies: Required dependencies.
        config_schema: Optional JSON schema for configuration.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    plugin_type: PluginType = PluginType.PASS
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema,
        }


class Plugin(ABC):
    """Abstract base class for all plugins.

    All plugins must inherit from this class and implement
    the required abstract methods.

    Example:
        >>> class MyPlugin(Plugin):
        ...     @property
        ...     def metadata(self) -> PluginMetadata:
        ...         return PluginMetadata(name="my_plugin")
        ...
        ...     def initialize(self, config):
        ...         self.config = config
        ...
        ...     def execute(self, data):
        ...         return data  # Transform data
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin configuration dictionary.
        """
        pass

    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute the plugin's main functionality.

        Args:
            data: Input data to process.
            **kwargs: Additional arguments.

        Returns:
            Processed data.
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema.

        Args:
            config: Configuration to validate.

        Returns:
            True if valid, False otherwise.
        """
        schema = self.metadata.config_schema
        if schema is None:
            return True

        # Basic validation - check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in config:
                logger.warning(f"Missing required config field: {field}")
                return False

        return True

    def cleanup(self) -> None:
        """Clean up resources used by the plugin."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.metadata.name} v{self.metadata.version}>"


class PluginRegistry:
    """Registry for managing loaded plugins.

    Maintains a collection of plugins organized by type and provides
    methods for registration, lookup, and iteration.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(my_plugin)
        >>> pass_plugins = registry.get_by_type(PluginType.PASS)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._plugins: Dict[str, Plugin] = {}
        self._by_type: Dict[PluginType, Dict[str, Plugin]] = {
            pt: {} for pt in PluginType
        }

    def register(
        self,
        plugin: Plugin,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a plugin.

        Args:
            plugin: Plugin instance to register.
            config: Optional configuration for initialization.

        Returns:
            True if registration successful.
        """
        name = plugin.metadata.name
        plugin_type = plugin.metadata.plugin_type

        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, replacing")

        # Validate and initialize
        if config and not plugin.validate_config(config):
            logger.error(f"Invalid configuration for plugin '{name}'")
            return False

        try:
            plugin.initialize(config)
        except Exception as e:
            logger.error(f"Failed to initialize plugin '{name}': {e}")
            return False

        self._plugins[name] = plugin
        self._by_type[plugin_type][name] = plugin

        logger.debug(f"Registered plugin: {plugin}")
        return True

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name.

        Args:
            name: Plugin name to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        if name not in self._plugins:
            return False

        plugin = self._plugins[name]
        plugin.cleanup()

        del self._plugins[name]
        del self._by_type[plugin.metadata.plugin_type][name]

        logger.debug(f"Unregistered plugin: {name}")
        return True

    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None.
        """
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all plugins of a specific type.

        Args:
            plugin_type: Type of plugins to retrieve.

        Returns:
            List of plugins of that type.
        """
        return list(self._by_type[plugin_type].values())

    def list_all(self) -> List[PluginMetadata]:
        """List metadata for all registered plugins."""
        return [p.metadata for p in self._plugins.values()]

    def has(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins

    def clear(self) -> None:
        """Unregister all plugins."""
        for name in list(self._plugins.keys()):
            self.unregister(name)

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self):
        return iter(self._plugins.values())


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def register_plugin(
    plugin: Plugin,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Register a plugin with the global registry."""
    return get_registry().register(plugin, config)


def get_plugin(name: str) -> Optional[Plugin]:
    """Get a plugin from the global registry."""
    return get_registry().get(name)
