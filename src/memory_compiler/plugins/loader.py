"""Plugin loader for discovering and loading plugins."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from memory_compiler.plugins.base import (
    Plugin,
    PluginMetadata,
    PluginRegistry,
    PluginType,
    get_registry,
)


class PluginLoadError(Exception):
    """Error loading a plugin."""

    pass


class PluginLoader:
    """Load plugins from various sources.

    Supports loading from:
    - Python modules
    - Directory of plugin files
    - Entry points
    - Plugin packages

    Example:
        >>> loader = PluginLoader()
        >>> loader.load_from_module("my_plugins.custom_pass")
        >>> loader.load_from_directory("./plugins")
    """

    def __init__(
        self,
        registry: Optional[PluginRegistry] = None,
        auto_register: bool = True,
    ) -> None:
        """Initialize loader.

        Args:
            registry: Plugin registry to use.
            auto_register: Whether to auto-register loaded plugins.
        """
        self.registry = registry or get_registry()
        self.auto_register = auto_register
        self._loaded_modules: Dict[str, Any] = {}

    def load_from_module(
        self,
        module_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Plugin]:
        """Load plugins from a Python module.

        Args:
            module_name: Fully qualified module name.
            config: Optional configuration for plugins.

        Returns:
            List of loaded plugins.
        """
        try:
            module = importlib.import_module(module_name)
            self._loaded_modules[module_name] = module
        except ImportError as e:
            raise PluginLoadError(f"Failed to import module '{module_name}': {e}")

        return self._discover_plugins_in_module(module, config)

    def load_from_file(
        self,
        file_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Plugin]:
        """Load plugins from a Python file.

        Args:
            file_path: Path to Python file.
            config: Optional configuration.

        Returns:
            List of loaded plugins.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise PluginLoadError(f"Plugin file not found: {file_path}")

        if not file_path.suffix == ".py":
            raise PluginLoadError(f"Plugin file must be .py: {file_path}")

        # Generate module name from file path
        module_name = f"memory_compiler_plugins.{file_path.stem}"

        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Failed to create module spec for: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise PluginLoadError(f"Error executing plugin module: {e}")

        self._loaded_modules[module_name] = module
        return self._discover_plugins_in_module(module, config)

    def load_from_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Plugin]:
        """Load all plugins from a directory.

        Args:
            directory: Directory containing plugin files.
            recursive: Whether to search subdirectories.
            config: Optional configuration.

        Returns:
            List of all loaded plugins.
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise PluginLoadError(f"Plugin directory not found: {directory}")

        all_plugins: List[Plugin] = []

        if recursive:
            pattern = "**/*.py"
        else:
            pattern = "*.py"

        for file_path in directory.glob(pattern):
            if file_path.name.startswith("_"):
                continue  # Skip __init__.py, etc.

            try:
                plugins = self.load_from_file(file_path, config)
                all_plugins.extend(plugins)
                logger.debug(f"Loaded {len(plugins)} plugins from {file_path}")
            except PluginLoadError as e:
                logger.warning(f"Failed to load plugins from {file_path}: {e}")

        return all_plugins

    def _discover_plugins_in_module(
        self,
        module: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Plugin]:
        """Discover and instantiate plugins from a module."""
        plugins: List[Plugin] = []

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # Check if it's a Plugin class (not the base class)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
                and hasattr(attr, "metadata")
            ):
                try:
                    # Instantiate the plugin
                    plugin = attr()

                    if self.auto_register:
                        self.registry.register(plugin, config)

                    plugins.append(plugin)
                    logger.debug(f"Discovered plugin: {plugin.metadata.name}")

                except Exception as e:
                    logger.warning(f"Failed to instantiate plugin {attr_name}: {e}")

        return plugins

    def load_from_entry_points(
        self,
        group: str = "memory_compiler.plugins",
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Plugin]:
        """Load plugins from package entry points.

        Args:
            group: Entry point group name.
            config: Optional configuration.

        Returns:
            List of loaded plugins.
        """
        plugins: List[Plugin] = []

        try:
            # Python 3.10+
            from importlib.metadata import entry_points

            eps = entry_points(group=group)
        except ImportError:
            # Fallback for older Python
            try:
                from importlib_metadata import entry_points
                eps = entry_points(group=group)
            except ImportError:
                logger.warning("Entry points not available")
                return plugins

        for ep in eps:
            try:
                plugin_class = ep.load()

                if issubclass(plugin_class, Plugin):
                    plugin = plugin_class()

                    if self.auto_register:
                        self.registry.register(plugin, config)

                    plugins.append(plugin)
                    logger.debug(f"Loaded plugin from entry point: {ep.name}")

            except Exception as e:
                logger.warning(f"Failed to load entry point {ep.name}: {e}")

        return plugins

    def unload_module(self, module_name: str) -> bool:
        """Unload a previously loaded module.

        Args:
            module_name: Module name to unload.

        Returns:
            True if unloaded.
        """
        if module_name in self._loaded_modules:
            del self._loaded_modules[module_name]

            if module_name in sys.modules:
                del sys.modules[module_name]

            return True

        return False


def load_plugins_from_directory(
    directory: str | Path,
    registry: Optional[PluginRegistry] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Plugin]:
    """Convenience function to load plugins from a directory.

    Args:
        directory: Plugin directory.
        registry: Optional registry.
        config: Optional configuration.

    Returns:
        List of loaded plugins.
    """
    loader = PluginLoader(registry=registry)
    return loader.load_from_directory(directory, config=config)


# Template for creating plugin files
PLUGIN_TEMPLATE = '''"""Custom plugin for MemoryCompiler."""

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.plugins.base import Plugin, PluginMetadata, PluginType
from memory_compiler.plugins.pass_plugin import PassPlugin


class {class_name}(PassPlugin):
    """Custom optimization pass plugin.

    {description}
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="{version}",
            description="{description}",
            plugin_type=PluginType.PASS,
        )

    def apply(self, ir: MemoryIR) -> MemoryIR:
        """Apply the optimization.

        Args:
            ir: Input Memory IR.

        Returns:
            Optimized Memory IR.
        """
        # TODO: Implement your optimization logic here

        return ir
'''


def generate_plugin_template(
    name: str,
    description: str = "Custom optimization pass",
    version: str = "1.0.0",
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate a plugin template file.

    Args:
        name: Plugin name.
        description: Plugin description.
        version: Plugin version.
        output_path: Optional path to write file.

    Returns:
        Plugin template string.
    """
    # Convert name to class name
    class_name = "".join(word.capitalize() for word in name.split("_")) + "Plugin"

    content = PLUGIN_TEMPLATE.format(
        class_name=class_name,
        name=name,
        version=version,
        description=description,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(content)
        logger.info(f"Generated plugin template at: {output_path}")

    return content
