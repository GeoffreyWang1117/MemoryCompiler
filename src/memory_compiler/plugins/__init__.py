"""Plugin system for extending MemoryCompiler.

Provides a flexible plugin architecture for:
- Custom optimization passes
- Custom extractors
- Custom generators
- Hooks for pipeline stages
"""

from memory_compiler.plugins.base import (
    Plugin,
    PluginType,
    PluginMetadata,
    PluginRegistry,
)
from memory_compiler.plugins.pass_plugin import (
    PassPlugin,
    create_pass_plugin,
)
from memory_compiler.plugins.loader import (
    PluginLoader,
    load_plugins_from_directory,
)
from memory_compiler.plugins.hooks import (
    HookType,
    HookManager,
    hook,
)

__all__ = [
    "Plugin",
    "PluginType",
    "PluginMetadata",
    "PluginRegistry",
    "PassPlugin",
    "create_pass_plugin",
    "PluginLoader",
    "load_plugins_from_directory",
    "HookType",
    "HookManager",
    "hook",
]
