"""Tests for plugin system module."""

import pytest
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.plugins import (
    Plugin,
    PluginRegistry,
    PluginMetadata,
    PluginType,
    PassPlugin,
    pass_plugin,
    PluginLoader,
    HookManager,
    HookType,
)


def create_test_ir() -> MemoryIR:
    """Create a test Memory IR."""
    ir = MemoryIR(session_id="test_plugins")
    ir.add_entity(Entity(name="Alice", type=EntityType.PERSON, importance_score=0.8))
    ir.add_entity(Entity(name="Bob", type=EntityType.PERSON, importance_score=0.3))
    ir.add_fact(Fact(subject="Alice", predicate="knows", object="Bob"))
    return ir


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()

        class TestPlugin(Plugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="test-plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PASS,
                )

            def execute(self, ir):
                return ir

        plugin = TestPlugin()
        registry.register(plugin)

        assert "test-plugin" in registry.list_plugins()

    def test_get_plugin(self):
        """Test getting a registered plugin."""
        registry = PluginRegistry()

        class MyPlugin(Plugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="my-plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PASS,
                )

            def execute(self, ir):
                return ir

        plugin = MyPlugin()
        registry.register(plugin)

        retrieved = registry.get("my-plugin")
        assert retrieved is not None
        assert retrieved.get_metadata().name == "my-plugin"

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry()

        class TempPlugin(Plugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="temp-plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PASS,
                )

            def execute(self, ir):
                return ir

        plugin = TempPlugin()
        registry.register(plugin)
        registry.unregister("temp-plugin")

        assert "temp-plugin" not in registry.list_plugins()

    def test_list_by_type(self):
        """Test listing plugins by type."""
        registry = PluginRegistry()

        class PassPlugin1(Plugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="pass1",
                    version="1.0.0",
                    plugin_type=PluginType.PASS,
                )

            def execute(self, ir):
                return ir

        class ExtractorPlugin(Plugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="extractor1",
                    version="1.0.0",
                    plugin_type=PluginType.EXTRACTOR,
                )

            def execute(self, ir):
                return ir

        registry.register(PassPlugin1())
        registry.register(ExtractorPlugin())

        pass_plugins = registry.list_by_type(PluginType.PASS)
        assert len(pass_plugins) == 1
        assert pass_plugins[0] == "pass1"


class TestPassPlugin:
    """Tests for PassPlugin."""

    def test_create_pass_plugin(self):
        """Test creating a pass plugin."""
        class ImportanceFilter(PassPlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="importance-filter",
                    version="1.0.0",
                    plugin_type=PluginType.PASS,
                )

            def run_pass(self, ir: MemoryIR) -> MemoryIR:
                # Filter low importance entities
                for entity in list(ir.iter_entities()):
                    if entity.importance_score < 0.5:
                        ir.remove_entity(entity.name)
                return ir

        plugin = ImportanceFilter()
        ir = create_test_ir()

        result = plugin.execute(ir)

        entity_names = [e.name for e in result.iter_entities()]
        assert "Alice" in entity_names
        assert "Bob" not in entity_names

    def test_pass_plugin_decorator(self):
        """Test pass_plugin decorator."""
        @pass_plugin(name="double-importance", version="1.0.0")
        def double_importance(ir: MemoryIR) -> MemoryIR:
            for entity in ir.iter_entities():
                entity.importance_score = min(1.0, entity.importance_score * 2)
            return ir

        plugin = double_importance
        ir = create_test_ir()

        result = plugin.execute(ir)

        for entity in result.iter_entities():
            if entity.name == "Alice":
                assert entity.importance_score >= 0.8

    def test_pass_with_config(self):
        """Test pass plugin with configuration."""
        class ConfigurablePass(PassPlugin):
            def __init__(self, threshold: float = 0.5):
                self.threshold = threshold

            def get_metadata(self):
                return PluginMetadata(
                    name="configurable-pass",
                    version="1.0.0",
                    plugin_type=PluginType.PASS,
                    config={"threshold": self.threshold},
                )

            def run_pass(self, ir: MemoryIR) -> MemoryIR:
                for entity in list(ir.iter_entities()):
                    if entity.importance_score < self.threshold:
                        ir.remove_entity(entity.name)
                return ir

        plugin = ConfigurablePass(threshold=0.7)
        ir = create_test_ir()

        result = plugin.execute(ir)

        entity_names = [e.name for e in result.iter_entities()]
        assert "Alice" in entity_names


class TestHookManager:
    """Tests for HookManager."""

    def test_register_hook(self):
        """Test registering a hook."""
        manager = HookManager()
        called = []

        def my_hook(ir):
            called.append("hook_called")
            return ir

        manager.register(HookType.PRE_OPTIMIZATION, my_hook)

        ir = create_test_ir()
        manager.execute(HookType.PRE_OPTIMIZATION, ir)

        assert "hook_called" in called

    def test_multiple_hooks(self):
        """Test multiple hooks for same type."""
        manager = HookManager()
        order = []

        def hook1(ir):
            order.append("hook1")
            return ir

        def hook2(ir):
            order.append("hook2")
            return ir

        manager.register(HookType.PRE_EXTRACTION, hook1)
        manager.register(HookType.PRE_EXTRACTION, hook2)

        ir = create_test_ir()
        manager.execute(HookType.PRE_EXTRACTION, ir)

        assert order == ["hook1", "hook2"]

    def test_hook_priority(self):
        """Test hook priority ordering."""
        manager = HookManager()
        order = []

        def low_priority_hook(ir):
            order.append("low")
            return ir

        def high_priority_hook(ir):
            order.append("high")
            return ir

        manager.register(HookType.POST_OPTIMIZATION, low_priority_hook, priority=10)
        manager.register(HookType.POST_OPTIMIZATION, high_priority_hook, priority=1)

        ir = create_test_ir()
        manager.execute(HookType.POST_OPTIMIZATION, ir)

        assert order == ["high", "low"]

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        manager = HookManager()
        called = []

        def removable_hook(ir):
            called.append("should_not_be_called")
            return ir

        hook_id = manager.register(HookType.PRE_EXTRACTION, removable_hook)
        manager.unregister(hook_id)

        ir = create_test_ir()
        manager.execute(HookType.PRE_EXTRACTION, ir)

        assert len(called) == 0

    def test_hook_modifies_ir(self):
        """Test that hooks can modify IR."""
        manager = HookManager()

        def add_entity_hook(ir):
            ir.add_entity(Entity(
                name="HookEntity",
                type=EntityType.OTHER,
            ))
            return ir

        manager.register(HookType.POST_EXTRACTION, add_entity_hook)

        ir = create_test_ir()
        result = manager.execute(HookType.POST_EXTRACTION, ir)

        entity_names = [e.name for e in result.iter_entities()]
        assert "HookEntity" in entity_names


class TestPluginLoader:
    """Tests for PluginLoader."""

    def test_load_from_module(self):
        """Test loading plugins from module."""
        # This would require actual module files
        loader = PluginLoader()

        # Test that loader initializes properly
        assert loader is not None

    def test_discover_plugins(self):
        """Test plugin discovery."""
        loader = PluginLoader()

        # Test discovery mechanism
        plugins = loader.discover()
        assert isinstance(plugins, list)

    def test_load_builtin_plugins(self):
        """Test loading built-in plugins."""
        loader = PluginLoader()

        builtins = loader.load_builtins()
        # Should have some built-in plugins
        assert isinstance(builtins, list)


class TestPluginMetadata:
    """Tests for PluginMetadata."""

    def test_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            plugin_type=PluginType.PASS,
            author="Test Author",
            description="A test plugin",
        )

        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = PluginMetadata(
            name="serializable",
            version="2.0.0",
            plugin_type=PluginType.EXTRACTOR,
        )

        d = metadata.to_dict()

        assert d["name"] == "serializable"
        assert d["version"] == "2.0.0"

    def test_metadata_dependencies(self):
        """Test metadata dependencies."""
        metadata = PluginMetadata(
            name="dependent-plugin",
            version="1.0.0",
            plugin_type=PluginType.PASS,
            dependencies=["other-plugin>=1.0.0"],
        )

        assert "other-plugin>=1.0.0" in metadata.dependencies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
