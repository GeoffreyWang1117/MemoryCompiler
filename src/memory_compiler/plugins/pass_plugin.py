"""Plugin support for custom optimization passes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import OptimizationPass
from memory_compiler.plugins.base import Plugin, PluginMetadata, PluginType


class PassPlugin(Plugin, OptimizationPass):
    """Base class for custom optimization pass plugins.

    Combines Plugin interface with OptimizationPass to create
    pluggable optimization passes.

    Example:
        >>> class MyPassPlugin(PassPlugin):
        ...     @property
        ...     def metadata(self):
        ...         return PluginMetadata(
        ...             name="my_pass",
        ...             plugin_type=PluginType.PASS,
        ...         )
        ...
        ...     def apply(self, ir: MemoryIR) -> MemoryIR:
        ...         # Custom optimization logic
        ...         return ir
    """

    def __init__(self) -> None:
        """Initialize pass plugin."""
        super().__init__()
        self._config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Get pass name from metadata."""
        return self.metadata.name

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the pass with configuration."""
        self._config = config or {}

    def execute(self, data: Any, **kwargs) -> Any:
        """Execute the pass (wrapper for apply)."""
        if isinstance(data, MemoryIR):
            return self.apply(data)
        raise TypeError(f"PassPlugin expects MemoryIR, got {type(data)}")

    @abstractmethod
    def apply(self, ir: MemoryIR) -> MemoryIR:
        """Apply the optimization pass.

        Args:
            ir: Input Memory IR.

        Returns:
            Optimized Memory IR.
        """
        pass


def create_pass_plugin(
    name: str,
    apply_fn: Callable[[MemoryIR, Dict[str, Any]], MemoryIR],
    description: str = "",
    version: str = "1.0.0",
    config_schema: Optional[Dict[str, Any]] = None,
) -> Type[PassPlugin]:
    """Factory function to create a PassPlugin from a function.

    This allows creating plugins without defining a full class.

    Example:
        >>> def my_optimization(ir, config):
        ...     # Custom logic
        ...     return ir
        ...
        >>> MyPass = create_pass_plugin(
        ...     name="my_optimization",
        ...     apply_fn=my_optimization,
        ...     description="My custom optimization",
        ... )
        >>> plugin = MyPass()

    Args:
        name: Pass name.
        apply_fn: Function that takes (MemoryIR, config) and returns MemoryIR.
        description: Pass description.
        version: Pass version.
        config_schema: Optional configuration schema.

    Returns:
        PassPlugin class.
    """

    class FunctionPassPlugin(PassPlugin):
        """Dynamically created pass plugin."""

        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name=name,
                version=version,
                description=description,
                plugin_type=PluginType.PASS,
                config_schema=config_schema,
            )

        def apply(self, ir: MemoryIR) -> MemoryIR:
            return apply_fn(ir, self._config)

    # Set class name for better debugging
    FunctionPassPlugin.__name__ = f"{name}_PassPlugin"
    FunctionPassPlugin.__qualname__ = f"{name}_PassPlugin"

    return FunctionPassPlugin


# Decorator for creating pass plugins
def pass_plugin(
    name: str,
    description: str = "",
    version: str = "1.0.0",
    config_schema: Optional[Dict[str, Any]] = None,
):
    """Decorator to create a PassPlugin from a function.

    Example:
        >>> @pass_plugin("my_custom_pass", description="My pass")
        ... def my_pass(ir: MemoryIR, config: dict) -> MemoryIR:
        ...     # Apply optimization
        ...     return ir
        ...
        >>> # my_pass is now a PassPlugin class
        >>> plugin = my_pass()
        >>> registry.register(plugin)
    """

    def decorator(fn: Callable[[MemoryIR, Dict[str, Any]], MemoryIR]):
        return create_pass_plugin(
            name=name,
            apply_fn=fn,
            description=description,
            version=version,
            config_schema=config_schema,
        )

    return decorator


# Pre-built example pass plugins
@pass_plugin(
    name="filter_low_confidence",
    description="Remove facts with confidence below threshold",
    config_schema={"properties": {"threshold": {"type": "number", "default": 0.5}}},
)
def filter_low_confidence_pass(ir: MemoryIR, config: Dict[str, Any]) -> MemoryIR:
    """Filter out low-confidence facts."""
    threshold = config.get("threshold", 0.5)

    ir.facts = {
        fid: fact
        for fid, fact in ir.facts.items()
        if fact.confidence >= threshold
    }

    return ir


@pass_plugin(
    name="boost_recent",
    description="Boost importance of recent items",
    config_schema={"properties": {"boost_factor": {"type": "number", "default": 1.2}}},
)
def boost_recent_pass(ir: MemoryIR, config: Dict[str, Any]) -> MemoryIR:
    """Boost importance scores of more recent items."""
    boost_factor = config.get("boost_factor", 1.2)

    # Boost entities mentioned recently
    for entity in ir.iter_entities():
        if entity.mentions:
            max_mention = max(entity.mentions)
            # More recent mentions get higher boost
            entity.importance_score = min(1.0, entity.importance_score * boost_factor)

    # Boost recent facts
    max_turn = max((f.source_turn for f in ir.iter_facts() if f.source_turn), default=0)
    for fact in ir.iter_facts():
        if fact.source_turn and max_turn > 0:
            recency = fact.source_turn / max_turn
            fact.importance_score = min(1.0, fact.importance_score * (1 + (boost_factor - 1) * recency))

    return ir


@pass_plugin(
    name="entity_centric",
    description="Keep only facts related to important entities",
    config_schema={"properties": {"top_k_entities": {"type": "integer", "default": 10}}},
)
def entity_centric_pass(ir: MemoryIR, config: Dict[str, Any]) -> MemoryIR:
    """Keep facts related to top-K most important entities."""
    top_k = config.get("top_k_entities", 10)

    # Get top entities
    entities = sorted(
        ir.iter_entities(),
        key=lambda e: e.importance_score,
        reverse=True,
    )[:top_k]

    important_names = set()
    for entity in entities:
        important_names.add(entity.name.lower())
        important_names.update(a.lower() for a in entity.aliases)

    # Filter facts
    ir.facts = {
        fid: fact
        for fid, fact in ir.facts.items()
        if (
            fact.subject.lower() in important_names
            or fact.object.lower() in important_names
        )
    }

    return ir


class PassPipeline:
    """Pipeline for chaining multiple pass plugins.

    Example:
        >>> pipeline = PassPipeline()
        >>> pipeline.add_pass(pass1)
        >>> pipeline.add_pass(pass2)
        >>> result = pipeline.execute(ir)
    """

    def __init__(self, passes: Optional[List[PassPlugin]] = None) -> None:
        """Initialize pipeline.

        Args:
            passes: Optional initial list of passes.
        """
        self._passes: List[PassPlugin] = passes or []

    def add_pass(self, pass_plugin: PassPlugin) -> "PassPipeline":
        """Add a pass to the pipeline.

        Args:
            pass_plugin: Pass to add.

        Returns:
            Self for chaining.
        """
        self._passes.append(pass_plugin)
        return self

    def remove_pass(self, name: str) -> bool:
        """Remove a pass by name.

        Args:
            name: Name of pass to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, p in enumerate(self._passes):
            if p.name == name:
                self._passes.pop(i)
                return True
        return False

    def execute(self, ir: MemoryIR) -> MemoryIR:
        """Execute all passes in sequence.

        Args:
            ir: Input Memory IR.

        Returns:
            Memory IR after all passes.
        """
        result = ir

        for pass_plugin in self._passes:
            logger.debug(f"Executing pass: {pass_plugin.name}")
            result = pass_plugin.apply(result)

        return result

    def __len__(self) -> int:
        return len(self._passes)

    def __iter__(self):
        return iter(self._passes)
