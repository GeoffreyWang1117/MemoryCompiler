"""Context window optimization module.

Provides tools for optimally packing compressed memory
into LLM context windows with query-aware selection.
"""

from memory_compiler.context.optimizer import (
    ContextOptimizer,
    OptimizationConfig,
    PackingStrategy,
)
from memory_compiler.context.selector import (
    ContextSelector,
    SelectionCriteria,
)
from memory_compiler.context.packer import (
    ContextPacker,
    PackedContext,
)

__all__ = [
    "ContextOptimizer",
    "OptimizationConfig",
    "PackingStrategy",
    "ContextSelector",
    "SelectionCriteria",
    "ContextPacker",
    "PackedContext",
]
