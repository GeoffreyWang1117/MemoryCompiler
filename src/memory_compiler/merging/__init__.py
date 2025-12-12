"""Memory merging module for consolidating multiple sessions.

Provides tools to merge multiple Memory IRs into a unified representation,
handling entity resolution, fact deduplication, and timeline alignment.
"""

from memory_compiler.merging.merger import (
    MemoryMerger,
    MergeStrategy,
    MergeConfig,
    MergeResult,
)
from memory_compiler.merging.entity_resolver import (
    EntityResolver,
    EntityMatch,
    ResolutionStrategy,
)
from memory_compiler.merging.conflict import (
    ConflictResolver,
    ConflictType,
    Conflict,
)

__all__ = [
    "MemoryMerger",
    "MergeStrategy",
    "MergeConfig",
    "MergeResult",
    "EntityResolver",
    "EntityMatch",
    "ResolutionStrategy",
    "ConflictResolver",
    "ConflictType",
    "Conflict",
]
