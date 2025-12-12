"""Memory versioning and diffing module.

Provides tools for tracking changes to Memory IR over time,
computing diffs between versions, and maintaining version history.
"""

from memory_compiler.versioning.diff import (
    MemoryDiff,
    DiffOperation,
    compute_diff,
    apply_diff,
)
from memory_compiler.versioning.history import (
    MemoryVersion,
    VersionHistory,
    VersionedMemoryStore,
)

__all__ = [
    "MemoryDiff",
    "DiffOperation",
    "compute_diff",
    "apply_diff",
    "MemoryVersion",
    "VersionHistory",
    "VersionedMemoryStore",
]
