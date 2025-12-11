"""
MemoryCompiler: A compiler-inspired framework for long conversation memory compression.

This framework treats dialogue memory compression as a compiler optimization problem,
using intermediate representations and optimization passes to achieve efficient compression
while maximizing information retention.

Key Components:
- MemoryIR: Structured intermediate representation of dialogue memory
- MemoryCompiler: Main compression pipeline
- Optimization Passes: Dead memory elimination, fact folding, temporal compression, importance pruning
- Evaluation: Comprehensive metrics for compression quality
"""

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import MemoryCompiler, CompressionResult, create_compiler

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "MemoryCompiler",
    "CompressionResult",
    "create_compiler",
    # IR
    "MemoryIR",
]


def get_version() -> str:
    """Return the package version."""
    return __version__
