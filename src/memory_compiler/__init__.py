"""
MemoryCompiler: A compiler-inspired framework for long conversation memory compression.

This framework treats dialogue memory compression as a compiler optimization problem,
using intermediate representations and optimization passes to achieve efficient compression
while maximizing information retention.
"""

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import MemoryCompiler, CompressionResult

__version__ = "0.1.0"
__all__ = ["MemoryCompiler", "MemoryIR", "CompressionResult"]
