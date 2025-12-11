"""Optimization passes for Memory IR compression."""

from memory_compiler.passes.base import OptimizationPass, PassResult
from memory_compiler.passes.dead_memory import DeadMemoryEliminationPass
from memory_compiler.passes.fact_folding import FactFoldingPass
from memory_compiler.passes.temporal import TemporalCompressionPass
from memory_compiler.passes.importance import ImportancePruningPass

__all__ = [
    "OptimizationPass",
    "PassResult",
    "DeadMemoryEliminationPass",
    "FactFoldingPass",
    "TemporalCompressionPass",
    "ImportancePruningPass",
]
