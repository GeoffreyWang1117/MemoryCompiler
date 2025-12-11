"""Optimization passes for Memory IR compression."""

from memory_compiler.passes.base import OptimizationPass, PassResult, PassManager
from memory_compiler.passes.dead_memory import DeadMemoryEliminationPass
from memory_compiler.passes.fact_folding import FactFoldingPass
from memory_compiler.passes.temporal import TemporalCompressionPass
from memory_compiler.passes.importance import ImportancePruningPass
from memory_compiler.passes.query_prediction import (
    QueryPredictor,
    QueryPredictionPruningPass,
    PredictedQuery,
)

__all__ = [
    "OptimizationPass",
    "PassResult",
    "PassManager",
    "DeadMemoryEliminationPass",
    "FactFoldingPass",
    "TemporalCompressionPass",
    "ImportancePruningPass",
    "QueryPredictor",
    "QueryPredictionPruningPass",
    "PredictedQuery",
]
