"""Evaluation benchmarks for Memory Compiler.

This module provides benchmark evaluation against standard datasets:
- MSC (Multi-Session Chat)
- LoCoMo (Long Context Memory)
- LongBench
- MultiWOZ
"""

from memory_compiler.benchmarks.base import (
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkRunner,
    EvaluationMetrics,
)
from memory_compiler.benchmarks.msc import MSCBenchmark
from memory_compiler.benchmarks.locomo import LoCoMoBenchmark
from memory_compiler.benchmarks.longbench import LongBenchBenchmark

__all__ = [
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkRunner",
    "EvaluationMetrics",
    "MSCBenchmark",
    "LoCoMoBenchmark",
    "LongBenchBenchmark",
]
