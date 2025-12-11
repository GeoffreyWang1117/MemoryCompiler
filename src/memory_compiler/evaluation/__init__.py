"""Evaluation module for Memory Compiler."""

from memory_compiler.evaluation.metrics import (
    CompressionMetrics,
    InformationRetentionMetrics,
    evaluate_compression,
)
from memory_compiler.evaluation.datasets import DatasetLoader

__all__ = [
    "CompressionMetrics",
    "InformationRetentionMetrics",
    "evaluate_compression",
    "DatasetLoader",
]
