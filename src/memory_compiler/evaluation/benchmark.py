"""Benchmark evaluation script for MemoryCompiler.

This module provides comprehensive benchmarking capabilities for
evaluating memory compression across multiple datasets and configurations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from memory_compiler.evaluation.datasets import DatasetLoader, create_synthetic_dialogue
from memory_compiler.evaluation.metrics import evaluate_compression, EvaluationRunner
from memory_compiler.pipeline import MemoryCompiler, create_compiler


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config_name: str
    dataset_name: str
    num_samples: int
    total_time: float

    # Compression metrics
    mean_compression_ratio: float
    std_compression_ratio: float
    mean_token_reduction: float

    # Retention metrics
    mean_fact_recall: float | None = None
    mean_entity_recall: float | None = None
    mean_qa_accuracy: float | None = None

    # Detailed results
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)

    # Pass statistics
    pass_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_name": self.config_name,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "total_time": self.total_time,
            "compression": {
                "mean_ratio": self.mean_compression_ratio,
                "std_ratio": self.std_compression_ratio,
                "mean_reduction": self.mean_token_reduction,
            },
            "retention": {
                "mean_fact_recall": self.mean_fact_recall,
                "mean_entity_recall": self.mean_entity_recall,
                "mean_qa_accuracy": self.mean_qa_accuracy,
            },
            "pass_stats": self.pass_stats,
        }


class Benchmark:
    """Comprehensive benchmarking for MemoryCompiler.

    Supports:
    - Multiple datasets (MSC, LoCoMo, MultiWOZ, synthetic)
    - Multiple configurations (default, aggressive, conservative)
    - Detailed metrics and timing
    - Result export to JSON

    Example:
        >>> benchmark = Benchmark()
        >>> results = benchmark.run(
        ...     datasets=["synthetic"],
        ...     configs=["default", "aggressive"],
        ...     num_samples=100,
        ... )
        >>> benchmark.save_results("benchmark_results.json")
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.verbose = verbose
        self.results: list[BenchmarkResult] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        datasets: list[str] | None = None,
        configs: list[str] | None = None,
        num_samples: int = 50,
        include_qa: bool = True,
    ) -> list[BenchmarkResult]:
        """Run benchmark evaluation.

        Args:
            datasets: List of dataset names to evaluate.
            configs: List of configuration names to test.
            num_samples: Number of samples per dataset.
            include_qa: Whether to compute QA accuracy.

        Returns:
            List of BenchmarkResult objects.
        """
        datasets = datasets or ["synthetic"]
        configs = configs or ["default", "aggressive", "conservative"]

        logger.info(
            f"Starting benchmark: {len(datasets)} datasets x {len(configs)} configs"
        )

        for config_name in configs:
            for dataset_name in datasets:
                logger.info(f"Evaluating {config_name} on {dataset_name}...")

                result = self._evaluate_config_dataset(
                    config_name=config_name,
                    dataset_name=dataset_name,
                    num_samples=num_samples,
                    include_qa=include_qa,
                )

                self.results.append(result)

                if self.verbose:
                    self._print_result(result)

        return self.results

    def _evaluate_config_dataset(
        self,
        config_name: str,
        dataset_name: str,
        num_samples: int,
        include_qa: bool,
    ) -> BenchmarkResult:
        """Evaluate a single config on a dataset."""
        # Create compiler
        compiler = create_compiler(config_name)

        # Load dataset
        dialogues = self._load_dataset(dataset_name, num_samples)

        # Run evaluation
        start_time = time.time()

        compression_ratios = []
        token_reductions = []
        fact_recalls = []
        entity_recalls = []
        qa_accuracies = []
        per_sample = []
        pass_stats: dict[str, list[dict]] = {}

        for i, dialogue_data in enumerate(dialogues):
            dialogue = dialogue_data["dialogue"]
            qa_pairs = dialogue_data.get("qa_pairs", []) if include_qa else None

            try:
                # Compress
                result = compiler.compress(dialogue)

                # Record metrics
                compression_ratios.append(result.compression_ratio)
                token_reductions.append(1 - result.compression_ratio)

                # Evaluate
                original_text = "\n".join(
                    f"{t['role']}: {t['content']}" for t in dialogue
                )

                metrics = evaluate_compression(
                    result,
                    qa_pairs=qa_pairs,
                    original_text=original_text,
                )

                # Extract retention metrics
                if metrics["retention"]["fact_recall"]:
                    fact_recalls.append(metrics["retention"]["fact_recall"])
                if metrics["retention"]["entity_recall"]:
                    entity_recalls.append(metrics["retention"]["entity_recall"])
                if metrics["retention"]["qa_accuracy"] and qa_pairs:
                    qa_accuracies.append(metrics["retention"]["qa_accuracy"])

                # Collect pass stats
                for pass_name, stats in metrics.get("passes", {}).items():
                    if pass_name not in pass_stats:
                        pass_stats[pass_name] = []
                    pass_stats[pass_name].append(stats)

                per_sample.append({
                    "sample_idx": i,
                    "compression_ratio": result.compression_ratio,
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                })

            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue

        total_time = time.time() - start_time

        # Aggregate pass stats
        aggregated_pass_stats = {}
        for pass_name, stats_list in pass_stats.items():
            aggregated_pass_stats[pass_name] = {
                "mean_facts_removed": np.mean([s.get("facts_removed", 0) for s in stats_list]),
                "mean_entities_removed": np.mean([s.get("entities_removed", 0) for s in stats_list]),
                "mean_events_removed": np.mean([s.get("events_removed", 0) for s in stats_list]),
            }

        return BenchmarkResult(
            config_name=config_name,
            dataset_name=dataset_name,
            num_samples=len(per_sample),
            total_time=total_time,
            mean_compression_ratio=float(np.mean(compression_ratios)) if compression_ratios else 0,
            std_compression_ratio=float(np.std(compression_ratios)) if compression_ratios else 0,
            mean_token_reduction=float(np.mean(token_reductions)) if token_reductions else 0,
            mean_fact_recall=float(np.mean(fact_recalls)) if fact_recalls else None,
            mean_entity_recall=float(np.mean(entity_recalls)) if entity_recalls else None,
            mean_qa_accuracy=float(np.mean(qa_accuracies)) if qa_accuracies else None,
            per_sample_results=per_sample,
            pass_stats=aggregated_pass_stats,
        )

    def _load_dataset(
        self, dataset_name: str, num_samples: int
    ) -> list[dict[str, Any]]:
        """Load dataset samples."""
        if dataset_name == "synthetic":
            return self._generate_synthetic_dataset(num_samples)

        try:
            loader = DatasetLoader(dataset_name)
            loader.load(max_samples=num_samples)

            samples = []
            for sample in loader:
                dialogue = sample.get("dialogue", [])
                qa_pairs = loader.create_qa_pairs(sample)

                samples.append({
                    "dialogue": dialogue,
                    "qa_pairs": qa_pairs,
                })

            return samples

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}, using synthetic: {e}")
            return self._generate_synthetic_dataset(num_samples)

    def _generate_synthetic_dataset(self, num_samples: int) -> list[dict[str, Any]]:
        """Generate synthetic dialogues for testing."""
        samples = []

        for _ in range(num_samples):
            # Vary dialogue length
            num_turns = np.random.randint(6, 30)
            dialogue = create_synthetic_dialogue(num_turns=num_turns)

            # Generate simple QA pairs
            qa_pairs = self._generate_qa_pairs(dialogue)

            samples.append({
                "dialogue": dialogue,
                "qa_pairs": qa_pairs,
            })

        return samples

    def _generate_qa_pairs(
        self, dialogue: list[dict[str, str]]
    ) -> list[tuple[str, str]]:
        """Generate QA pairs from dialogue."""
        import re

        qa_pairs = []

        for turn in dialogue:
            content = turn.get("content", "")

            # Extract name mentions
            name_match = re.search(r"my name is (\w+)", content, re.IGNORECASE)
            if name_match:
                qa_pairs.append(("What is the user's name?", name_match.group(1)))

            # Extract workplace mentions
            work_match = re.search(r"work (?:at|for) ([^.]+)", content, re.IGNORECASE)
            if work_match:
                qa_pairs.append(
                    ("Where does the user work?", work_match.group(1).strip())
                )

        return qa_pairs[:5]  # Limit to 5 QA pairs

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print a benchmark result."""
        print(f"\n{'='*60}")
        print(f"Config: {result.config_name} | Dataset: {result.dataset_name}")
        print(f"{'='*60}")
        print(f"Samples: {result.num_samples} | Time: {result.total_time:.2f}s")
        print(f"Compression Ratio: {result.mean_compression_ratio:.3f} "
              f"(±{result.std_compression_ratio:.3f})")
        print(f"Token Reduction: {result.mean_token_reduction:.1%}")

        if result.mean_fact_recall is not None:
            print(f"Fact Recall: {result.mean_fact_recall:.3f}")
        if result.mean_entity_recall is not None:
            print(f"Entity Recall: {result.mean_entity_recall:.3f}")
        if result.mean_qa_accuracy is not None:
            print(f"QA Accuracy: {result.mean_qa_accuracy:.3f}")

    def save_results(self, filename: str | None = None) -> Path:
        """Save benchmark results to JSON."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        output_path = self.output_dir / filename

        results_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def compare_configs(self) -> dict[str, Any]:
        """Compare results across configurations."""
        config_stats = {}

        for result in self.results:
            if result.config_name not in config_stats:
                config_stats[result.config_name] = {
                    "compression_ratios": [],
                    "token_reductions": [],
                    "fact_recalls": [],
                    "times": [],
                }

            stats = config_stats[result.config_name]
            stats["compression_ratios"].append(result.mean_compression_ratio)
            stats["token_reductions"].append(result.mean_token_reduction)
            stats["times"].append(result.total_time)

            if result.mean_fact_recall is not None:
                stats["fact_recalls"].append(result.mean_fact_recall)

        # Compute aggregates
        comparison = {}
        for config_name, stats in config_stats.items():
            comparison[config_name] = {
                "avg_compression_ratio": np.mean(stats["compression_ratios"]),
                "avg_token_reduction": np.mean(stats["token_reductions"]),
                "avg_fact_recall": np.mean(stats["fact_recalls"]) if stats["fact_recalls"] else None,
                "total_time": sum(stats["times"]),
            }

        return comparison

    def generate_report(self) -> str:
        """Generate a text report of benchmark results."""
        lines = [
            "=" * 70,
            "MEMORYCOMPILER BENCHMARK REPORT",
            "=" * 70,
            "",
        ]

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)

        comparison = self.compare_configs()
        for config_name, stats in comparison.items():
            lines.append(f"\n{config_name}:")
            lines.append(f"  Compression Ratio: {stats['avg_compression_ratio']:.3f}")
            lines.append(f"  Token Reduction:   {stats['avg_token_reduction']:.1%}")
            if stats["avg_fact_recall"]:
                lines.append(f"  Fact Recall:       {stats['avg_fact_recall']:.3f}")
            lines.append(f"  Total Time:        {stats['total_time']:.2f}s")

        # Detailed results
        lines.append("\n" + "=" * 70)
        lines.append("DETAILED RESULTS")
        lines.append("=" * 70)

        for result in self.results:
            lines.append(f"\n{result.config_name} on {result.dataset_name}:")
            lines.append(f"  Samples: {result.num_samples}")
            lines.append(f"  Compression: {result.mean_compression_ratio:.3f} ± {result.std_compression_ratio:.3f}")

            if result.pass_stats:
                lines.append("  Pass Statistics:")
                for pass_name, stats in result.pass_stats.items():
                    lines.append(f"    {pass_name}: {stats}")

        return "\n".join(lines)


def run_quick_benchmark() -> None:
    """Run a quick benchmark for testing."""
    benchmark = Benchmark()

    results = benchmark.run(
        datasets=["synthetic"],
        configs=["default", "aggressive"],
        num_samples=10,
        include_qa=True,
    )

    print("\n" + benchmark.generate_report())
    benchmark.save_results("quick_benchmark.json")


if __name__ == "__main__":
    run_quick_benchmark()
