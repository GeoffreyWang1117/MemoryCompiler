"""Base classes for benchmark evaluation."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import MemoryCompiler


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating memory compression quality.

    Attributes:
        compression_ratio: Ratio of original to compressed size.
        information_retention: How much key information is preserved (0-1).
        entity_recall: Fraction of entities retained.
        fact_recall: Fraction of facts retained.
        rouge_l: ROUGE-L score against original.
        bleu: BLEU score for generation quality.
        latency_ms: Processing latency in milliseconds.
        tokens_saved: Number of tokens saved.
    """

    compression_ratio: float = 0.0
    information_retention: float = 0.0
    entity_recall: float = 0.0
    fact_recall: float = 0.0
    rouge_l: float = 0.0
    bleu: float = 0.0
    latency_ms: float = 0.0
    tokens_saved: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "compression_ratio": self.compression_ratio,
            "information_retention": self.information_retention,
            "entity_recall": self.entity_recall,
            "fact_recall": self.fact_recall,
            "rouge_l": self.rouge_l,
            "bleu": self.bleu,
            "latency_ms": self.latency_ms,
            "tokens_saved": self.tokens_saved,
        }


@dataclass
class BenchmarkSample:
    """A single benchmark sample.

    Attributes:
        id: Sample identifier.
        dialogue: Input dialogue text.
        reference_summary: Optional reference compressed summary.
        key_entities: Key entities that should be preserved.
        key_facts: Key facts that should be preserved.
        questions: Optional QA pairs for evaluation.
        metadata: Additional sample metadata.
    """

    id: str
    dialogue: str
    reference_summary: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    questions: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of evaluating on a benchmark.

    Attributes:
        benchmark_name: Name of the benchmark.
        num_samples: Number of samples evaluated.
        metrics: Aggregated metrics.
        per_sample_metrics: Metrics for each sample.
        timestamp: When evaluation was run.
        config: Configuration used.
    """

    benchmark_name: str
    num_samples: int
    metrics: EvaluationMetrics
    per_sample_metrics: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics.to_dict(),
            "per_sample_metrics": self.per_sample_metrics,
            "timestamp": self.timestamp,
            "config": self.config,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark results to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResult":
        """Load results from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            benchmark_name=data["benchmark_name"],
            num_samples=data["num_samples"],
            metrics=EvaluationMetrics(**data["metrics"]),
            per_sample_metrics=data.get("per_sample_metrics", []),
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
        )


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        """Initialize benchmark dataset.

        Args:
            data_dir: Directory containing dataset files.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._samples: List[BenchmarkSample] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Get benchmark name."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load dataset from disk or download."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Iterate over samples."""
        pass

    def get_sample(self, idx: int) -> BenchmarkSample:
        """Get sample by index."""
        return self._samples[idx]


class MetricsCalculator:
    """Calculate evaluation metrics for compression quality."""

    def __init__(self) -> None:
        self._rouge_scorer = None
        self._bleu_scorer = None

    def _get_rouge_scorer(self):
        """Lazy load ROUGE scorer."""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rougeL"], use_stemmer=True
                )
            except ImportError:
                logger.warning("rouge_score not installed, ROUGE metrics unavailable")
        return self._rouge_scorer

    def calculate_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L score."""
        scorer = self._get_rouge_scorer()
        if scorer is None:
            return 0.0

        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())

            smoothing = SmoothingFunction().method1
            return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        except ImportError:
            logger.warning("NLTK not installed, BLEU metrics unavailable")
            return 0.0

    def calculate_entity_recall(
        self,
        key_entities: List[str],
        ir: MemoryIR,
    ) -> float:
        """Calculate entity recall."""
        if not key_entities:
            return 1.0

        extracted_names = set()
        for entity in ir.iter_entities():
            extracted_names.add(entity.name.lower())
            for alias in entity.aliases:
                extracted_names.add(alias.lower())

        found = sum(
            1 for e in key_entities if e.lower() in extracted_names
        )
        return found / len(key_entities)

    def calculate_fact_recall(
        self,
        key_facts: List[str],
        ir: MemoryIR,
    ) -> float:
        """Calculate fact recall by semantic matching."""
        if not key_facts:
            return 1.0

        # Build fact strings from IR
        ir_facts = []
        for fact in ir.iter_facts():
            ir_facts.append(f"{fact.subject} {fact.predicate} {fact.object}".lower())

        # Check how many key facts are covered
        found = 0
        for key_fact in key_facts:
            key_fact_lower = key_fact.lower()
            # Simple substring matching
            for ir_fact in ir_facts:
                if self._fuzzy_match(key_fact_lower, ir_fact):
                    found += 1
                    break

        return found / len(key_facts)

    def _fuzzy_match(self, a: str, b: str, threshold: float = 0.6) -> bool:
        """Check if two strings match with some fuzziness."""
        # Simple word overlap
        a_words = set(a.split())
        b_words = set(b.split())

        if not a_words or not b_words:
            return False

        overlap = len(a_words & b_words)
        return overlap / len(a_words) >= threshold

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    def calculate_compression_ratio(
        self,
        original: str,
        compressed: str,
    ) -> float:
        """Calculate compression ratio."""
        original_tokens = self.estimate_tokens(original)
        compressed_tokens = self.estimate_tokens(compressed)

        if compressed_tokens == 0:
            return float("inf")

        return original_tokens / compressed_tokens


class BenchmarkRunner:
    """Run benchmarks and collect results."""

    def __init__(
        self,
        compiler: Optional[MemoryCompiler] = None,
        token_budget: int = 2048,
        output_style: str = "narrative",
    ) -> None:
        """Initialize benchmark runner.

        Args:
            compiler: MemoryCompiler instance to evaluate.
            token_budget: Token budget for compression.
            output_style: Style for generated output.
        """
        self.compiler = compiler or MemoryCompiler(use_mock=True)
        self.token_budget = token_budget
        self.output_style = output_style
        self.metrics_calculator = MetricsCalculator()

    def run(
        self,
        dataset: BenchmarkDataset,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BenchmarkResult:
        """Run benchmark on a dataset.

        Args:
            dataset: Benchmark dataset to evaluate.
            max_samples: Maximum number of samples to evaluate.
            progress_callback: Optional callback for progress updates.

        Returns:
            Benchmark results with aggregated metrics.
        """
        logger.info(f"Running benchmark: {dataset.name}")

        per_sample_metrics = []
        total_samples = min(len(dataset), max_samples or len(dataset))

        for idx, sample in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break

            if progress_callback:
                progress_callback(idx + 1, total_samples)

            try:
                metrics = self._evaluate_sample(sample)
                per_sample_metrics.append({
                    "sample_id": sample.id,
                    **metrics.to_dict(),
                })
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.id}: {e}")
                continue

        # Aggregate metrics
        aggregated = self._aggregate_metrics(per_sample_metrics)

        return BenchmarkResult(
            benchmark_name=dataset.name,
            num_samples=len(per_sample_metrics),
            metrics=aggregated,
            per_sample_metrics=per_sample_metrics,
            config={
                "token_budget": self.token_budget,
                "output_style": self.output_style,
            },
        )

    def _evaluate_sample(self, sample: BenchmarkSample) -> EvaluationMetrics:
        """Evaluate a single sample."""
        start_time = time.perf_counter()

        # Compress dialogue
        ir = self.compiler.compile(sample.dialogue, token_budget=self.token_budget)
        compressed = self.compiler.generate(ir, style=self.output_style)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate metrics
        compression_ratio = self.metrics_calculator.calculate_compression_ratio(
            sample.dialogue, compressed
        )

        entity_recall = self.metrics_calculator.calculate_entity_recall(
            sample.key_entities, ir
        )

        fact_recall = self.metrics_calculator.calculate_fact_recall(
            sample.key_facts, ir
        )

        # ROUGE-L against reference if available
        rouge_l = 0.0
        if sample.reference_summary:
            rouge_l = self.metrics_calculator.calculate_rouge_l(
                sample.reference_summary, compressed
            )

        # Information retention is average of entity and fact recall
        information_retention = (entity_recall + fact_recall) / 2

        tokens_saved = (
            self.metrics_calculator.estimate_tokens(sample.dialogue)
            - self.metrics_calculator.estimate_tokens(compressed)
        )

        return EvaluationMetrics(
            compression_ratio=compression_ratio,
            information_retention=information_retention,
            entity_recall=entity_recall,
            fact_recall=fact_recall,
            rouge_l=rouge_l,
            latency_ms=latency_ms,
            tokens_saved=tokens_saved,
        )

    def _aggregate_metrics(
        self, per_sample_metrics: List[Dict[str, Any]]
    ) -> EvaluationMetrics:
        """Aggregate metrics across samples."""
        if not per_sample_metrics:
            return EvaluationMetrics()

        def mean(key: str) -> float:
            values = [m[key] for m in per_sample_metrics if key in m]
            return sum(values) / len(values) if values else 0.0

        return EvaluationMetrics(
            compression_ratio=mean("compression_ratio"),
            information_retention=mean("information_retention"),
            entity_recall=mean("entity_recall"),
            fact_recall=mean("fact_recall"),
            rouge_l=mean("rouge_l"),
            bleu=mean("bleu"),
            latency_ms=mean("latency_ms"),
            tokens_saved=int(mean("tokens_saved")),
        )

    def compare_configs(
        self,
        dataset: BenchmarkDataset,
        configs: List[Dict[str, Any]],
        max_samples: int = 100,
    ) -> List[BenchmarkResult]:
        """Compare multiple configurations.

        Args:
            dataset: Benchmark dataset.
            configs: List of configuration dicts with token_budget, etc.
            max_samples: Maximum samples per config.

        Returns:
            List of results for each configuration.
        """
        results = []

        for config in configs:
            self.token_budget = config.get("token_budget", 2048)
            self.output_style = config.get("output_style", "narrative")

            result = self.run(dataset, max_samples=max_samples)
            result.config.update(config)
            results.append(result)

        return results
