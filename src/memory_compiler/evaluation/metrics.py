"""Evaluation metrics for memory compression.

This module provides metrics for evaluating:
1. Compression efficiency (token reduction)
2. Information retention (fact recall, QA accuracy)
3. Downstream task performance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import CompressionResult


@dataclass
class CompressionMetrics:
    """Metrics for compression efficiency.

    Attributes:
        original_tokens: Tokens in original dialogue.
        compressed_tokens: Tokens in compressed output.
        compression_ratio: Ratio of compressed to original.
        token_reduction: Percentage of tokens removed.
        entities_original: Number of entities before compression.
        entities_retained: Number of entities after compression.
        facts_original: Number of facts before compression.
        facts_retained: Number of facts after compression.
    """

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    token_reduction: float
    entities_original: int = 0
    entities_retained: int = 0
    facts_original: int = 0
    facts_retained: int = 0

    @classmethod
    def from_result(
        cls,
        result: CompressionResult,
        original_ir: MemoryIR | None = None,
    ) -> CompressionMetrics:
        """Create metrics from a compression result."""
        metrics = cls(
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
            token_reduction=1.0 - result.compression_ratio,
            entities_retained=len(result.ir.entities),
            facts_retained=len(result.ir.facts),
        )

        if original_ir:
            metrics.entities_original = len(original_ir.entities)
            metrics.facts_original = len(original_ir.facts)

        return metrics


@dataclass
class InformationRetentionMetrics:
    """Metrics for information retention quality.

    Attributes:
        fact_recall: Fraction of original facts preserved.
        entity_recall: Fraction of original entities preserved.
        fact_precision: Fraction of retained facts that are accurate.
        qa_accuracy: Accuracy on memory-based QA task.
        semantic_similarity: Semantic similarity between original and compressed.
    """

    fact_recall: float = 0.0
    entity_recall: float = 0.0
    fact_precision: float = 1.0
    qa_accuracy: float = 0.0
    semantic_similarity: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def compute_fact_recall(
    original_facts: list[Fact],
    retained_facts: list[Fact],
    threshold: float = 0.8,
) -> tuple[float, list[str]]:
    """Compute fact recall - fraction of original facts preserved.

    Args:
        original_facts: Facts from original dialogue.
        retained_facts: Facts in compressed output.
        threshold: Similarity threshold for matching.

    Returns:
        Tuple of (recall score, list of missed fact descriptions).
    """
    if not original_facts:
        return 1.0, []

    matched = 0
    missed_facts = []

    for orig_fact in original_facts:
        found = False
        for ret_fact in retained_facts:
            similarity = compute_fact_similarity(orig_fact, ret_fact)
            if similarity >= threshold:
                found = True
                break

        if found:
            matched += 1
        else:
            missed_facts.append(orig_fact.natural_language)

    recall = matched / len(original_facts)
    return recall, missed_facts


def compute_fact_similarity(fact1: Fact, fact2: Fact) -> float:
    """Compute similarity between two facts.

    Returns a score between 0 and 1.
    """
    # Check for exact triple match
    if fact1.triple == fact2.triple:
        return 1.0

    # Component-wise comparison
    score = 0.0

    # Subject similarity
    if fact1.subject.lower() == fact2.subject.lower():
        score += 0.4
    elif fact1.subject.lower() in fact2.subject.lower():
        score += 0.2
    elif fact2.subject.lower() in fact1.subject.lower():
        score += 0.2

    # Predicate similarity
    if fact1.predicate.lower() == fact2.predicate.lower():
        score += 0.3
    elif _are_predicates_similar(fact1.predicate, fact2.predicate):
        score += 0.15

    # Object similarity
    if fact1.object.lower() == fact2.object.lower():
        score += 0.3
    elif fact1.object.lower() in fact2.object.lower():
        score += 0.15
    elif fact2.object.lower() in fact1.object.lower():
        score += 0.15

    return score


def _are_predicates_similar(pred1: str, pred2: str) -> bool:
    """Check if two predicates are semantically similar."""
    pred1 = pred1.lower()
    pred2 = pred2.lower()

    similar_groups = [
        {"is", "are", "was", "were", "be"},
        {"works at", "works for", "employed at", "employed by"},
        {"likes", "enjoys", "loves", "prefers"},
        {"has", "owns", "possesses"},
        {"lives in", "resides in", "located in"},
    ]

    for group in similar_groups:
        if pred1 in group and pred2 in group:
            return True

    return False


def compute_entity_recall(
    original_entities: list[str],
    retained_entities: list[str],
) -> float:
    """Compute entity recall - fraction of original entities preserved."""
    if not original_entities:
        return 1.0

    original_lower = {e.lower() for e in original_entities}
    retained_lower = {e.lower() for e in retained_entities}

    matched = len(original_lower & retained_lower)
    return matched / len(original_lower)


def evaluate_qa_accuracy(
    compressed_context: str,
    qa_pairs: list[tuple[str, str]],
    model: Any = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate QA accuracy using compressed context.

    Args:
        compressed_context: The compressed dialogue context.
        qa_pairs: List of (question, answer) tuples.
        model: Language model for answering (optional).

    Returns:
        Tuple of (accuracy, detailed results).
    """
    if not qa_pairs:
        return 1.0, {}

    # Simple keyword-based evaluation if no model
    if model is None:
        correct = 0
        results = []

        for question, expected_answer in qa_pairs:
            # Check if answer keywords are in context
            answer_keywords = set(expected_answer.lower().split())
            context_words = set(compressed_context.lower().split())

            overlap = len(answer_keywords & context_words) / max(len(answer_keywords), 1)
            is_correct = overlap >= 0.5

            if is_correct:
                correct += 1

            results.append(
                {
                    "question": question,
                    "expected": expected_answer,
                    "correct": is_correct,
                    "overlap": overlap,
                }
            )

        accuracy = correct / len(qa_pairs)
        return accuracy, {"detailed_results": results}

    # Model-based evaluation
    correct = 0
    results = []

    for question, expected_answer in qa_pairs:
        prompt = f"Context: {compressed_context}\n\nQuestion: {question}\nAnswer:"
        predicted = model.generate(prompt, max_length=100)

        # Simple match check
        is_correct = expected_answer.lower() in predicted.lower()
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "expected": expected_answer,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(qa_pairs)
    return accuracy, {"detailed_results": results}


def compute_semantic_similarity(
    original_text: str,
    compressed_text: str,
    model: Any = None,
) -> float:
    """Compute semantic similarity between original and compressed text."""
    if model is None:
        # Simple word overlap as fallback
        orig_words = set(original_text.lower().split())
        comp_words = set(compressed_text.lower().split())

        intersection = len(orig_words & comp_words)
        union = len(orig_words | comp_words)

        return intersection / max(union, 1)

    # Use embedding model
    try:
        orig_emb = model.encode(original_text)
        comp_emb = model.encode(compressed_text)

        # Cosine similarity
        import numpy as np

        similarity = np.dot(orig_emb, comp_emb) / (
            np.linalg.norm(orig_emb) * np.linalg.norm(comp_emb)
        )
        return float(similarity)
    except Exception as e:
        logger.warning(f"Embedding similarity failed: {e}")
        return 0.0


def evaluate_compression(
    result: CompressionResult,
    original_ir: MemoryIR | None = None,
    qa_pairs: list[tuple[str, str]] | None = None,
    original_text: str | None = None,
) -> dict[str, Any]:
    """Comprehensive evaluation of compression quality.

    Args:
        result: The compression result to evaluate.
        original_ir: Original Memory IR before compression.
        qa_pairs: QA pairs for memory evaluation.
        original_text: Original dialogue text for similarity.

    Returns:
        Dictionary with all evaluation metrics.
    """
    metrics: dict[str, Any] = {}

    # Compression metrics
    comp_metrics = CompressionMetrics.from_result(result, original_ir)
    metrics["compression"] = {
        "original_tokens": comp_metrics.original_tokens,
        "compressed_tokens": comp_metrics.compressed_tokens,
        "compression_ratio": comp_metrics.compression_ratio,
        "token_reduction": comp_metrics.token_reduction,
    }

    # Information retention
    retention_metrics = InformationRetentionMetrics()

    if original_ir:
        # Fact recall
        original_facts = list(original_ir.iter_facts())
        retained_facts = list(result.ir.iter_facts())
        fact_recall, missed = compute_fact_recall(original_facts, retained_facts)
        retention_metrics.fact_recall = fact_recall
        retention_metrics.details["missed_facts"] = missed

        # Entity recall
        original_entities = [e.name for e in original_ir.iter_entities()]
        retained_entities = [e.name for e in result.ir.iter_entities()]
        retention_metrics.entity_recall = compute_entity_recall(
            original_entities, retained_entities
        )

    # QA accuracy
    if qa_pairs:
        qa_acc, qa_details = evaluate_qa_accuracy(result.text, qa_pairs)
        retention_metrics.qa_accuracy = qa_acc
        retention_metrics.details["qa_results"] = qa_details

    # Semantic similarity
    if original_text:
        retention_metrics.semantic_similarity = compute_semantic_similarity(
            original_text, result.text
        )

    metrics["retention"] = {
        "fact_recall": retention_metrics.fact_recall,
        "entity_recall": retention_metrics.entity_recall,
        "qa_accuracy": retention_metrics.qa_accuracy,
        "semantic_similarity": retention_metrics.semantic_similarity,
    }

    # Pass-specific metrics
    metrics["passes"] = {}
    for pass_name, pass_result in result.pass_results.items():
        metrics["passes"][pass_name] = {
            "entities_removed": pass_result.entities_removed,
            "facts_removed": pass_result.facts_removed,
            "facts_merged": pass_result.facts_merged,
            "events_removed": pass_result.events_removed,
            "events_compressed": pass_result.events_compressed,
            **pass_result.stats,
        }

    return metrics


class EvaluationRunner:
    """Runs evaluation across multiple dialogues and aggregates results."""

    def __init__(self, compiler: Any) -> None:
        self.compiler = compiler
        self.results: list[dict[str, Any]] = []

    def evaluate_single(
        self,
        dialogue: list[dict[str, str]],
        qa_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate compression on a single dialogue."""
        # Get original text
        original_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in dialogue
        )

        # Compress
        result = self.compiler.compress(dialogue)

        # Evaluate
        metrics = evaluate_compression(
            result,
            qa_pairs=qa_pairs,
            original_text=original_text,
        )

        self.results.append(metrics)
        return metrics

    def evaluate_batch(
        self,
        dialogues: list[list[dict[str, str]]],
        qa_pairs_list: list[list[tuple[str, str]]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate compression on multiple dialogues."""
        if qa_pairs_list is None:
            qa_pairs_list = [None] * len(dialogues)

        for dialogue, qa_pairs in zip(dialogues, qa_pairs_list):
            self.evaluate_single(dialogue, qa_pairs)

        return self.aggregate_results()

    def aggregate_results(self) -> dict[str, Any]:
        """Aggregate results across all evaluated dialogues."""
        if not self.results:
            return {}

        import numpy as np

        # Aggregate compression metrics
        comp_ratios = [r["compression"]["compression_ratio"] for r in self.results]
        token_reductions = [r["compression"]["token_reduction"] for r in self.results]

        # Aggregate retention metrics
        fact_recalls = [
            r["retention"]["fact_recall"]
            for r in self.results
            if r["retention"]["fact_recall"] > 0
        ]
        entity_recalls = [
            r["retention"]["entity_recall"]
            for r in self.results
            if r["retention"]["entity_recall"] > 0
        ]
        qa_accs = [
            r["retention"]["qa_accuracy"]
            for r in self.results
            if r["retention"]["qa_accuracy"] > 0
        ]

        aggregated = {
            "num_samples": len(self.results),
            "compression": {
                "mean_ratio": float(np.mean(comp_ratios)),
                "std_ratio": float(np.std(comp_ratios)),
                "mean_reduction": float(np.mean(token_reductions)),
            },
            "retention": {
                "mean_fact_recall": float(np.mean(fact_recalls)) if fact_recalls else None,
                "mean_entity_recall": float(np.mean(entity_recalls)) if entity_recalls else None,
                "mean_qa_accuracy": float(np.mean(qa_accs)) if qa_accs else None,
            },
        }

        return aggregated

    def clear(self) -> None:
        """Clear stored results."""
        self.results = []
