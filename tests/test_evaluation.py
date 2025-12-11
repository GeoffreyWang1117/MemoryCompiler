"""Tests for evaluation module."""

import pytest

from memory_compiler.evaluation.metrics import (
    CompressionMetrics,
    InformationRetentionMetrics,
    compute_fact_recall,
    compute_entity_recall,
    compute_fact_similarity,
    evaluate_compression,
)
from memory_compiler.evaluation.datasets import (
    DatasetLoader,
    create_synthetic_dialogue,
)
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import CompressionResult


class TestCompressionMetrics:
    """Tests for CompressionMetrics."""

    def test_from_result(self):
        """Test creating metrics from compression result."""
        ir = MemoryIR()
        ir.create_fact("Alice", "is", "engineer")

        result = CompressionResult(
            text="Alice is an engineer",
            ir=ir,
            original_tokens=100,
            compressed_tokens=20,
            compression_ratio=0.2,
        )

        metrics = CompressionMetrics.from_result(result)

        assert metrics.original_tokens == 100
        assert metrics.compressed_tokens == 20
        assert metrics.compression_ratio == 0.2
        assert metrics.token_reduction == 0.8


class TestInformationRetention:
    """Tests for information retention metrics."""

    def test_compute_fact_recall(self):
        """Test fact recall computation."""
        original_facts = [
            Fact("f1", "Alice", "is", "engineer"),
            Fact("f2", "Bob", "works at", "TechCorp"),
        ]

        retained_facts = [
            Fact("f3", "Alice", "is", "engineer"),  # Same as f1
        ]

        recall, missed = compute_fact_recall(original_facts, retained_facts)

        assert 0 <= recall <= 1
        assert len(missed) == 1  # Bob fact is missing

    def test_compute_fact_recall_empty(self):
        """Test fact recall with empty lists."""
        recall, missed = compute_fact_recall([], [])
        assert recall == 1.0

    def test_compute_entity_recall(self):
        """Test entity recall computation."""
        original = ["Alice", "Bob", "TechCorp"]
        retained = ["Alice", "TechCorp"]

        recall = compute_entity_recall(original, retained)

        assert recall == pytest.approx(2/3, rel=0.01)

    def test_compute_fact_similarity(self):
        """Test fact similarity computation."""
        f1 = Fact("f1", "Alice", "is", "engineer")
        f2 = Fact("f2", "Alice", "is", "engineer")
        f3 = Fact("f3", "Bob", "likes", "coffee")

        sim_same = compute_fact_similarity(f1, f2)
        sim_diff = compute_fact_similarity(f1, f3)

        assert sim_same == 1.0  # Exact match
        assert sim_diff < sim_same


class TestEvaluateCompression:
    """Tests for comprehensive evaluation."""

    def test_evaluate_compression(self):
        """Test full evaluation pipeline."""
        ir = MemoryIR()
        ir.create_fact("Alice", "is", "engineer")

        result = CompressionResult(
            text="Alice is an engineer.",
            ir=ir,
            original_tokens=100,
            compressed_tokens=25,
            compression_ratio=0.25,
        )

        metrics = evaluate_compression(
            result,
            original_text="User said Alice is an engineer and works at TechCorp.",
        )

        assert "compression" in metrics
        assert "retention" in metrics
        assert metrics["compression"]["compression_ratio"] == 0.25

    def test_evaluate_with_qa(self):
        """Test evaluation with QA pairs."""
        ir = MemoryIR()
        ir.create_fact("Alice", "works at", "TechCorp")

        result = CompressionResult(
            text="Alice works at TechCorp as an engineer.",
            ir=ir,
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.3,
        )

        qa_pairs = [
            ("Where does Alice work?", "TechCorp"),
            ("What is Alice's job?", "engineer"),
        ]

        metrics = evaluate_compression(result, qa_pairs=qa_pairs)

        assert "retention" in metrics
        assert "qa_accuracy" in metrics["retention"]


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_supported_datasets(self):
        """Test that expected datasets are supported."""
        assert "msc" in DatasetLoader.SUPPORTED_DATASETS
        assert "multiwoz" in DatasetLoader.SUPPORTED_DATASETS
        assert "daily_dialog" in DatasetLoader.SUPPORTED_DATASETS

    def test_unsupported_dataset(self):
        """Test error for unsupported dataset."""
        with pytest.raises(ValueError):
            DatasetLoader("nonexistent_dataset")

    def test_create_synthetic_dialogue(self):
        """Test synthetic dialogue creation."""
        dialogue = create_synthetic_dialogue(num_turns=10)

        assert len(dialogue) == 10
        assert all("role" in turn for turn in dialogue)
        assert all("content" in turn for turn in dialogue)

    def test_synthetic_dialogue_content(self):
        """Test that synthetic dialogue has meaningful content."""
        dialogue = create_synthetic_dialogue(num_turns=4)

        # First turn should introduce user
        assert "name" in dialogue[0]["content"].lower()

        # Should have user and assistant turns
        roles = [t["role"] for t in dialogue]
        assert "user" in roles
        assert "assistant" in roles
