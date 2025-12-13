"""Tests for context optimization module."""

import pytest
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.context import (
    ContextOptimizer,
    PackingStrategy,
    ContextSelector,
    SelectionCriteria,
    ContextPacker,
    OutputFormat,
)


def create_test_ir() -> MemoryIR:
    """Create a test Memory IR with sample data."""
    ir = MemoryIR(session_id="test_context")

    # Add entities
    ir.add_entity(Entity(
        name="Alice",
        type=EntityType.PERSON,
        importance_score=0.9,
        attributes={"role": "engineer"},
    ))
    ir.add_entity(Entity(
        name="Bob",
        type=EntityType.PERSON,
        importance_score=0.7,
    ))
    ir.add_entity(Entity(
        name="TechCorp",
        type=EntityType.ORGANIZATION,
        importance_score=0.6,
    ))

    # Add facts
    ir.add_fact(Fact(
        subject="Alice",
        predicate="works_at",
        object="TechCorp",
        confidence=0.95,
        importance=0.8,
    ))
    ir.add_fact(Fact(
        subject="Bob",
        predicate="manages",
        object="Alice",
        confidence=0.9,
        importance=0.7,
    ))
    ir.add_fact(Fact(
        subject="Alice",
        predicate="likes",
        object="Python",
        confidence=0.8,
        importance=0.5,
    ))

    return ir


class TestContextOptimizer:
    """Tests for ContextOptimizer."""

    def test_greedy_packing(self):
        """Test greedy packing strategy."""
        optimizer = ContextOptimizer(strategy=PackingStrategy.GREEDY)
        ir = create_test_ir()

        result = optimizer.optimize(ir, max_tokens=100)

        assert result is not None
        assert len(result.content) > 0
        assert result.token_count <= 100

    def test_knapsack_packing(self):
        """Test knapsack packing strategy."""
        optimizer = ContextOptimizer(strategy=PackingStrategy.KNAPSACK)
        ir = create_test_ir()

        result = optimizer.optimize(ir, max_tokens=100)

        assert result is not None
        assert result.token_count <= 100

    def test_query_focused_packing(self):
        """Test query-focused packing."""
        optimizer = ContextOptimizer(strategy=PackingStrategy.QUERY_FOCUSED)
        ir = create_test_ir()

        result = optimizer.optimize(
            ir,
            max_tokens=100,
            query="Tell me about Alice's work",
        )

        assert result is not None
        # Should prioritize Alice-related content
        assert "Alice" in result.content

    def test_balanced_packing(self):
        """Test balanced packing strategy."""
        optimizer = ContextOptimizer(strategy=PackingStrategy.BALANCED)
        ir = create_test_ir()

        result = optimizer.optimize(ir, max_tokens=200)

        assert result is not None
        # Should include diverse content
        assert result.entity_count > 0
        assert result.fact_count > 0

    def test_token_limit_respected(self):
        """Test that token limit is always respected."""
        optimizer = ContextOptimizer()
        ir = create_test_ir()

        for limit in [50, 100, 200]:
            result = optimizer.optimize(ir, max_tokens=limit)
            assert result.token_count <= limit


class TestContextSelector:
    """Tests for ContextSelector."""

    def test_importance_selection(self):
        """Test selection by importance."""
        selector = ContextSelector()
        ir = create_test_ir()

        criteria = SelectionCriteria(
            min_importance=0.7,
            max_items=10,
        )

        selected = selector.select(ir, criteria)

        # Should only include high-importance items
        for entity in selected.iter_entities():
            assert entity.importance_score >= 0.7

    def test_recency_selection(self):
        """Test selection by recency."""
        selector = ContextSelector()
        ir = create_test_ir()

        criteria = SelectionCriteria(
            prefer_recent=True,
            max_items=5,
        )

        selected = selector.select(ir, criteria)
        assert selected is not None

    def test_entity_type_filter(self):
        """Test filtering by entity type."""
        selector = ContextSelector()
        ir = create_test_ir()

        criteria = SelectionCriteria(
            entity_types=[EntityType.PERSON],
        )

        selected = selector.select(ir, criteria)

        for entity in selected.iter_entities():
            assert entity.type == EntityType.PERSON

    def test_max_items_limit(self):
        """Test max items limit."""
        selector = ContextSelector()
        ir = create_test_ir()

        criteria = SelectionCriteria(max_items=2)
        selected = selector.select(ir, criteria)

        entity_count = sum(1 for _ in selected.iter_entities())
        assert entity_count <= 2


class TestContextPacker:
    """Tests for ContextPacker."""

    def test_plain_format(self):
        """Test plain text output format."""
        packer = ContextPacker(format=OutputFormat.PLAIN)
        ir = create_test_ir()

        result = packer.pack(ir, max_tokens=200)

        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_structured_format(self):
        """Test structured output format."""
        packer = ContextPacker(format=OutputFormat.STRUCTURED)
        ir = create_test_ir()

        result = packer.pack(ir, max_tokens=200)

        # Should have clear sections
        assert "Entities" in result.content or "entities" in result.content.lower()

    def test_xml_format(self):
        """Test XML output format."""
        packer = ContextPacker(format=OutputFormat.XML)
        ir = create_test_ir()

        result = packer.pack(ir, max_tokens=200)

        # Should have XML tags
        assert "<" in result.content
        assert ">" in result.content

    def test_markdown_format(self):
        """Test Markdown output format."""
        packer = ContextPacker(format=OutputFormat.MARKDOWN)
        ir = create_test_ir()

        result = packer.pack(ir, max_tokens=200)

        # Should have markdown formatting
        assert "#" in result.content or "-" in result.content

    def test_json_format(self):
        """Test JSON output format."""
        packer = ContextPacker(format=OutputFormat.JSON)
        ir = create_test_ir()

        result = packer.pack(ir, max_tokens=500)

        # Should be valid JSON structure indicators
        assert "{" in result.content
        assert "}" in result.content

    def test_include_metadata(self):
        """Test including metadata in output."""
        packer = ContextPacker(include_metadata=True)
        ir = create_test_ir()
        ir.metadata["test_key"] = "test_value"

        result = packer.pack(ir, max_tokens=300)

        assert result.metadata is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
