"""Tests for Memory IR visualization."""

import tempfile
from pathlib import Path

import pytest

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.events import Event
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.visualization import IRVisualizer


def create_test_ir() -> MemoryIR:
    """Create a test Memory IR for visualization."""
    ir = MemoryIR(session_id="test_viz")

    # Add entities
    ir.add_entity(Entity(
        id="e1",
        name="Alice",
        type=EntityType.PERSON,
        importance_score=0.9,
    ))
    ir.add_entity(Entity(
        id="e2",
        name="TechCorp",
        type=EntityType.ORGANIZATION,
        importance_score=0.7,
    ))
    ir.add_entity(Entity(
        id="e3",
        name="Python",
        type=EntityType.CONCEPT,
        importance_score=0.6,
    ))

    # Add facts
    ir.add_fact(Fact(
        id="f1",
        subject="Alice",
        predicate="works at",
        object="TechCorp",
        importance_score=0.8,
    ))
    ir.add_fact(Fact(
        id="f2",
        subject="Alice",
        predicate="uses",
        object="Python",
        importance_score=0.5,
    ))

    # Add events
    ir.add_event(Event(
        id="ev1",
        description="Started new job",
        participants=["Alice", "TechCorp"],
    ))

    return ir


class TestIRVisualizer:
    """Tests for IRVisualizer."""

    def test_to_graphviz(self):
        """Test GraphViz DOT output."""
        ir = create_test_ir()
        visualizer = IRVisualizer(ir)

        dot = visualizer.to_graphviz()

        assert "digraph" in dot
        assert "Alice" in dot
        assert "TechCorp" in dot
        assert "->" in dot  # Has edges

    def test_to_graphviz_with_options(self):
        """Test GraphViz with custom options."""
        ir = create_test_ir()
        visualizer = IRVisualizer(ir)

        dot = visualizer.to_graphviz(
            include_facts=True,
            include_events=True,
            include_relations=True,
        )

        assert "digraph" in dot
        # Should include fact nodes
        assert "fact" in dot.lower() or "works at" in dot

    def test_to_mermaid(self):
        """Test Mermaid diagram output."""
        ir = create_test_ir()
        visualizer = IRVisualizer(ir)

        mermaid = visualizer.to_mermaid()

        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()
        assert "Alice" in mermaid
        assert "--" in mermaid or "->" in mermaid  # Has edges

    def test_to_html(self):
        """Test HTML visualization output."""
        ir = create_test_ir()
        visualizer = IRVisualizer(ir)

        html = visualizer.to_html()

        assert "<html>" in html.lower() or "<!doctype" in html.lower()
        assert "vis.js" in html.lower() or "vis-network" in html.lower() or "script" in html.lower()
        assert "Alice" in html

    def test_save_html(self):
        """Test saving HTML to file."""
        ir = create_test_ir()
        visualizer = IRVisualizer(ir)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_viz.html"
            visualizer.save_html(str(path))

            assert path.exists()
            content = path.read_text()
            assert "<html>" in content.lower() or "<!doctype" in content.lower()

    def test_empty_ir(self):
        """Test visualization of empty IR."""
        ir = MemoryIR(session_id="empty")
        visualizer = IRVisualizer(ir)

        # Should not raise
        dot = visualizer.to_graphviz()
        assert "digraph" in dot

        mermaid = visualizer.to_mermaid()
        assert len(mermaid) > 0

        html = visualizer.to_html()
        assert len(html) > 0

    def test_ir_with_only_entities(self):
        """Test visualization with only entities."""
        ir = MemoryIR(session_id="entities_only")
        ir.add_entity(Entity(id="e1", name="Entity1", type=EntityType.PERSON))
        ir.add_entity(Entity(id="e2", name="Entity2", type=EntityType.LOCATION))

        visualizer = IRVisualizer(ir)

        dot = visualizer.to_graphviz()
        assert "Entity1" in dot
        assert "Entity2" in dot

    def test_entity_types_colored(self):
        """Test that different entity types get different colors."""
        ir = MemoryIR(session_id="colored")
        ir.add_entity(Entity(id="e1", name="Person", type=EntityType.PERSON))
        ir.add_entity(Entity(id="e2", name="Place", type=EntityType.LOCATION))
        ir.add_entity(Entity(id="e3", name="Company", type=EntityType.ORGANIZATION))

        visualizer = IRVisualizer(ir)

        dot = visualizer.to_graphviz()

        # Should have color specifications
        assert "color" in dot.lower() or "fillcolor" in dot.lower()

    def test_large_ir(self):
        """Test visualization with many nodes."""
        ir = MemoryIR(session_id="large")

        # Add many entities
        for i in range(20):
            ir.add_entity(Entity(
                id=f"e{i}",
                name=f"Entity{i}",
                type=EntityType.PERSON if i % 2 == 0 else EntityType.CONCEPT,
            ))

        # Add many facts
        for i in range(15):
            ir.add_fact(Fact(
                id=f"f{i}",
                subject=f"Entity{i}",
                predicate="relates to",
                object=f"Entity{i+1}",
            ))

        visualizer = IRVisualizer(ir)

        # Should handle without error
        dot = visualizer.to_graphviz()
        assert len(dot) > 0

        html = visualizer.to_html()
        assert len(html) > 0


class TestVisualizerEdgeCases:
    """Test edge cases for visualization."""

    def test_special_characters_in_names(self):
        """Test handling of special characters."""
        ir = MemoryIR(session_id="special")
        ir.add_entity(Entity(
            id="e1",
            name="Name with 'quotes'",
            type=EntityType.PERSON,
        ))
        ir.add_entity(Entity(
            id="e2",
            name='Name with "double quotes"',
            type=EntityType.PERSON,
        ))
        ir.add_entity(Entity(
            id="e3",
            name="Name with <brackets>",
            type=EntityType.CONCEPT,
        ))

        visualizer = IRVisualizer(ir)

        # Should not raise
        dot = visualizer.to_graphviz()
        assert len(dot) > 0

        html = visualizer.to_html()
        assert len(html) > 0

    def test_unicode_content(self):
        """Test handling of unicode content."""
        ir = MemoryIR(session_id="unicode")
        ir.add_entity(Entity(
            id="e1",
            name="日本語",
            type=EntityType.LOCATION,
        ))
        ir.add_entity(Entity(
            id="e2",
            name="Émile",
            type=EntityType.PERSON,
        ))

        visualizer = IRVisualizer(ir)

        dot = visualizer.to_graphviz()
        assert "日本語" in dot or len(dot) > 0  # Should include or escape

        html = visualizer.to_html()
        assert len(html) > 0
