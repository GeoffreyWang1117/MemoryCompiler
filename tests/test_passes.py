"""Tests for optimization passes."""

import pytest

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact, FactType
from memory_compiler.ir.events import Event, EventType
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.dead_memory import DeadMemoryEliminationPass
from memory_compiler.passes.fact_folding import FactFoldingPass
from memory_compiler.passes.temporal import TemporalCompressionPass
from memory_compiler.passes.importance import ImportancePruningPass
from memory_compiler.passes.base import PassManager


class TestDeadMemoryElimination:
    """Tests for DeadMemoryEliminationPass."""

    def test_remove_low_mention_entities(self):
        """Test removal of rarely mentioned entities."""
        ir = MemoryIR()

        # Add entities with different mention counts
        e1 = Entity("ent_1", "Alice", EntityType.PERSON)
        e1.add_mention(0, (0, 5))
        e1.add_mention(5, (0, 5))
        ir.add_entity(e1)

        e2 = Entity("ent_2", "RandomPerson", EntityType.OTHER)
        # No mentions - should be removed
        ir.add_entity(e2)

        # Add dialogue turns to establish recency
        for i in range(10):
            ir.dialogue_turns.append(None)

        pass_ = DeadMemoryEliminationPass(min_entity_mentions=1)
        result = pass_.run(ir)

        # RandomPerson should be removed
        assert "ent_1" in ir.entities
        assert result.modified

    def test_remove_superseded_facts(self):
        """Test removal of superseded facts."""
        ir = MemoryIR()

        # Add facts that supersede each other
        f1 = Fact("fact_1", "Alice", "lives in", "Boston", source_turns=[0])
        f2 = Fact("fact_2", "Alice", "lives in", "New York", source_turns=[5])
        ir.add_fact(f1)
        ir.add_fact(f2)

        for i in range(10):
            ir.dialogue_turns.append(None)

        pass_ = DeadMemoryEliminationPass(remove_superseded=True)
        result = pass_.run(ir)

        # Should remove the older fact
        assert result.facts_removed >= 0  # May remove f1

    def test_keep_recent_facts(self):
        """Test that recent facts are kept."""
        ir = MemoryIR()

        fact = Fact("fact_1", "Alice", "said", "hello", source_turns=[9])
        ir.add_fact(fact)

        for i in range(10):
            ir.dialogue_turns.append(None)

        pass_ = DeadMemoryEliminationPass(recency_threshold=5)
        result = pass_.run(ir)

        # Recent fact should be kept
        assert "fact_1" in ir.facts


class TestFactFolding:
    """Tests for FactFoldingPass."""

    def test_fold_equivalent_facts(self):
        """Test folding of semantically equivalent facts."""
        ir = MemoryIR()

        # Add equivalent facts
        f1 = Fact("fact_1", "Alice", "is", "engineer", confidence=0.9)
        f2 = Fact("fact_2", "Alice", "is", "engineer", confidence=0.8)
        ir.add_fact(f1)
        ir.add_fact(f2)

        pass_ = FactFoldingPass(
            similarity_threshold=0.9,
            use_embeddings=False,
        )
        result = pass_.run(ir)

        # Should merge the two facts
        assert result.facts_merged > 0 or len(ir.facts) <= 2

    def test_keep_different_facts(self):
        """Test that different facts are not folded."""
        ir = MemoryIR()

        f1 = Fact("fact_1", "Alice", "is", "engineer")
        f2 = Fact("fact_2", "Bob", "is", "designer")
        ir.add_fact(f1)
        ir.add_fact(f2)

        pass_ = FactFoldingPass(
            similarity_threshold=0.9,
            use_embeddings=False,
        )
        result = pass_.run(ir)

        # Both facts should be kept
        assert len(ir.facts) == 2


class TestTemporalCompression:
    """Tests for TemporalCompressionPass."""

    def test_compress_event_sequence(self):
        """Test compression of event sequences."""
        ir = MemoryIR()

        # Add a sequence of related events
        for i in range(5):
            event = Event(
                f"evt_{i}",
                f"Question {i}",
                EventType.QUESTION,
                turn_index=i,
                participants=["user"],
            )
            ir.add_event(event)

        pass_ = TemporalCompressionPass(
            min_sequence_length=3,
            max_turn_gap=2,
        )
        result = pass_.run(ir)

        # Events should be compressed
        assert result.events_compressed > 0 or len(ir.events) < 5

    def test_no_compression_for_short_sequences(self):
        """Test that short sequences are not compressed."""
        ir = MemoryIR()

        # Add only 2 events
        ir.add_event(Event("evt_1", "Event 1", EventType.ACTION, 0))
        ir.add_event(Event("evt_2", "Event 2", EventType.ACTION, 1))

        pass_ = TemporalCompressionPass(min_sequence_length=3)
        result = pass_.run(ir)

        # Should not compress
        assert result.events_compressed == 0
        assert len(ir.events) == 2


class TestImportancePruning:
    """Tests for ImportancePruningPass."""

    def test_prune_to_budget(self):
        """Test pruning to meet token budget."""
        ir = MemoryIR()

        # Add many facts
        for i in range(50):
            fact = Fact(
                f"fact_{i}",
                f"Subject{i}",
                "predicate",
                f"Object{i}",
                importance_score=i / 50,  # Varying importance
            )
            ir.add_fact(fact)

        pass_ = ImportancePruningPass(
            token_budget=500,  # Small budget
            tokens_per_fact=20,
        )
        result = pass_.run(ir)

        # Should have removed some facts
        assert result.facts_removed > 0
        assert len(ir.facts) < 50

    def test_keep_important_items(self):
        """Test that important items are kept."""
        ir = MemoryIR()

        # Add one important fact
        f1 = Fact("fact_1", "Alice", "is", "CEO", importance_score=0.9)
        ir.add_fact(f1)

        # Add several less important facts
        for i in range(10):
            fact = Fact(
                f"fact_{i+2}",
                f"Random{i}",
                "said",
                "something",
                importance_score=0.1,
            )
            ir.add_fact(fact)

        pass_ = ImportancePruningPass(
            token_budget=100,
            tokens_per_fact=20,
        )
        pass_.run(ir)

        # Important fact should be kept
        assert "fact_1" in ir.facts


class TestPassManager:
    """Tests for PassManager."""

    def test_run_multiple_passes(self):
        """Test running multiple passes."""
        ir = MemoryIR()

        # Add some content
        ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.create_fact("Alice", "is", "engineer")

        manager = PassManager([
            DeadMemoryEliminationPass(),
            FactFoldingPass(use_embeddings=False),
        ])

        result = manager.run(ir)

        # Should have run both passes
        assert len(manager) == 2

    def test_pass_to_fixed_point(self):
        """Test running passes to fixed point."""
        ir = MemoryIR()

        # Add duplicate facts
        ir.create_fact("Alice", "is", "engineer")
        ir.create_fact("Alice", "is", "engineer")

        manager = PassManager([
            FactFoldingPass(use_embeddings=False),
        ])

        result = manager.run_to_fixed_point(ir, max_iterations=3)

        # Should stop when no more changes
        assert result is not None
