"""Tests for Memory IR data structures."""

import pytest

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact, FactType
from memory_compiler.ir.relations import Relation, RelationType, RelationGraph
from memory_compiler.ir.events import Event, EventSequence, EventType
from memory_compiler.ir.memory_ir import MemoryIR


class TestEntity:
    """Tests for Entity class."""

    def test_create_entity(self):
        """Test entity creation."""
        entity = Entity(
            id="ent_1",
            name="Alice",
            type=EntityType.PERSON,
        )
        assert entity.id == "ent_1"
        assert entity.name == "Alice"
        assert entity.type == EntityType.PERSON

    def test_entity_mentions(self):
        """Test adding mentions to entity."""
        entity = Entity(id="ent_1", name="Alice", type=EntityType.PERSON)
        entity.add_mention(0, (0, 5))
        entity.add_mention(2, (10, 15))

        assert entity.reference_count == 2
        assert entity.first_mention_turn == 0
        assert entity.last_mention_turn == 2

    def test_entity_aliases(self):
        """Test adding aliases to entity."""
        entity = Entity(id="ent_1", name="Alice", type=EntityType.PERSON)
        entity.add_alias("Al")
        entity.add_alias("Alice")  # Same as name, should not add

        assert "Al" in entity.aliases
        assert len(entity.aliases) == 1

    def test_entity_serialization(self):
        """Test entity serialization."""
        entity = Entity(
            id="ent_1",
            name="Alice",
            type=EntityType.PERSON,
            aliases=["Al"],
            attributes={"age": 30},
        )

        data = entity.to_dict()
        restored = Entity.from_dict(data)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.type == entity.type
        assert restored.aliases == entity.aliases


class TestFact:
    """Tests for Fact class."""

    def test_create_fact(self):
        """Test fact creation."""
        fact = Fact(
            id="fact_1",
            subject="Alice",
            predicate="works at",
            object="TechCorp",
            fact_type=FactType.MEMBERSHIP,
        )

        assert fact.triple == ("Alice", "works at", "TechCorp")
        assert fact.natural_language == "Alice works at TechCorp"

    def test_fact_negation(self):
        """Test negated fact."""
        fact = Fact(
            id="fact_1",
            subject="Alice",
            predicate="likes",
            object="coffee",
            negated=True,
        )

        assert "not" in fact.natural_language.lower()

    def test_fact_contradiction(self):
        """Test fact contradiction detection."""
        fact1 = Fact(
            id="fact_1",
            subject="Alice",
            predicate="lives in",
            object="New York",
        )
        fact2 = Fact(
            id="fact_2",
            subject="Alice",
            predicate="lives in",
            object="Boston",
        )

        assert fact1.contradicts(fact2)

    def test_fact_serialization(self):
        """Test fact serialization."""
        fact = Fact(
            id="fact_1",
            subject="Alice",
            predicate="is",
            object="engineer",
            confidence=0.9,
            source_turns=[0, 2],
        )

        data = fact.to_dict()
        restored = Fact.from_dict(data)

        assert restored.id == fact.id
        assert restored.triple == fact.triple
        assert restored.confidence == fact.confidence


class TestRelation:
    """Tests for Relation and RelationGraph classes."""

    def test_create_relation(self):
        """Test relation creation."""
        relation = Relation(
            id="rel_1",
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=RelationType.CAUSES,
        )

        assert relation.is_causal
        assert not relation.is_temporal

    def test_relation_inverse(self):
        """Test relation inverse."""
        relation = Relation(
            id="rel_1",
            source_fact_id="fact_1",
            target_fact_id="fact_2",
            relation_type=RelationType.CAUSES,
        )

        inverse = relation.inverse()
        assert inverse.source_fact_id == "fact_2"
        assert inverse.target_fact_id == "fact_1"
        assert inverse.relation_type == RelationType.CAUSED_BY

    def test_relation_graph(self):
        """Test relation graph operations."""
        graph = RelationGraph()

        rel1 = Relation("rel_1", "fact_1", "fact_2", RelationType.CAUSES)
        rel2 = Relation("rel_2", "fact_2", "fact_3", RelationType.ENTAILS)

        graph.add_relation(rel1)
        graph.add_relation(rel2)

        assert len(graph) == 2

        # Test query
        relations_from_1 = graph.get_relations_from("fact_1")
        assert len(relations_from_1) == 1

        related = graph.get_related_facts("fact_2")
        assert "fact_1" in related
        assert "fact_3" in related


class TestEvent:
    """Tests for Event and EventSequence classes."""

    def test_create_event(self):
        """Test event creation."""
        event = Event(
            id="evt_1",
            description="User asked about Python",
            event_type=EventType.QUESTION,
            turn_index=0,
        )

        assert event.event_type == EventType.QUESTION
        assert event.turn_index == 0

    def test_event_sequence(self):
        """Test event sequence."""
        events = [
            Event("evt_1", "Question 1", EventType.QUESTION, 0),
            Event("evt_2", "Answer 1", EventType.ANSWER, 1),
            Event("evt_3", "Question 2", EventType.QUESTION, 2),
        ]

        seq = EventSequence(id="seq_1", events=events)

        assert seq.start_turn == 0
        assert seq.end_turn == 2
        assert seq.duration == 3

    def test_sequence_merge(self):
        """Test merging event sequences."""
        seq1 = EventSequence(
            id="seq_1",
            events=[Event("evt_1", "Event 1", EventType.ACTION, 0)],
        )
        seq2 = EventSequence(
            id="seq_2",
            events=[Event("evt_2", "Event 2", EventType.ACTION, 2)],
        )

        # Add shared participant
        seq1.events[0].participants = ["Alice"]
        seq2.events[0].participants = ["Alice"]

        assert seq1.can_merge_with(seq2, max_gap=3)

        merged = seq1.merge_with(seq2)
        assert len(merged.events) == 2


class TestMemoryIR:
    """Tests for MemoryIR class."""

    def test_create_memory_ir(self):
        """Test Memory IR creation."""
        ir = MemoryIR()
        assert len(ir.entities) == 0
        assert len(ir.facts) == 0

    def test_add_entity(self):
        """Test adding entities to IR."""
        ir = MemoryIR()

        entity = Entity("ent_1", "Alice", EntityType.PERSON)
        ir.add_entity(entity)

        assert len(ir.entities) == 1
        assert ir.get_entity("ent_1") == entity
        assert ir.get_entity_by_name("alice") == entity

    def test_add_fact(self):
        """Test adding facts to IR."""
        ir = MemoryIR()

        fact = ir.create_fact("Alice", "works at", "TechCorp")

        assert len(ir.facts) == 1
        facts_about = ir.get_facts_about("Alice")
        assert len(facts_about) == 1

    def test_find_or_create_entity(self):
        """Test find_or_create_entity."""
        ir = MemoryIR()

        entity1 = ir.find_or_create_entity("Alice", EntityType.PERSON)
        entity2 = ir.find_or_create_entity("Alice", EntityType.PERSON)

        assert entity1.id == entity2.id
        assert len(ir.entities) == 1

    def test_remove_entity(self):
        """Test removing entity and associated facts."""
        ir = MemoryIR()

        entity = ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.create_fact("Alice", "is", "engineer")

        ir.remove_entity(entity.id)

        assert len(ir.entities) == 0
        assert len(ir.facts) == 0

    def test_ir_serialization(self):
        """Test IR serialization."""
        ir = MemoryIR()

        ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.create_fact("Alice", "works at", "TechCorp")
        ir.create_event("User introduced themselves", EventType.ACTION, 0)

        data = ir.to_dict()
        restored = MemoryIR.from_dict(data)

        assert len(restored.entities) == len(ir.entities)
        assert len(restored.facts) == len(ir.facts)
        assert len(restored.events) == len(ir.events)

    def test_ir_stats(self):
        """Test IR statistics."""
        ir = MemoryIR()

        ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.find_or_create_entity("TechCorp", EntityType.ORGANIZATION)
        ir.create_fact("Alice", "works at", "TechCorp")

        stats = ir.get_stats()

        assert stats.num_entities == 2
        assert stats.num_facts == 1
        assert "person" in stats.entity_types

    def test_set_dialogue(self):
        """Test setting dialogue."""
        ir = MemoryIR()

        dialogue = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        ir.set_dialogue(dialogue)

        assert len(ir.dialogue_turns) == 2
        assert ir.dialogue_turns[0].role == "user"
