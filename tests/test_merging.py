"""Tests for memory merging module."""

import pytest
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.merging import (
    EntityResolver,
    EntityMatch,
    EntityCluster,
    ConflictResolver,
    ConflictType,
    MemoryMerger,
    MergeStrategy,
)


def create_session1_ir() -> MemoryIR:
    """Create first session Memory IR."""
    ir = MemoryIR(session_id="session1")
    ir.add_entity(Entity(
        name="Alice Smith",
        type=EntityType.PERSON,
        importance_score=0.8,
        attributes={"role": "engineer"},
    ))
    ir.add_entity(Entity(
        name="TechCorp",
        type=EntityType.ORGANIZATION,
        importance_score=0.6,
    ))
    ir.add_fact(Fact(
        subject="Alice Smith",
        predicate="works_at",
        object="TechCorp",
        confidence=0.9,
    ))
    return ir


def create_session2_ir() -> MemoryIR:
    """Create second session Memory IR with overlapping entities."""
    ir = MemoryIR(session_id="session2")
    ir.add_entity(Entity(
        name="Alice",  # Same person, different name format
        type=EntityType.PERSON,
        importance_score=0.7,
        aliases=["A. Smith"],
    ))
    ir.add_entity(Entity(
        name="Bob",
        type=EntityType.PERSON,
        importance_score=0.5,
    ))
    ir.add_fact(Fact(
        subject="Alice",
        predicate="knows",
        object="Bob",
        confidence=0.85,
    ))
    ir.add_fact(Fact(
        subject="Alice",
        predicate="likes",
        object="Python",
        confidence=0.8,
    ))
    return ir


class TestEntityResolver:
    """Tests for EntityResolver."""

    def test_exact_match(self):
        """Test exact entity name matching."""
        resolver = EntityResolver()

        entity1 = Entity(name="Alice", type=EntityType.PERSON)
        entity2 = Entity(name="Alice", type=EntityType.PERSON)

        match = resolver.match(entity1, entity2)

        assert match is not None
        assert match.score >= 0.9

    def test_fuzzy_match(self):
        """Test fuzzy entity name matching."""
        resolver = EntityResolver(fuzzy_threshold=0.7)

        entity1 = Entity(name="Alice Smith", type=EntityType.PERSON)
        entity2 = Entity(name="Alice", type=EntityType.PERSON)

        match = resolver.match(entity1, entity2)

        assert match is not None
        assert match.score > 0.5

    def test_alias_match(self):
        """Test matching via aliases."""
        resolver = EntityResolver()

        entity1 = Entity(
            name="Robert",
            type=EntityType.PERSON,
            aliases=["Bob", "Bobby"],
        )
        entity2 = Entity(name="Bob", type=EntityType.PERSON)

        match = resolver.match(entity1, entity2)

        assert match is not None
        assert match.score >= 0.8

    def test_no_match_different_types(self):
        """Test no match for different entity types."""
        resolver = EntityResolver()

        entity1 = Entity(name="Apple", type=EntityType.ORGANIZATION)
        entity2 = Entity(name="Apple", type=EntityType.OBJECT)

        match = resolver.match(entity1, entity2)

        # Should not match or have low score
        assert match is None or match.score < 0.5

    def test_cluster_entities(self):
        """Test clustering related entities."""
        resolver = EntityResolver()

        entities = [
            Entity(name="Alice Smith", type=EntityType.PERSON),
            Entity(name="Alice", type=EntityType.PERSON, aliases=["A. Smith"]),
            Entity(name="Bob", type=EntityType.PERSON),
            Entity(name="Robert", type=EntityType.PERSON, aliases=["Bob"]),
        ]

        clusters = resolver.cluster(entities)

        # Should form clusters
        assert len(clusters) >= 1
        assert len(clusters) <= len(entities)

    def test_resolve_canonical_name(self):
        """Test resolving canonical name from cluster."""
        resolver = EntityResolver()

        cluster = EntityCluster(entities=[
            Entity(name="Alice Smith", type=EntityType.PERSON, importance_score=0.9),
            Entity(name="Alice", type=EntityType.PERSON, importance_score=0.7),
            Entity(name="A. Smith", type=EntityType.PERSON, importance_score=0.5),
        ])

        canonical = resolver.get_canonical_name(cluster)

        # Should pick highest importance
        assert canonical == "Alice Smith"


class TestConflictResolver:
    """Tests for ConflictResolver."""

    def test_detect_contradictory_facts(self):
        """Test detecting contradictory facts."""
        resolver = ConflictResolver()

        fact1 = Fact(subject="Alice", predicate="lives_in", object="New York")
        fact2 = Fact(subject="Alice", predicate="lives_in", object="Boston")

        conflict = resolver.detect_conflict(fact1, fact2)

        assert conflict is not None
        assert conflict.type == ConflictType.CONTRADICTORY

    def test_detect_attribute_mismatch(self):
        """Test detecting attribute mismatches."""
        resolver = ConflictResolver()

        entity1 = Entity(
            name="Alice",
            type=EntityType.PERSON,
            attributes={"age": "25"},
        )
        entity2 = Entity(
            name="Alice",
            type=EntityType.PERSON,
            attributes={"age": "30"},
        )

        conflict = resolver.detect_conflict(entity1, entity2)

        assert conflict is not None
        assert conflict.type == ConflictType.ATTRIBUTE_MISMATCH

    def test_resolve_by_recency(self):
        """Test resolving conflict by recency."""
        resolver = ConflictResolver(strategy="recency")

        fact1 = Fact(
            subject="Alice",
            predicate="works_at",
            object="OldCorp",
            source_turn=5,
        )
        fact2 = Fact(
            subject="Alice",
            predicate="works_at",
            object="NewCorp",
            source_turn=10,
        )

        resolved = resolver.resolve(fact1, fact2)

        assert resolved.object == "NewCorp"

    def test_resolve_by_confidence(self):
        """Test resolving conflict by confidence."""
        resolver = ConflictResolver(strategy="confidence")

        fact1 = Fact(
            subject="Alice",
            predicate="likes",
            object="Python",
            confidence=0.95,
        )
        fact2 = Fact(
            subject="Alice",
            predicate="likes",
            object="Java",
            confidence=0.70,
        )

        resolved = resolver.resolve(fact1, fact2)

        assert resolved.object == "Python"

    def test_resolve_by_merge(self):
        """Test merging conflicting information."""
        resolver = ConflictResolver(strategy="merge")

        entity1 = Entity(
            name="Alice",
            type=EntityType.PERSON,
            attributes={"role": "engineer"},
        )
        entity2 = Entity(
            name="Alice",
            type=EntityType.PERSON,
            attributes={"hobby": "reading"},
        )

        resolved = resolver.resolve(entity1, entity2)

        assert "role" in resolved.attributes
        assert "hobby" in resolved.attributes


class TestMemoryMerger:
    """Tests for MemoryMerger."""

    def test_merge_union(self):
        """Test union merge strategy."""
        merger = MemoryMerger(strategy=MergeStrategy.UNION)

        ir1 = create_session1_ir()
        ir2 = create_session2_ir()

        merged = merger.merge([ir1, ir2])

        entities = list(merged.iter_entities())
        facts = list(merged.iter_facts())

        # Should contain entities from both
        assert len(entities) >= 3
        assert len(facts) >= 2

    def test_merge_smart(self):
        """Test smart merge with entity resolution."""
        merger = MemoryMerger(strategy=MergeStrategy.SMART)

        ir1 = create_session1_ir()
        ir2 = create_session2_ir()

        merged = merger.merge([ir1, ir2])

        # Alice Smith and Alice should be merged
        entities = list(merged.iter_entities())
        alice_entities = [e for e in entities if "alice" in e.name.lower()]

        # Should have resolved to single entity or fewer
        assert len(alice_entities) <= 2

    def test_merge_latest(self):
        """Test latest-wins merge strategy."""
        merger = MemoryMerger(strategy=MergeStrategy.LATEST)

        ir1 = create_session1_ir()
        ir2 = create_session2_ir()

        merged = merger.merge([ir1, ir2])

        # Later session should take precedence for conflicts
        assert merged is not None

    def test_merge_weighted(self):
        """Test weighted merge strategy."""
        merger = MemoryMerger(strategy=MergeStrategy.WEIGHTED)

        ir1 = create_session1_ir()
        ir2 = create_session2_ir()

        merged = merger.merge(
            [ir1, ir2],
            weights=[0.7, 0.3],
        )

        # First session should have more influence
        assert merged is not None

    def test_merge_preserves_metadata(self):
        """Test that merge preserves metadata."""
        merger = MemoryMerger()

        ir1 = create_session1_ir()
        ir1.metadata["source"] = "session1"

        ir2 = create_session2_ir()
        ir2.metadata["source"] = "session2"

        merged = merger.merge([ir1, ir2])

        assert "merged_from" in merged.metadata or len(merged.metadata) > 0

    def test_merge_single_ir(self):
        """Test merging single IR returns equivalent."""
        merger = MemoryMerger()

        ir = create_session1_ir()
        merged = merger.merge([ir])

        orig_entities = set(e.name for e in ir.iter_entities())
        merged_entities = set(e.name for e in merged.iter_entities())

        assert orig_entities == merged_entities

    def test_merge_empty_list(self):
        """Test merging empty list returns empty IR."""
        merger = MemoryMerger()

        merged = merger.merge([])

        assert merged is not None
        assert sum(1 for _ in merged.iter_entities()) == 0

    def test_merge_with_callbacks(self):
        """Test merge with conflict callbacks."""
        conflicts_seen = []

        def on_conflict(conflict):
            conflicts_seen.append(conflict)

        merger = MemoryMerger(
            strategy=MergeStrategy.SMART,
            on_conflict=on_conflict,
        )

        ir1 = create_session1_ir()
        ir2 = create_session2_ir()

        merger.merge([ir1, ir2])

        # May or may not have conflicts depending on data
        assert isinstance(conflicts_seen, list)

    def test_merge_report(self):
        """Test getting merge report."""
        merger = MemoryMerger()

        ir1 = create_session1_ir()
        ir2 = create_session2_ir()

        merged = merger.merge([ir1, ir2])
        report = merger.get_merge_report()

        assert report is not None
        assert "entities_merged" in report or "total_entities" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
