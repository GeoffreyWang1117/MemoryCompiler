"""Tests for memory versioning module."""

import pytest
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.versioning import (
    MemoryDiff,
    DiffOperation,
    compute_diff,
    apply_diff,
    VersionHistory,
    VersionedMemoryStore,
)


def create_base_ir() -> MemoryIR:
    """Create a base Memory IR for testing."""
    ir = MemoryIR(session_id="test_version")
    ir.add_entity(Entity(name="Alice", type=EntityType.PERSON))
    ir.add_entity(Entity(name="Bob", type=EntityType.PERSON))
    ir.add_fact(Fact(subject="Alice", predicate="knows", object="Bob"))
    return ir


def create_modified_ir() -> MemoryIR:
    """Create a modified Memory IR for testing."""
    ir = MemoryIR(session_id="test_version")
    ir.add_entity(Entity(name="Alice", type=EntityType.PERSON))
    ir.add_entity(Entity(name="Charlie", type=EntityType.PERSON))  # Changed
    ir.add_fact(Fact(subject="Alice", predicate="knows", object="Charlie"))  # Changed
    ir.add_fact(Fact(subject="Alice", predicate="works_at", object="TechCorp"))  # Added
    return ir


class TestMemoryDiff:
    """Tests for MemoryDiff."""

    def test_compute_diff_additions(self):
        """Test detecting additions in diff."""
        old_ir = create_base_ir()
        new_ir = create_base_ir()
        new_ir.add_entity(Entity(name="Charlie", type=EntityType.PERSON))

        diff = compute_diff(old_ir, new_ir)

        assert diff is not None
        additions = [op for op in diff.operations if op.operation == DiffOperation.ADD]
        assert len(additions) > 0

    def test_compute_diff_removals(self):
        """Test detecting removals in diff."""
        old_ir = create_base_ir()
        new_ir = MemoryIR(session_id="test")
        new_ir.add_entity(Entity(name="Alice", type=EntityType.PERSON))
        # Bob is removed

        diff = compute_diff(old_ir, new_ir)

        removals = [op for op in diff.operations if op.operation == DiffOperation.REMOVE]
        assert len(removals) > 0

    def test_compute_diff_modifications(self):
        """Test detecting modifications in diff."""
        old_ir = create_base_ir()
        new_ir = create_base_ir()

        # Modify Alice's attributes
        for entity in new_ir.iter_entities():
            if entity.name == "Alice":
                entity.importance_score = 0.9
                entity.attributes["role"] = "engineer"

        diff = compute_diff(old_ir, new_ir)
        assert diff is not None

    def test_empty_diff(self):
        """Test diff between identical IRs."""
        ir = create_base_ir()
        diff = compute_diff(ir, ir)

        assert diff is not None
        assert len(diff.operations) == 0

    def test_diff_summary(self):
        """Test diff summary generation."""
        old_ir = create_base_ir()
        new_ir = create_modified_ir()

        diff = compute_diff(old_ir, new_ir)
        summary = diff.get_summary()

        assert "added" in summary.lower() or "removed" in summary.lower() or "modified" in summary.lower()


class TestApplyDiff:
    """Tests for apply_diff."""

    def test_apply_additions(self):
        """Test applying additions from diff."""
        old_ir = create_base_ir()
        new_ir = create_base_ir()
        new_ir.add_entity(Entity(name="Charlie", type=EntityType.PERSON))

        diff = compute_diff(old_ir, new_ir)
        result = apply_diff(old_ir, diff)

        entity_names = [e.name for e in result.iter_entities()]
        assert "Charlie" in entity_names

    def test_apply_removals(self):
        """Test applying removals from diff."""
        old_ir = create_base_ir()
        new_ir = MemoryIR(session_id="test")
        new_ir.add_entity(Entity(name="Alice", type=EntityType.PERSON))

        diff = compute_diff(old_ir, new_ir)
        result = apply_diff(old_ir, diff)

        entity_names = [e.name for e in result.iter_entities()]
        assert "Bob" not in entity_names

    def test_roundtrip(self):
        """Test that applying diff produces equivalent IR."""
        old_ir = create_base_ir()
        new_ir = create_modified_ir()

        diff = compute_diff(old_ir, new_ir)
        result = apply_diff(old_ir, diff)

        # Result should have same entities as new_ir
        result_entities = set(e.name for e in result.iter_entities())
        new_entities = set(e.name for e in new_ir.iter_entities())
        assert result_entities == new_entities


class TestVersionHistory:
    """Tests for VersionHistory."""

    def test_commit(self):
        """Test committing a version."""
        history = VersionHistory()
        ir = create_base_ir()

        commit_id = history.commit(ir, message="Initial commit")

        assert commit_id is not None
        assert len(commit_id) > 0

    def test_multiple_commits(self):
        """Test multiple commits."""
        history = VersionHistory()

        ir1 = create_base_ir()
        commit1 = history.commit(ir1, message="First commit")

        ir2 = create_modified_ir()
        commit2 = history.commit(ir2, message="Second commit")

        assert commit1 != commit2
        assert history.get_commit_count() == 2

    def test_checkout(self):
        """Test checking out a version."""
        history = VersionHistory()

        ir1 = create_base_ir()
        commit1 = history.commit(ir1, message="First")

        ir2 = create_modified_ir()
        history.commit(ir2, message="Second")

        # Checkout first commit
        restored = history.checkout(commit1)

        entity_names = set(e.name for e in restored.iter_entities())
        expected_names = set(e.name for e in ir1.iter_entities())
        assert entity_names == expected_names

    def test_get_history(self):
        """Test getting commit history."""
        history = VersionHistory()

        history.commit(create_base_ir(), message="First")
        history.commit(create_modified_ir(), message="Second")

        commits = history.get_history()

        assert len(commits) == 2
        assert commits[0].message == "First"
        assert commits[1].message == "Second"

    def test_rollback(self):
        """Test rollback to previous version."""
        history = VersionHistory()

        ir1 = create_base_ir()
        history.commit(ir1, message="First")
        history.commit(create_modified_ir(), message="Second")

        # Rollback to first
        history.rollback(steps=1)
        current = history.get_current()

        entity_names = set(e.name for e in current.iter_entities())
        expected_names = set(e.name for e in ir1.iter_entities())
        assert entity_names == expected_names

    def test_diff_between_commits(self):
        """Test getting diff between commits."""
        history = VersionHistory()

        commit1 = history.commit(create_base_ir(), message="First")
        commit2 = history.commit(create_modified_ir(), message="Second")

        diff = history.diff(commit1, commit2)

        assert diff is not None
        assert len(diff.operations) > 0


class TestVersionedMemoryStore:
    """Tests for VersionedMemoryStore."""

    def test_save_creates_version(self):
        """Test that saving creates a version."""
        store = VersionedMemoryStore()
        ir = create_base_ir()

        store.save("test_key", ir)

        history = store.get_history("test_key")
        assert history.get_commit_count() == 1

    def test_multiple_saves(self):
        """Test multiple saves create versions."""
        store = VersionedMemoryStore()

        store.save("key", create_base_ir())
        store.save("key", create_modified_ir())

        history = store.get_history("key")
        assert history.get_commit_count() == 2

    def test_load_specific_version(self):
        """Test loading a specific version."""
        store = VersionedMemoryStore()

        ir1 = create_base_ir()
        store.save("key", ir1)

        history = store.get_history("key")
        commit1 = history.get_history()[0].commit_id

        store.save("key", create_modified_ir())

        # Load first version
        restored = store.load("key", version=commit1)

        entity_names = set(e.name for e in restored.iter_entities())
        expected_names = set(e.name for e in ir1.iter_entities())
        assert entity_names == expected_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
