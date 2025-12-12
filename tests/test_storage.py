"""Tests for storage backends."""

import tempfile
from pathlib import Path

import pytest

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.storage import FileMemoryStore, SQLiteMemoryStore


def create_test_ir() -> MemoryIR:
    """Create a test Memory IR."""
    ir = MemoryIR(session_id="test_session")

    # Add entities
    ir.add_entity(Entity(id="e1", name="Alice", type=EntityType.PERSON))
    ir.add_entity(Entity(id="e2", name="TechCorp", type=EntityType.ORGANIZATION))

    # Add facts
    ir.add_fact(Fact(
        id="f1",
        subject="Alice",
        predicate="works at",
        object="TechCorp",
    ))
    ir.add_fact(Fact(
        id="f2",
        subject="Alice",
        predicate="is",
        object="engineer",
    ))

    return ir


class TestFileMemoryStore:
    """Tests for FileMemoryStore."""

    def test_save_and_load(self):
        """Test saving and loading Memory IR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(tmpdir)
            ir = create_test_ir()

            # Save
            store.save("test_key", ir, {"user": "test"})

            # Load
            loaded = store.load("test_key")
            assert loaded is not None
            assert len(list(loaded.iter_entities())) == 2
            assert len(list(loaded.iter_facts())) == 2

    def test_delete(self):
        """Test deleting a memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(tmpdir)
            ir = create_test_ir()

            store.save("to_delete", ir)
            assert store.exists("to_delete")

            deleted = store.delete("to_delete")
            assert deleted
            assert not store.exists("to_delete")

    def test_list_keys(self):
        """Test listing memory keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(tmpdir)
            ir = create_test_ir()

            store.save("session_1", ir)
            store.save("session_2", ir)
            store.save("other", ir)

            all_keys = store.list_keys()
            assert len(all_keys) == 3

            session_keys = store.list_keys(prefix="session")
            assert len(session_keys) == 2

    def test_metadata(self):
        """Test metadata operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(tmpdir)
            ir = create_test_ir()

            store.save("meta_test", ir, {"version": 1})
            meta = store.get_metadata("meta_test")
            assert meta["version"] == 1

            store.update_metadata("meta_test", {"version": 2})
            meta = store.get_metadata("meta_test")
            assert meta["version"] == 2

    def test_search_by_entity(self):
        """Test searching by entity name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(tmpdir)
            ir = create_test_ir()

            store.save("search_test", ir)

            matches = store.search_by_entity("Alice")
            assert "search_test" in matches

            no_match = store.search_by_entity("Bob")
            assert len(no_match) == 0

    def test_export_import(self):
        """Test exporting and importing all memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = FileMemoryStore(Path(tmpdir) / "store1")
            ir = create_test_ir()

            store1.save("mem1", ir)
            store1.save("mem2", ir)

            export_path = Path(tmpdir) / "export.json"
            store1.export_all(export_path)

            # Import into new store
            store2 = FileMemoryStore(Path(tmpdir) / "store2")
            count = store2.import_all(export_path)

            assert count == 2
            assert store2.exists("mem1")
            assert store2.exists("mem2")


class TestSQLiteMemoryStore:
    """Tests for SQLiteMemoryStore."""

    def test_save_and_load(self):
        """Test saving and loading Memory IR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)
            ir = create_test_ir()

            store.save("test_key", ir, {"user": "test"})

            loaded = store.load("test_key")
            assert loaded is not None
            assert len(list(loaded.iter_entities())) == 2
            store.close()

    def test_delete(self):
        """Test deleting a memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)
            ir = create_test_ir()

            store.save("to_delete", ir)
            assert store.exists("to_delete")

            deleted = store.delete("to_delete")
            assert deleted
            assert not store.exists("to_delete")
            store.close()

    def test_list_keys_with_prefix(self):
        """Test listing keys with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)
            ir = create_test_ir()

            store.save("user_1_session_1", ir)
            store.save("user_1_session_2", ir)
            store.save("user_2_session_1", ir)

            user1_keys = store.list_keys(prefix="user_1")
            assert len(user1_keys) == 2

            store.close()

    def test_search_by_entity_indexed(self):
        """Test indexed entity search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)
            ir = create_test_ir()

            store.save("indexed_test", ir)

            # Search using index
            matches = store.search_by_entity("Alice")
            assert "indexed_test" in matches

            store.close()

    def test_search_by_fact_indexed(self):
        """Test indexed fact search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)
            ir = create_test_ir()

            store.save("fact_test", ir)

            matches = store.search_by_fact("Alice", "works")
            assert "fact_test" in matches

            store.close()

    def test_get_stats(self):
        """Test storage statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)
            ir = create_test_ir()

            store.save("stats_test", ir)

            stats = store.get_stats()
            assert stats["num_memories"] == 1
            assert stats["total_entities"] == 2
            assert stats["total_facts"] == 2

            store.close()

    def test_update_replaces_indexes(self):
        """Test that updating a memory replaces indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteMemoryStore(db_path)

            # First save
            ir1 = MemoryIR(session_id="test")
            ir1.add_entity(Entity(id="e1", name="Alice", type=EntityType.PERSON))
            store.save("update_test", ir1)

            # Update with different entity
            ir2 = MemoryIR(session_id="test")
            ir2.add_entity(Entity(id="e2", name="Bob", type=EntityType.PERSON))
            store.save("update_test", ir2)

            # Old entity should not be found
            alice_matches = store.search_by_entity("Alice")
            assert "update_test" not in alice_matches

            # New entity should be found
            bob_matches = store.search_by_entity("Bob")
            assert "update_test" in bob_matches

            store.close()


class TestContextManager:
    """Test context manager support."""

    def test_file_store_context(self):
        """Test FileMemoryStore as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with FileMemoryStore(tmpdir) as store:
                ir = create_test_ir()
                store.save("context_test", ir)
                assert store.exists("context_test")

    def test_sqlite_store_context(self):
        """Test SQLiteMemoryStore as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                ir = create_test_ir()
                store.save("context_test", ir)
                assert store.exists("context_test")
