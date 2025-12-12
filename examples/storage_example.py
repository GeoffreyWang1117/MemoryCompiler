#!/usr/bin/env python3
"""Example: Memory storage backends.

This example demonstrates how to use different storage backends
to persist Memory IR for long-term use.
"""

import tempfile
from pathlib import Path

from memory_compiler import MemoryCompiler
from memory_compiler.storage import FileMemoryStore, SQLiteMemoryStore


def main():
    # Initialize compiler
    compiler = MemoryCompiler(use_mock=True)

    # Sample dialogue
    dialogue = """
    User: Hi, I'm Alice and I work as a data scientist at TechCorp.
    Assistant: Nice to meet you, Alice! Data science is an exciting field.
    User: I specialize in natural language processing and machine learning.
    Assistant: NLP and ML are very complementary areas. What projects are you working on?
    User: Currently building a sentiment analysis model for customer reviews.
    Assistant: That sounds interesting! Are you using transformer models?
    User: Yes, I'm fine-tuning BERT for our specific domain.
    """

    # Extract Memory IR
    ir = compiler.compile(dialogue, token_budget=1024)

    print("Memory IR extracted:")
    print(f"  Entities: {len(list(ir.iter_entities()))}")
    print(f"  Facts: {len(list(ir.iter_facts()))}")
    print()

    # Create temp directories for examples
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ========== File Store Example ==========
        print("=" * 60)
        print("File Store Example")
        print("=" * 60)

        file_store = FileMemoryStore(tmpdir / "file_store")

        # Save memory
        file_store.save(
            "alice_session_1",
            ir,
            metadata={"user_id": "alice", "topic": "work"},
        )
        print(f"Saved to file store: alice_session_1")

        # List keys
        keys = file_store.list_keys()
        print(f"Stored keys: {keys}")

        # Load back
        loaded_ir = file_store.load("alice_session_1")
        if loaded_ir:
            print(f"Loaded IR has {len(list(loaded_ir.iter_entities()))} entities")

        # Get metadata
        meta = file_store.get_metadata("alice_session_1")
        print(f"Metadata: {meta}")

        # Search by entity
        matches = file_store.search_by_entity("Alice")
        print(f"Memories with 'Alice': {matches}")

        # Export all
        export_path = tmpdir / "export.json"
        file_store.export_all(export_path)
        print(f"Exported to: {export_path}")

        # Get stats
        stats = file_store.get_stats()
        print(f"Store stats: {stats}")

        file_store.close()
        print()

        # ========== SQLite Store Example ==========
        print("=" * 60)
        print("SQLite Store Example")
        print("=" * 60)

        sqlite_store = SQLiteMemoryStore(tmpdir / "memories.db")

        # Save multiple sessions
        sessions = [
            ("alice_session_1", {"user_id": "alice", "topic": "work"}),
            ("alice_session_2", {"user_id": "alice", "topic": "hobbies"}),
            ("bob_session_1", {"user_id": "bob", "topic": "tech"}),
        ]

        for key, meta in sessions:
            sqlite_store.save(key, ir, metadata=meta)
            print(f"Saved: {key}")

        # List all keys
        all_keys = sqlite_store.list_keys()
        print(f"\nAll keys: {all_keys}")

        # List with prefix
        alice_keys = sqlite_store.list_keys(prefix="alice")
        print(f"Alice's keys: {alice_keys}")

        # Check existence
        exists = sqlite_store.exists("alice_session_1")
        print(f"alice_session_1 exists: {exists}")

        # Search by entity (uses index!)
        matches = sqlite_store.search_by_entity("Alice")
        print(f"\nMemories mentioning 'Alice': {matches}")

        # Search by fact
        fact_matches = sqlite_store.search_by_fact("Alice", "works at")
        print(f"Memories with fact about Alice's work: {fact_matches}")

        # Update metadata
        sqlite_store.update_metadata("alice_session_1", {"priority": "high"})
        updated_meta = sqlite_store.get_metadata("alice_session_1")
        print(f"\nUpdated metadata: {updated_meta}")

        # Get detailed stats
        stats = sqlite_store.get_stats()
        print(f"\nSQLite store stats:")
        print(f"  Memories: {stats['num_memories']}")
        print(f"  Entities indexed: {stats['total_entities']}")
        print(f"  Facts indexed: {stats['total_facts']}")
        print(f"  DB size: {stats['db_size_mb']:.4f} MB")

        # Vacuum for optimization
        sqlite_store.vacuum()
        print("Database vacuumed")

        # Delete a session
        deleted = sqlite_store.delete("bob_session_1")
        print(f"\nDeleted bob_session_1: {deleted}")

        remaining = sqlite_store.list_keys()
        print(f"Remaining keys: {remaining}")

        sqlite_store.close()

    print("\n" + "=" * 60)
    print("Storage examples completed!")


if __name__ == "__main__":
    main()
