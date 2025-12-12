"""SQLite storage backend for Memory IR persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.storage.base import MemoryStore


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based persistent storage for Memory IR.

    Provides durable storage with support for search operations
    and efficient querying.

    Example:
        >>> store = SQLiteMemoryStore("memories.db")
        >>> store.save("session_1", ir, {"user_id": "alice"})
        >>> loaded = store.load("session_1")

    Attributes:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str | Path = "memory_store.db") -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                ir_data TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                importance REAL DEFAULT 0,
                FOREIGN KEY (memory_key) REFERENCES memories(key) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0,
                FOREIGN KEY (memory_key) REFERENCES memories(key) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_memory ON entities(memory_key);
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_memory ON facts(memory_key);
        """)

        conn.commit()
        logger.debug(f"Initialized SQLite database: {self.db_path}")

    def save(
        self,
        key: str,
        ir: MemoryIR,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save Memory IR to database."""
        conn = self._get_connection()

        ir_data = json.dumps(ir.to_dict())
        metadata_str = json.dumps(metadata) if metadata else None

        # Upsert main record
        conn.execute("""
            INSERT OR REPLACE INTO memories (key, ir_data, metadata, updated_at)
            VALUES (?, ?, ?, ?)
        """, (key, ir_data, metadata_str, datetime.now().isoformat()))

        # Clear and rebuild index tables
        conn.execute("DELETE FROM entities WHERE memory_key = ?", (key,))
        conn.execute("DELETE FROM facts WHERE memory_key = ?", (key,))

        # Index entities
        for entity in ir.iter_entities():
            conn.execute("""
                INSERT INTO entities (memory_key, entity_id, name, type, importance)
                VALUES (?, ?, ?, ?, ?)
            """, (key, entity.id, entity.name, entity.type.value, entity.importance_score))

        # Index facts
        for fact in ir.iter_facts():
            conn.execute("""
                INSERT INTO facts (memory_key, fact_id, subject, predicate, object, confidence, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (key, fact.id, fact.subject, fact.predicate, fact.object,
                  fact.confidence, fact.importance_score))

        conn.commit()
        logger.debug(f"Saved Memory IR: {key}")

    def load(self, key: str) -> Optional[MemoryIR]:
        """Load Memory IR from database."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT ir_data FROM memories WHERE key = ?", (key,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        ir_data = json.loads(row["ir_data"])
        return MemoryIR.from_dict(ir_data)

    def delete(self, key: str) -> bool:
        """Delete Memory IR from database."""
        conn = self._get_connection()

        cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted Memory IR: {key}")

        return deleted

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all stored memory keys."""
        conn = self._get_connection()

        if prefix:
            cursor = conn.execute(
                "SELECT key FROM memories WHERE key LIKE ?",
                (f"{prefix}%",)
            )
        else:
            cursor = conn.execute("SELECT key FROM memories")

        return [row["key"] for row in cursor.fetchall()]

    def exists(self, key: str) -> bool:
        """Check if a memory exists."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT 1 FROM memories WHERE key = ?", (key,)
        )
        return cursor.fetchone() is not None

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a stored memory."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT metadata FROM memories WHERE key = ?", (key,)
        )
        row = cursor.fetchone()

        if row and row["metadata"]:
            return json.loads(row["metadata"])
        return None

    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a stored memory."""
        conn = self._get_connection()

        # Merge with existing metadata
        existing = self.get_metadata(key) or {}
        existing.update(metadata)

        cursor = conn.execute("""
            UPDATE memories SET metadata = ?, updated_at = ?
            WHERE key = ?
        """, (json.dumps(existing), datetime.now().isoformat(), key))

        conn.commit()
        return cursor.rowcount > 0

    def search_by_entity(self, entity_name: str) -> List[str]:
        """Search for memories containing an entity."""
        conn = self._get_connection()

        cursor = conn.execute("""
            SELECT DISTINCT memory_key FROM entities
            WHERE name LIKE ?
        """, (f"%{entity_name}%",))

        return [row["memory_key"] for row in cursor.fetchall()]

    def search_by_fact(
        self,
        subject: str,
        predicate: Optional[str] = None,
    ) -> List[str]:
        """Search for memories containing matching facts."""
        conn = self._get_connection()

        if predicate:
            cursor = conn.execute("""
                SELECT DISTINCT memory_key FROM facts
                WHERE subject LIKE ? AND predicate LIKE ?
            """, (f"%{subject}%", f"%{predicate}%"))
        else:
            cursor = conn.execute("""
                SELECT DISTINCT memory_key FROM facts
                WHERE subject LIKE ?
            """, (f"%{subject}%",))

        return [row["memory_key"] for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()

        cursor = conn.execute("SELECT COUNT(*) as count FROM memories")
        num_memories = cursor.fetchone()["count"]

        cursor = conn.execute("SELECT COUNT(*) as count FROM entities")
        num_entities = cursor.fetchone()["count"]

        cursor = conn.execute("SELECT COUNT(*) as count FROM facts")
        num_facts = cursor.fetchone()["count"]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "num_memories": num_memories,
            "total_entities": num_entities,
            "total_facts": num_facts,
            "db_size_mb": db_size / (1024 * 1024),
            "db_path": str(self.db_path),
        }

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        conn = self._get_connection()
        conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
