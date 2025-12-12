"""File-based storage backend for Memory IR persistence."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.storage.base import MemoryStore


class FileMemoryStore(MemoryStore):
    """File-based storage for Memory IR.

    Stores each Memory IR as a separate JSON file in a directory.
    Simple and portable, good for development and small-scale use.

    Example:
        >>> store = FileMemoryStore("./memories")
        >>> store.save("session_1", ir)
        >>> loaded = store.load("session_1")

    Attributes:
        base_dir: Directory for storing memory files.
    """

    def __init__(self, base_dir: str | Path = "./memory_store") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_dir = self.base_dir / ".metadata"
        self._metadata_dir.mkdir(exist_ok=True)

        logger.debug(f"Initialized file store at: {self.base_dir}")

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.base_dir / f"{safe_key}.json"

    def _metadata_path(self, key: str) -> Path:
        """Get metadata file path for a key."""
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._metadata_dir / f"{safe_key}.meta.json"

    def save(
        self,
        key: str,
        ir: MemoryIR,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save Memory IR to file."""
        path = self._key_to_path(key)

        # Save IR data
        ir_data = ir.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ir_data, f, indent=2, ensure_ascii=False)

        # Save metadata
        meta = metadata or {}
        meta["_created_at"] = meta.get("_created_at", datetime.now().isoformat())
        meta["_updated_at"] = datetime.now().isoformat()
        meta["_key"] = key

        meta_path = self._metadata_path(key)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.debug(f"Saved Memory IR to: {path}")

    def load(self, key: str) -> Optional[MemoryIR]:
        """Load Memory IR from file."""
        path = self._key_to_path(key)

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                ir_data = json.load(f)
            return MemoryIR.from_dict(ir_data)
        except Exception as e:
            logger.error(f"Failed to load Memory IR from {path}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete Memory IR file."""
        path = self._key_to_path(key)
        meta_path = self._metadata_path(key)

        deleted = False

        if path.exists():
            path.unlink()
            deleted = True
            logger.debug(f"Deleted Memory IR: {path}")

        if meta_path.exists():
            meta_path.unlink()

        return deleted

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all stored memory keys."""
        keys = []

        for path in self.base_dir.glob("*.json"):
            key = path.stem  # Filename without extension

            if prefix and not key.startswith(prefix):
                continue

            keys.append(key)

        return sorted(keys)

    def exists(self, key: str) -> bool:
        """Check if a memory exists."""
        path = self._key_to_path(key)
        return path.exists()

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a stored memory."""
        meta_path = self._metadata_path(key)

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a stored memory."""
        if not self.exists(key):
            return False

        existing = self.get_metadata(key) or {}
        existing.update(metadata)
        existing["_updated_at"] = datetime.now().isoformat()

        meta_path = self._metadata_path(key)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        return True

    def search_by_entity(self, entity_name: str) -> List[str]:
        """Search for memories containing an entity."""
        matches = []
        entity_name_lower = entity_name.lower()

        for key in self.list_keys():
            ir = self.load(key)
            if ir is None:
                continue

            for entity in ir.iter_entities():
                if entity_name_lower in entity.name.lower():
                    matches.append(key)
                    break

        return matches

    def search_by_fact(
        self,
        subject: str,
        predicate: Optional[str] = None,
    ) -> List[str]:
        """Search for memories containing matching facts."""
        matches = []
        subject_lower = subject.lower()
        predicate_lower = predicate.lower() if predicate else None

        for key in self.list_keys():
            ir = self.load(key)
            if ir is None:
                continue

            for fact in ir.iter_facts():
                if subject_lower not in fact.subject.lower():
                    continue

                if predicate_lower and predicate_lower not in fact.predicate.lower():
                    continue

                matches.append(key)
                break

        return matches

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        keys = self.list_keys()
        total_size = sum(
            self._key_to_path(key).stat().st_size
            for key in keys
            if self._key_to_path(key).exists()
        )

        return {
            "num_memories": len(keys),
            "total_size_mb": total_size / (1024 * 1024),
            "base_dir": str(self.base_dir),
        }

    def export_all(self, output_path: str | Path) -> None:
        """Export all memories to a single JSON file."""
        output_path = Path(output_path)

        all_data = {}
        for key in self.list_keys():
            ir = self.load(key)
            if ir:
                all_data[key] = {
                    "ir": ir.to_dict(),
                    "metadata": self.get_metadata(key),
                }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(all_data)} memories to: {output_path}")

    def import_all(self, input_path: str | Path) -> int:
        """Import memories from an exported JSON file."""
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        count = 0
        for key, data in all_data.items():
            ir = MemoryIR.from_dict(data["ir"])
            metadata = data.get("metadata")
            self.save(key, ir, metadata)
            count += 1

        logger.info(f"Imported {count} memories from: {input_path}")
        return count

    def backup(self, backup_dir: str | Path) -> None:
        """Create a backup of the entire store."""
        backup_dir = Path(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"memory_backup_{timestamp}"

        shutil.copytree(self.base_dir, backup_path)
        logger.info(f"Created backup at: {backup_path}")

    def close(self) -> None:
        """No-op for file store."""
        pass
