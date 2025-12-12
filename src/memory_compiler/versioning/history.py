"""Version history and versioned storage for Memory IR."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.storage.base import MemoryStore
from memory_compiler.versioning.diff import MemoryDiff, compute_diff, apply_diff


@dataclass
class MemoryVersion:
    """A versioned snapshot of Memory IR.

    Attributes:
        version_id: Unique version identifier.
        timestamp: When this version was created.
        parent_version: ID of parent version (None for initial).
        ir_hash: Hash of the IR content.
        metadata: Additional version metadata.
        message: Optional commit message.
    """

    version_id: str
    timestamp: float
    parent_version: Optional[str] = None
    ir_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "parent_version": self.parent_version,
            "ir_hash": self.ir_hash,
            "metadata": self.metadata,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryVersion":
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            timestamp=data["timestamp"],
            parent_version=data.get("parent_version"),
            ir_hash=data.get("ir_hash", ""),
            metadata=data.get("metadata", {}),
            message=data.get("message", ""),
        )

    @property
    def datetime(self) -> datetime:
        """Get datetime of version."""
        return datetime.fromtimestamp(self.timestamp)


class VersionHistory:
    """Track version history for a memory.

    Maintains a linked list of versions with diffs between them.
    Supports operations like rollback, branch comparison, etc.

    Example:
        >>> history = VersionHistory("session_1")
        >>> v1 = history.commit(ir_v1, message="Initial version")
        >>> v2 = history.commit(ir_v2, message="Added new facts")
        >>> diff = history.get_diff(v1, v2)
    """

    def __init__(self, memory_key: str) -> None:
        """Initialize version history.

        Args:
            memory_key: Key identifying the memory being versioned.
        """
        self.memory_key = memory_key
        self._versions: Dict[str, MemoryVersion] = {}
        self._snapshots: Dict[str, MemoryIR] = {}
        self._diffs: Dict[Tuple[str, str], MemoryDiff] = {}
        self._head: Optional[str] = None
        self._version_counter = 0

    @property
    def head(self) -> Optional[MemoryVersion]:
        """Get current head version."""
        if self._head is None:
            return None
        return self._versions.get(self._head)

    @property
    def head_ir(self) -> Optional[MemoryIR]:
        """Get Memory IR at head."""
        if self._head is None:
            return None
        return self._snapshots.get(self._head)

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        self._version_counter += 1
        timestamp = int(time.time() * 1000)
        return f"v{self._version_counter}_{timestamp}"

    def _compute_ir_hash(self, ir: MemoryIR) -> str:
        """Compute hash of Memory IR content."""
        content = json.dumps(ir.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def commit(
        self,
        ir: MemoryIR,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryVersion:
        """Commit a new version.

        Args:
            ir: Memory IR to commit.
            message: Commit message.
            metadata: Additional metadata.

        Returns:
            The new version.
        """
        version_id = self._generate_version_id()
        ir_hash = self._compute_ir_hash(ir)

        version = MemoryVersion(
            version_id=version_id,
            timestamp=time.time(),
            parent_version=self._head,
            ir_hash=ir_hash,
            metadata=metadata or {},
            message=message,
        )

        # Store version and snapshot
        self._versions[version_id] = version
        self._snapshots[version_id] = ir

        # Compute diff from parent if exists
        if self._head and self._head in self._snapshots:
            parent_ir = self._snapshots[self._head]
            diff = compute_diff(parent_ir, ir, self._head, version_id)
            self._diffs[(self._head, version_id)] = diff

        self._head = version_id
        logger.debug(f"Committed version {version_id} for {self.memory_key}")

        return version

    def get_version(self, version_id: str) -> Optional[MemoryVersion]:
        """Get a specific version."""
        return self._versions.get(version_id)

    def get_ir(self, version_id: str) -> Optional[MemoryIR]:
        """Get Memory IR at a specific version."""
        return self._snapshots.get(version_id)

    def get_diff(
        self,
        from_version: str,
        to_version: str,
    ) -> Optional[MemoryDiff]:
        """Get diff between two versions.

        Args:
            from_version: Source version ID.
            to_version: Target version ID.

        Returns:
            Diff between versions, or None if not found.
        """
        # Check if we have cached diff
        if (from_version, to_version) in self._diffs:
            return self._diffs[(from_version, to_version)]

        # Compute diff if both versions exist
        from_ir = self._snapshots.get(from_version)
        to_ir = self._snapshots.get(to_version)

        if from_ir and to_ir:
            diff = compute_diff(from_ir, to_ir, from_version, to_version)
            self._diffs[(from_version, to_version)] = diff
            return diff

        return None

    def rollback(self, version_id: str) -> Optional[MemoryIR]:
        """Rollback to a previous version.

        Args:
            version_id: Version to rollback to.

        Returns:
            Memory IR at that version, or None if not found.
        """
        if version_id not in self._versions:
            logger.warning(f"Version {version_id} not found")
            return None

        self._head = version_id
        logger.info(f"Rolled back to version {version_id}")
        return self._snapshots.get(version_id)

    def list_versions(self) -> List[MemoryVersion]:
        """List all versions in chronological order."""
        return sorted(
            self._versions.values(),
            key=lambda v: v.timestamp,
        )

    def get_history(self) -> List[Tuple[MemoryVersion, Optional[MemoryDiff]]]:
        """Get version history with diffs.

        Returns:
            List of (version, diff_from_parent) tuples.
        """
        history = []

        for version in self.list_versions():
            diff = None
            if version.parent_version:
                diff = self._diffs.get((version.parent_version, version.version_id))

            history.append((version, diff))

        return history

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memory_key": self.memory_key,
            "head": self._head,
            "version_counter": self._version_counter,
            "versions": {vid: v.to_dict() for vid, v in self._versions.items()},
            "snapshots": {vid: ir.to_dict() for vid, ir in self._snapshots.items()},
            "diffs": {
                f"{k[0]}:{k[1]}": d.to_dict()
                for k, d in self._diffs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionHistory":
        """Deserialize from dictionary."""
        history = cls(data["memory_key"])
        history._head = data.get("head")
        history._version_counter = data.get("version_counter", 0)

        for vid, vdata in data.get("versions", {}).items():
            history._versions[vid] = MemoryVersion.from_dict(vdata)

        for vid, ir_data in data.get("snapshots", {}).items():
            history._snapshots[vid] = MemoryIR.from_dict(ir_data)

        for key_str, diff_data in data.get("diffs", {}).items():
            parts = key_str.split(":")
            if len(parts) == 2:
                key = (parts[0], parts[1])
                history._diffs[key] = MemoryDiff.from_dict(diff_data)

        return history


class VersionedMemoryStore:
    """Memory store with versioning support.

    Wraps a base MemoryStore to add version tracking for all memories.

    Example:
        >>> base_store = SQLiteMemoryStore("./memories.db")
        >>> versioned = VersionedMemoryStore(base_store)
        >>> versioned.save("session_1", ir, message="Initial")
        >>> versioned.save("session_1", ir_v2, message="Updated")
        >>> versions = versioned.get_history("session_1")
    """

    def __init__(
        self,
        base_store: MemoryStore,
        history_store_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize versioned store.

        Args:
            base_store: Underlying storage backend.
            history_store_path: Path to store version history.
        """
        self.base_store = base_store
        self.history_store_path = Path(history_store_path) if history_store_path else None
        self._histories: Dict[str, VersionHistory] = {}

        if self.history_store_path and self.history_store_path.exists():
            self._load_histories()

    def _load_histories(self) -> None:
        """Load version histories from disk."""
        if not self.history_store_path:
            return

        history_file = self.history_store_path / "version_histories.json"
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, hist_data in data.items():
                self._histories[key] = VersionHistory.from_dict(hist_data)

            logger.info(f"Loaded {len(self._histories)} version histories")

    def _save_histories(self) -> None:
        """Save version histories to disk."""
        if not self.history_store_path:
            return

        self.history_store_path.mkdir(parents=True, exist_ok=True)
        history_file = self.history_store_path / "version_histories.json"

        data = {key: hist.to_dict() for key, hist in self._histories.items()}

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _get_or_create_history(self, key: str) -> VersionHistory:
        """Get or create version history for a key."""
        if key not in self._histories:
            self._histories[key] = VersionHistory(key)
        return self._histories[key]

    def save(
        self,
        key: str,
        ir: MemoryIR,
        metadata: Optional[Dict[str, Any]] = None,
        message: str = "",
    ) -> MemoryVersion:
        """Save Memory IR with version tracking.

        Args:
            key: Memory key.
            ir: Memory IR to save.
            metadata: Additional metadata.
            message: Version message.

        Returns:
            The new version.
        """
        # Save to base store
        self.base_store.save(key, ir, metadata)

        # Create version
        history = self._get_or_create_history(key)
        version = history.commit(ir, message=message, metadata=metadata)

        # Persist histories
        self._save_histories()

        return version

    def load(
        self,
        key: str,
        version_id: Optional[str] = None,
    ) -> Optional[MemoryIR]:
        """Load Memory IR, optionally at specific version.

        Args:
            key: Memory key.
            version_id: Optional version to load.

        Returns:
            Memory IR or None.
        """
        if version_id:
            history = self._histories.get(key)
            if history:
                return history.get_ir(version_id)
            return None

        return self.base_store.load(key)

    def get_history(self, key: str) -> Optional[VersionHistory]:
        """Get version history for a key."""
        return self._histories.get(key)

    def get_versions(self, key: str) -> List[MemoryVersion]:
        """Get all versions for a key."""
        history = self._histories.get(key)
        return history.list_versions() if history else []

    def get_diff(
        self,
        key: str,
        from_version: str,
        to_version: str,
    ) -> Optional[MemoryDiff]:
        """Get diff between versions."""
        history = self._histories.get(key)
        if history:
            return history.get_diff(from_version, to_version)
        return None

    def rollback(self, key: str, version_id: str) -> Optional[MemoryIR]:
        """Rollback a memory to a previous version.

        Args:
            key: Memory key.
            version_id: Version to rollback to.

        Returns:
            Memory IR at that version.
        """
        history = self._histories.get(key)
        if not history:
            return None

        ir = history.rollback(version_id)
        if ir:
            # Update base store
            self.base_store.save(key, ir)
            self._save_histories()

        return ir

    def compare_versions(
        self,
        key: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two versions and return summary.

        Args:
            key: Memory key.
            version1: First version ID.
            version2: Second version ID.

        Returns:
            Comparison summary.
        """
        diff = self.get_diff(key, version1, version2)
        if not diff:
            return {"error": "Could not compute diff"}

        return {
            "from_version": version1,
            "to_version": version2,
            "additions": diff.num_additions,
            "removals": diff.num_removals,
            "modifications": diff.num_modifications,
            "summary": diff.summary(),
            "diff": diff.to_dict(),
        }

    def delete(self, key: str) -> bool:
        """Delete a memory and its version history."""
        success = self.base_store.delete(key)

        if key in self._histories:
            del self._histories[key]
            self._save_histories()

        return success

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all memory keys."""
        return self.base_store.list_keys(prefix)

    def close(self) -> None:
        """Close the store."""
        self._save_histories()
        self.base_store.close()

    def __enter__(self) -> "VersionedMemoryStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
