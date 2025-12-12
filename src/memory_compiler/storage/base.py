"""Base storage interface for Memory IR persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from memory_compiler.ir.memory_ir import MemoryIR


class MemoryStore(ABC):
    """Abstract base class for Memory IR storage backends.

    Implementations should provide persistent storage for Memory IR
    instances, supporting CRUD operations and efficient retrieval.
    """

    @abstractmethod
    def save(self, key: str, ir: MemoryIR, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a Memory IR instance.

        Args:
            key: Unique identifier for this memory.
            ir: The Memory IR to save.
            metadata: Optional metadata (e.g., user_id, session_id).
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[MemoryIR]:
        """Load a Memory IR instance.

        Args:
            key: Identifier of the memory to load.

        Returns:
            The loaded MemoryIR or None if not found.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a Memory IR instance.

        Args:
            key: Identifier of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all stored memory keys.

        Args:
            prefix: Optional prefix to filter keys.

        Returns:
            List of memory keys.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a memory exists.

        Args:
            key: Identifier to check.

        Returns:
            True if exists.
        """
        pass

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a stored memory.

        Args:
            key: Identifier of the memory.

        Returns:
            Metadata dictionary or None.
        """
        return None

    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a stored memory.

        Args:
            key: Identifier of the memory.
            metadata: New metadata to merge.

        Returns:
            True if updated successfully.
        """
        return False

    def search_by_entity(self, entity_name: str) -> List[str]:
        """Search for memories containing an entity.

        Args:
            entity_name: Name of the entity to search for.

        Returns:
            List of memory keys containing the entity.
        """
        return []

    def search_by_fact(self, subject: str, predicate: Optional[str] = None) -> List[str]:
        """Search for memories containing matching facts.

        Args:
            subject: Subject to search for.
            predicate: Optional predicate to match.

        Returns:
            List of memory keys containing matching facts.
        """
        return []

    def close(self) -> None:
        """Close the storage connection."""
        pass

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
