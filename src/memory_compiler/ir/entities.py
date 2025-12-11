"""Entity representations for Memory IR.

Entities represent the key objects mentioned in dialogues, such as people,
organizations, projects, concepts, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityType(Enum):
    """Types of entities that can be extracted from dialogues."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PROJECT = "project"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    TIME = "time"
    QUANTITY = "quantity"
    OTHER = "other"


@dataclass
class Entity:
    """Represents an entity mentioned in the dialogue.

    Attributes:
        id: Unique identifier for the entity.
        name: Canonical name of the entity.
        type: The type/category of the entity.
        aliases: Alternative names or references to this entity.
        attributes: Key-value pairs of entity properties.
        mentions: List of (turn_index, span) tuples indicating where entity appears.
        embedding: Optional vector embedding of the entity.
        importance_score: Computed importance score for pruning decisions.
        metadata: Additional metadata about the entity.
    """

    id: str
    name: str
    type: EntityType
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    mentions: list[tuple[int, tuple[int, int]]] = field(default_factory=list)
    embedding: list[float] | None = None
    importance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    @property
    def reference_count(self) -> int:
        """Number of times this entity is mentioned."""
        return len(self.mentions)

    @property
    def first_mention_turn(self) -> int | None:
        """Turn index where entity was first mentioned."""
        if not self.mentions:
            return None
        return min(turn for turn, _ in self.mentions)

    @property
    def last_mention_turn(self) -> int | None:
        """Turn index where entity was last mentioned."""
        if not self.mentions:
            return None
        return max(turn for turn, _ in self.mentions)

    def add_mention(self, turn_index: int, span: tuple[int, int]) -> None:
        """Add a mention of this entity."""
        self.mentions.append((turn_index, span))

    def add_alias(self, alias: str) -> None:
        """Add an alternative name for this entity."""
        if alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)

    def merge_with(self, other: Entity) -> None:
        """Merge another entity into this one (for coreference resolution)."""
        for alias in other.aliases:
            self.add_alias(alias)
        self.add_alias(other.name)
        self.mentions.extend(other.mentions)
        self.attributes.update(other.attributes)
        self.metadata.update(other.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "mentions": self.mentions,
            "importance_score": self.importance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create entity from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            type=EntityType(data["type"]),
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
            mentions=[tuple(m) for m in data.get("mentions", [])],
            importance_score=data.get("importance_score", 0.0),
            metadata=data.get("metadata", {}),
        )
