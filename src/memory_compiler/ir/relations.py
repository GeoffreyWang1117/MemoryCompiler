"""Relation representations for Memory IR.

Relations describe semantic connections between facts, enabling reasoning
about causality, temporality, and logical entailment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RelationType(Enum):
    """Types of relations between facts."""

    # Temporal relations
    BEFORE = "before"  # Fact A happened before Fact B
    AFTER = "after"  # Fact A happened after Fact B
    DURING = "during"  # Fact A occurred during Fact B
    SIMULTANEOUS = "simultaneous"  # Facts occurred at the same time

    # Causal relations
    CAUSES = "causes"  # Fact A causes Fact B
    CAUSED_BY = "caused_by"  # Fact A is caused by Fact B
    ENABLES = "enables"  # Fact A enables Fact B
    PREVENTS = "prevents"  # Fact A prevents Fact B

    # Logical relations
    ENTAILS = "entails"  # Fact A logically implies Fact B
    CONTRADICTS = "contradicts"  # Fact A contradicts Fact B
    SUPPORTS = "supports"  # Fact A provides evidence for Fact B
    REFUTES = "refutes"  # Fact A provides evidence against Fact B

    # Semantic relations
    ELABORATES = "elaborates"  # Fact A provides more detail about Fact B
    SUMMARIZES = "summarizes"  # Fact A summarizes Fact B
    EQUIVALENT = "equivalent"  # Facts are semantically equivalent
    SIMILAR = "similar"  # Facts are semantically similar

    # Update relations
    UPDATES = "updates"  # Fact A updates/supersedes Fact B
    CORRECTS = "corrects"  # Fact A corrects Fact B

    # Other
    RELATED = "related"  # General relatedness
    OTHER = "other"


@dataclass
class Relation:
    """Represents a relation between two facts.

    Attributes:
        id: Unique identifier for the relation.
        source_fact_id: ID of the source fact.
        target_fact_id: ID of the target fact.
        relation_type: Type of the relation.
        confidence: Confidence score (0-1) in this relation.
        bidirectional: Whether the relation holds in both directions.
        metadata: Additional metadata about the relation.
    """

    id: str
    source_fact_id: str
    target_fact_id: str
    relation_type: RelationType
    confidence: float = 1.0
    bidirectional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return False
        return self.id == other.id

    @property
    def is_temporal(self) -> bool:
        """Check if this is a temporal relation."""
        return self.relation_type in [
            RelationType.BEFORE,
            RelationType.AFTER,
            RelationType.DURING,
            RelationType.SIMULTANEOUS,
        ]

    @property
    def is_causal(self) -> bool:
        """Check if this is a causal relation."""
        return self.relation_type in [
            RelationType.CAUSES,
            RelationType.CAUSED_BY,
            RelationType.ENABLES,
            RelationType.PREVENTS,
        ]

    @property
    def is_logical(self) -> bool:
        """Check if this is a logical relation."""
        return self.relation_type in [
            RelationType.ENTAILS,
            RelationType.CONTRADICTS,
            RelationType.SUPPORTS,
            RelationType.REFUTES,
        ]

    @property
    def is_semantic(self) -> bool:
        """Check if this is a semantic relation."""
        return self.relation_type in [
            RelationType.ELABORATES,
            RelationType.SUMMARIZES,
            RelationType.EQUIVALENT,
            RelationType.SIMILAR,
        ]

    def inverse(self) -> Relation:
        """Create the inverse relation (swap source and target)."""
        inverse_types = {
            RelationType.BEFORE: RelationType.AFTER,
            RelationType.AFTER: RelationType.BEFORE,
            RelationType.CAUSES: RelationType.CAUSED_BY,
            RelationType.CAUSED_BY: RelationType.CAUSES,
            RelationType.ENTAILS: RelationType.ENTAILS,  # Not always invertible
            RelationType.ELABORATES: RelationType.SUMMARIZES,
            RelationType.SUMMARIZES: RelationType.ELABORATES,
        }
        inverse_type = inverse_types.get(self.relation_type, RelationType.RELATED)

        return Relation(
            id=f"{self.id}_inv",
            source_fact_id=self.target_fact_id,
            target_fact_id=self.source_fact_id,
            relation_type=inverse_type,
            confidence=self.confidence,
            bidirectional=self.bidirectional,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert relation to dictionary representation."""
        return {
            "id": self.id,
            "source_fact_id": self.source_fact_id,
            "target_fact_id": self.target_fact_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        """Create relation from dictionary representation."""
        return cls(
            id=data["id"],
            source_fact_id=data["source_fact_id"],
            target_fact_id=data["target_fact_id"],
            relation_type=RelationType(data["relation_type"]),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            metadata=data.get("metadata", {}),
        )


class RelationGraph:
    """Graph structure for managing relations between facts.

    Provides efficient querying of fact relationships.
    """

    def __init__(self) -> None:
        self._relations: dict[str, Relation] = {}
        self._outgoing: dict[str, list[str]] = {}  # fact_id -> [relation_ids]
        self._incoming: dict[str, list[str]] = {}  # fact_id -> [relation_ids]

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the graph."""
        self._relations[relation.id] = relation

        if relation.source_fact_id not in self._outgoing:
            self._outgoing[relation.source_fact_id] = []
        self._outgoing[relation.source_fact_id].append(relation.id)

        if relation.target_fact_id not in self._incoming:
            self._incoming[relation.target_fact_id] = []
        self._incoming[relation.target_fact_id].append(relation.id)

        # Handle bidirectional relations
        if relation.bidirectional:
            if relation.target_fact_id not in self._outgoing:
                self._outgoing[relation.target_fact_id] = []
            self._outgoing[relation.target_fact_id].append(relation.id)

            if relation.source_fact_id not in self._incoming:
                self._incoming[relation.source_fact_id] = []
            self._incoming[relation.source_fact_id].append(relation.id)

    def remove_relation(self, relation_id: str) -> None:
        """Remove a relation from the graph."""
        if relation_id not in self._relations:
            return

        relation = self._relations[relation_id]

        if relation.source_fact_id in self._outgoing:
            self._outgoing[relation.source_fact_id] = [
                r for r in self._outgoing[relation.source_fact_id] if r != relation_id
            ]

        if relation.target_fact_id in self._incoming:
            self._incoming[relation.target_fact_id] = [
                r for r in self._incoming[relation.target_fact_id] if r != relation_id
            ]

        del self._relations[relation_id]

    def get_relations_from(
        self, fact_id: str, relation_type: RelationType | None = None
    ) -> list[Relation]:
        """Get all relations originating from a fact."""
        relation_ids = self._outgoing.get(fact_id, [])
        relations = [self._relations[rid] for rid in relation_ids]

        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]

        return relations

    def get_relations_to(
        self, fact_id: str, relation_type: RelationType | None = None
    ) -> list[Relation]:
        """Get all relations targeting a fact."""
        relation_ids = self._incoming.get(fact_id, [])
        relations = [self._relations[rid] for rid in relation_ids]

        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]

        return relations

    def get_related_facts(self, fact_id: str) -> set[str]:
        """Get all fact IDs related to a given fact."""
        related = set()

        for relation in self.get_relations_from(fact_id):
            related.add(relation.target_fact_id)

        for relation in self.get_relations_to(fact_id):
            related.add(relation.source_fact_id)

        return related

    def find_equivalent_facts(self, fact_id: str) -> set[str]:
        """Find all facts that are semantically equivalent."""
        equivalent = {fact_id}

        for relation in self.get_relations_from(fact_id, RelationType.EQUIVALENT):
            equivalent.add(relation.target_fact_id)

        for relation in self.get_relations_to(fact_id, RelationType.EQUIVALENT):
            equivalent.add(relation.source_fact_id)

        return equivalent

    def all_relations(self) -> list[Relation]:
        """Get all relations in the graph."""
        return list(self._relations.values())

    def __len__(self) -> int:
        return len(self._relations)
