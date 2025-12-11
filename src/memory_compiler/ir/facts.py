"""Fact representations for Memory IR.

Facts represent information stated in dialogues using subject-predicate-object triples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FactType(Enum):
    """Types of facts that can be extracted."""

    ATTRIBUTE = "attribute"  # X has property Y
    RELATION = "relation"  # X is related to Y
    ACTION = "action"  # X does/did Y
    STATE = "state"  # X is in state Y
    PREFERENCE = "preference"  # X prefers/likes Y
    BELIEF = "belief"  # X believes Y
    INTENTION = "intention"  # X intends to Y
    EXPERIENCE = "experience"  # X experienced Y
    POSSESSION = "possession"  # X has/owns Y
    MEMBERSHIP = "membership"  # X belongs to Y
    OTHER = "other"


@dataclass
class Fact:
    """Represents a fact extracted from dialogue as a triple.

    Attributes:
        id: Unique identifier for the fact.
        subject: The subject entity or string.
        predicate: The relationship or property.
        object: The object entity or string.
        fact_type: Category of the fact.
        confidence: Confidence score (0-1) in the extraction.
        source_turns: List of turn indices where this fact was stated/implied.
        negated: Whether this is a negated fact.
        temporal_scope: Time period this fact applies to (past/present/future/always).
        embedding: Optional vector embedding of the fact.
        importance_score: Computed importance for pruning decisions.
        is_derived: Whether this fact was derived/inferred vs directly stated.
        supporting_facts: IDs of facts that support this derived fact.
        metadata: Additional metadata.
    """

    id: str
    subject: str
    predicate: str
    object: str
    fact_type: FactType = FactType.OTHER
    confidence: float = 1.0
    source_turns: list[int] = field(default_factory=list)
    negated: bool = False
    temporal_scope: str = "present"
    embedding: list[float] | None = None
    importance_score: float = 0.0
    is_derived: bool = False
    supporting_facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return False
        return self.id == other.id

    @property
    def triple(self) -> tuple[str, str, str]:
        """Return the fact as a (subject, predicate, object) triple."""
        return (self.subject, self.predicate, self.object)

    @property
    def natural_language(self) -> str:
        """Convert fact to natural language string."""
        negation = "not " if self.negated else ""
        return f"{self.subject} {negation}{self.predicate} {self.object}"

    def semantic_similarity(self, other: Fact) -> float:
        """Compute semantic similarity with another fact.

        Returns a score between 0 and 1.
        """
        if self.embedding is None or other.embedding is None:
            # Fall back to string matching if no embeddings
            score = 0.0
            if self.subject.lower() == other.subject.lower():
                score += 0.4
            if self.predicate.lower() == other.predicate.lower():
                score += 0.3
            if self.object.lower() == other.object.lower():
                score += 0.3
            return score

        # Cosine similarity
        import numpy as np

        a = np.array(self.embedding)
        b = np.array(other.embedding)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def entails(self, other: Fact) -> bool:
        """Check if this fact entails another fact.

        This is a simplified check - full entailment would require reasoning.
        """
        # Same subject and predicate, with this fact being more specific
        if self.subject == other.subject and self.predicate == other.predicate:
            if self.object == other.object:
                return True
        return False

    def contradicts(self, other: Fact) -> bool:
        """Check if this fact contradicts another fact."""
        if self.subject == other.subject and self.predicate == other.predicate:
            if self.negated != other.negated and self.object == other.object:
                return True
            # Different objects for same subject-predicate (for functional predicates)
            if self.object != other.object and self.predicate in [
                "is",
                "has name",
                "was born in",
                "lives in",
            ]:
                return True
        return False

    def merge_with(self, other: Fact) -> Fact:
        """Merge with another semantically equivalent fact."""
        merged = Fact(
            id=self.id,
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            fact_type=self.fact_type,
            confidence=max(self.confidence, other.confidence),
            source_turns=sorted(set(self.source_turns + other.source_turns)),
            negated=self.negated,
            temporal_scope=self.temporal_scope,
            importance_score=max(self.importance_score, other.importance_score),
            metadata={**self.metadata, **other.metadata},
        )
        return merged

    def to_dict(self) -> dict[str, Any]:
        """Convert fact to dictionary representation."""
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "fact_type": self.fact_type.value,
            "confidence": self.confidence,
            "source_turns": self.source_turns,
            "negated": self.negated,
            "temporal_scope": self.temporal_scope,
            "importance_score": self.importance_score,
            "is_derived": self.is_derived,
            "supporting_facts": self.supporting_facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fact:
        """Create fact from dictionary representation."""
        return cls(
            id=data["id"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            fact_type=FactType(data.get("fact_type", "other")),
            confidence=data.get("confidence", 1.0),
            source_turns=data.get("source_turns", []),
            negated=data.get("negated", False),
            temporal_scope=data.get("temporal_scope", "present"),
            importance_score=data.get("importance_score", 0.0),
            is_derived=data.get("is_derived", False),
            supporting_facts=data.get("supporting_facts", []),
            metadata=data.get("metadata", {}),
        )
