"""Conflict detection and resolution for memory merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from memory_compiler.ir.facts import Fact
from memory_compiler.ir.entities import Entity


class ConflictType(Enum):
    """Types of conflicts that can occur during merging."""

    CONTRADICTORY_FACTS = "contradictory_facts"
    ATTRIBUTE_MISMATCH = "attribute_mismatch"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    DUPLICATE_FACT = "duplicate_fact"
    ENTITY_TYPE_CONFLICT = "entity_type_conflict"


class ResolutionMethod(Enum):
    """Methods for resolving conflicts."""

    KEEP_FIRST = "keep_first"
    KEEP_LATEST = "keep_latest"
    KEEP_BOTH = "keep_both"
    MERGE = "merge"
    DROP = "drop"
    MANUAL = "manual"


@dataclass
class Conflict:
    """Represents a conflict between items.

    Attributes:
        conflict_type: Type of conflict.
        item1: First conflicting item.
        item2: Second conflicting item.
        description: Human-readable description.
        resolution: How conflict was resolved.
        resolved_item: Result after resolution.
    """

    conflict_type: ConflictType
    item1: Any
    item2: Any
    description: str
    resolution: Optional[ResolutionMethod] = None
    resolved_item: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_type": self.conflict_type.value,
            "item1": str(self.item1),
            "item2": str(self.item2),
            "description": self.description,
            "resolution": self.resolution.value if self.resolution else None,
            "resolved_item": str(self.resolved_item) if self.resolved_item else None,
            "metadata": self.metadata,
        }


class ConflictResolver:
    """Detect and resolve conflicts during memory merging.

    Handles various types of conflicts that can arise when merging
    multiple Memory IRs, including contradictory facts and attribute
    mismatches.

    Example:
        >>> resolver = ConflictResolver()
        >>> conflicts = resolver.detect_conflicts(facts1, facts2)
        >>> resolved = resolver.resolve_all(conflicts)
    """

    def __init__(
        self,
        default_resolution: ResolutionMethod = ResolutionMethod.KEEP_LATEST,
        custom_resolvers: Optional[Dict[ConflictType, Callable]] = None,
    ) -> None:
        """Initialize resolver.

        Args:
            default_resolution: Default method for resolving conflicts.
            custom_resolvers: Custom resolution functions per conflict type.
        """
        self.default_resolution = default_resolution
        self.custom_resolvers = custom_resolvers or {}

    def detect_fact_conflicts(
        self,
        facts1: List[Fact],
        facts2: List[Fact],
    ) -> List[Conflict]:
        """Detect conflicts between two sets of facts.

        Args:
            facts1: First set of facts.
            facts2: Second set of facts.

        Returns:
            List of detected conflicts.
        """
        conflicts = []

        # Build index by subject-predicate
        fact_index: Dict[Tuple[str, str], List[Fact]] = {}
        for fact in facts1 + facts2:
            key = (fact.subject.lower(), fact.predicate.lower())
            fact_index.setdefault(key, []).append(fact)

        # Find conflicts
        for key, related_facts in fact_index.items():
            if len(related_facts) < 2:
                continue

            # Check for contradictions within this subject-predicate pair
            for i, f1 in enumerate(related_facts):
                for f2 in related_facts[i + 1:]:
                    conflict = self._check_fact_conflict(f1, f2)
                    if conflict:
                        conflicts.append(conflict)

        return conflicts

    def _check_fact_conflict(
        self,
        f1: Fact,
        f2: Fact,
    ) -> Optional[Conflict]:
        """Check if two facts conflict."""
        # Same subject and predicate
        if (
            f1.subject.lower() == f2.subject.lower()
            and f1.predicate.lower() == f2.predicate.lower()
        ):
            # Check if objects differ
            if f1.object.lower() == f2.object.lower():
                # Duplicate, not conflict
                return Conflict(
                    conflict_type=ConflictType.DUPLICATE_FACT,
                    item1=f1,
                    item2=f2,
                    description=f"Duplicate fact: {f1.subject} {f1.predicate} {f1.object}",
                )

            # Check for contradictory predicates
            if self._is_contradictory_predicate(f1.predicate):
                return Conflict(
                    conflict_type=ConflictType.CONTRADICTORY_FACTS,
                    item1=f1,
                    item2=f2,
                    description=(
                        f"Contradictory facts: "
                        f"'{f1.subject} {f1.predicate} {f1.object}' vs "
                        f"'{f2.subject} {f2.predicate} {f2.object}'"
                    ),
                )

        return None

    def _is_contradictory_predicate(self, predicate: str) -> bool:
        """Check if predicate implies unique value (contradictions possible)."""
        unique_predicates = {
            "is", "has age", "was born", "lives in", "works at",
            "is married to", "has job", "has name", "is called",
        }
        return any(p in predicate.lower() for p in unique_predicates)

    def detect_entity_conflicts(
        self,
        entities1: List[Entity],
        entities2: List[Entity],
    ) -> List[Conflict]:
        """Detect conflicts between entity sets.

        Args:
            entities1: First set of entities.
            entities2: Second set of entities.

        Returns:
            List of detected conflicts.
        """
        conflicts = []

        # Build name index
        name_to_entities: Dict[str, List[Entity]] = {}
        for entity in entities1 + entities2:
            name_to_entities.setdefault(entity.name.lower(), []).append(entity)

        # Check for type conflicts
        for name, entities in name_to_entities.items():
            if len(entities) < 2:
                continue

            types = set(e.type for e in entities)
            if len(types) > 1:
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.ENTITY_TYPE_CONFLICT,
                        item1=entities[0],
                        item2=entities[1],
                        description=f"Entity '{name}' has conflicting types: {types}",
                    )
                )

            # Check for attribute conflicts
            attr_conflict = self._check_attribute_conflicts(entities)
            if attr_conflict:
                conflicts.append(attr_conflict)

        return conflicts

    def _check_attribute_conflicts(
        self,
        entities: List[Entity],
    ) -> Optional[Conflict]:
        """Check for attribute conflicts among entities with same name."""
        all_attrs: Dict[str, List[Tuple[Entity, Any]]] = {}

        for entity in entities:
            for key, value in entity.attributes.items():
                all_attrs.setdefault(key, []).append((entity, value))

        for attr_key, values in all_attrs.items():
            if len(values) < 2:
                continue

            # Check for conflicting values
            unique_values = set(str(v) for _, v in values)
            if len(unique_values) > 1:
                return Conflict(
                    conflict_type=ConflictType.ATTRIBUTE_MISMATCH,
                    item1=values[0][0],
                    item2=values[1][0],
                    description=f"Attribute '{attr_key}' has conflicting values: {unique_values}",
                    metadata={"attribute": attr_key, "values": list(unique_values)},
                )

        return None

    def resolve(self, conflict: Conflict) -> Conflict:
        """Resolve a single conflict.

        Args:
            conflict: Conflict to resolve.

        Returns:
            Conflict with resolution applied.
        """
        # Check for custom resolver
        if conflict.conflict_type in self.custom_resolvers:
            resolver_fn = self.custom_resolvers[conflict.conflict_type]
            return resolver_fn(conflict)

        # Apply default resolution based on type
        if conflict.conflict_type == ConflictType.DUPLICATE_FACT:
            return self._resolve_duplicate(conflict)

        elif conflict.conflict_type == ConflictType.CONTRADICTORY_FACTS:
            return self._resolve_contradictory_facts(conflict)

        elif conflict.conflict_type == ConflictType.ATTRIBUTE_MISMATCH:
            return self._resolve_attribute_mismatch(conflict)

        elif conflict.conflict_type == ConflictType.ENTITY_TYPE_CONFLICT:
            return self._resolve_type_conflict(conflict)

        else:
            # Default: keep first
            conflict.resolution = ResolutionMethod.KEEP_FIRST
            conflict.resolved_item = conflict.item1
            return conflict

    def _resolve_duplicate(self, conflict: Conflict) -> Conflict:
        """Resolve duplicate fact conflict."""
        conflict.resolution = ResolutionMethod.MERGE

        f1, f2 = conflict.item1, conflict.item2

        # Merge by taking higher confidence
        if hasattr(f1, 'confidence') and hasattr(f2, 'confidence'):
            if f1.confidence >= f2.confidence:
                conflict.resolved_item = f1
            else:
                conflict.resolved_item = f2
        else:
            conflict.resolved_item = f1

        return conflict

    def _resolve_contradictory_facts(self, conflict: Conflict) -> Conflict:
        """Resolve contradictory facts."""
        f1, f2 = conflict.item1, conflict.item2

        if self.default_resolution == ResolutionMethod.KEEP_FIRST:
            conflict.resolution = ResolutionMethod.KEEP_FIRST
            conflict.resolved_item = f1

        elif self.default_resolution == ResolutionMethod.KEEP_LATEST:
            # Use higher confidence or importance as proxy for "latest"
            if getattr(f2, 'importance_score', 0) > getattr(f1, 'importance_score', 0):
                conflict.resolved_item = f2
            else:
                conflict.resolved_item = f1
            conflict.resolution = ResolutionMethod.KEEP_LATEST

        elif self.default_resolution == ResolutionMethod.KEEP_BOTH:
            conflict.resolution = ResolutionMethod.KEEP_BOTH
            conflict.resolved_item = [f1, f2]

        elif self.default_resolution == ResolutionMethod.DROP:
            conflict.resolution = ResolutionMethod.DROP
            conflict.resolved_item = None

        else:
            conflict.resolution = ResolutionMethod.KEEP_FIRST
            conflict.resolved_item = f1

        return conflict

    def _resolve_attribute_mismatch(self, conflict: Conflict) -> Conflict:
        """Resolve attribute mismatch."""
        conflict.resolution = ResolutionMethod.MERGE

        e1, e2 = conflict.item1, conflict.item2

        # Merge attributes, preferring first
        merged_attrs = dict(e2.attributes)
        merged_attrs.update(e1.attributes)

        # Create merged entity
        from memory_compiler.ir.entities import Entity
        conflict.resolved_item = Entity(
            id=e1.id,
            name=e1.name,
            type=e1.type,
            aliases=e1.aliases | e2.aliases,
            attributes=merged_attrs,
            importance_score=max(e1.importance_score, e2.importance_score),
        )

        return conflict

    def _resolve_type_conflict(self, conflict: Conflict) -> Conflict:
        """Resolve entity type conflict."""
        e1, e2 = conflict.item1, conflict.item2

        # Keep type of entity with higher importance
        if e1.importance_score >= e2.importance_score:
            conflict.resolved_item = e1
        else:
            conflict.resolved_item = e2

        conflict.resolution = ResolutionMethod.KEEP_FIRST
        return conflict

    def resolve_all(self, conflicts: List[Conflict]) -> List[Conflict]:
        """Resolve all conflicts.

        Args:
            conflicts: List of conflicts to resolve.

        Returns:
            List of resolved conflicts.
        """
        return [self.resolve(c) for c in conflicts]

    def get_resolution_summary(
        self,
        conflicts: List[Conflict],
    ) -> Dict[str, Any]:
        """Get summary of resolutions.

        Args:
            conflicts: List of resolved conflicts.

        Returns:
            Summary statistics.
        """
        by_type: Dict[str, int] = {}
        by_resolution: Dict[str, int] = {}

        for conflict in conflicts:
            type_key = conflict.conflict_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            if conflict.resolution:
                res_key = conflict.resolution.value
                by_resolution[res_key] = by_resolution.get(res_key, 0) + 1

        return {
            "total_conflicts": len(conflicts),
            "by_type": by_type,
            "by_resolution": by_resolution,
        }
