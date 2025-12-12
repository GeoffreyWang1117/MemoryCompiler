"""Entity resolution for memory merging.

Handles matching and merging entities across multiple Memory IRs,
resolving cases where the same entity appears with different names or IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from memory_compiler.ir.entities import Entity, EntityType


class ResolutionStrategy(Enum):
    """Strategy for entity resolution."""

    STRICT = "strict"  # Only exact name match
    FUZZY = "fuzzy"  # Allow fuzzy name matching
    SEMANTIC = "semantic"  # Use embedding similarity
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class EntityMatch:
    """A match between two entities.

    Attributes:
        entity1_id: ID of first entity.
        entity2_id: ID of second entity.
        confidence: Match confidence (0-1).
        match_type: Type of match (name, alias, semantic).
        merged_entity: The merged entity result.
    """

    entity1_id: str
    entity2_id: str
    confidence: float
    match_type: str
    merged_entity: Optional[Entity] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity1_id": self.entity1_id,
            "entity2_id": self.entity2_id,
            "confidence": self.confidence,
            "match_type": self.match_type,
            "merged_entity": self.merged_entity.to_dict() if self.merged_entity else None,
        }


class EntityResolver:
    """Resolve and merge entities across Memory IRs.

    Identifies same entities across different sessions and merges
    their attributes, aliases, and importance scores.

    Example:
        >>> resolver = EntityResolver(strategy=ResolutionStrategy.FUZZY)
        >>> matches = resolver.find_matches(entities1, entities2)
        >>> merged = resolver.merge_matched(matches)
    """

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.HYBRID,
        similarity_threshold: float = 0.8,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        """Initialize resolver.

        Args:
            strategy: Resolution strategy.
            similarity_threshold: Minimum similarity for matching.
            embed_fn: Optional embedding function for semantic matching.
        """
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.embed_fn = embed_fn or self._default_embed

    def _default_embed(self, text: str) -> np.ndarray:
        """Default embedding (hash-based for testing)."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(128).astype(np.float32)

    def find_matches(
        self,
        entities1: List[Entity],
        entities2: List[Entity],
    ) -> List[EntityMatch]:
        """Find matching entities between two sets.

        Args:
            entities1: First set of entities.
            entities2: Second set of entities.

        Returns:
            List of entity matches.
        """
        matches = []
        matched2 = set()

        for e1 in entities1:
            best_match = None
            best_score = 0.0
            best_type = ""

            for e2 in entities2:
                if e2.id in matched2:
                    continue

                score, match_type = self._compute_match_score(e1, e2)

                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = e2
                    best_type = match_type

            if best_match:
                matches.append(
                    EntityMatch(
                        entity1_id=e1.id,
                        entity2_id=best_match.id,
                        confidence=best_score,
                        match_type=best_type,
                    )
                )
                matched2.add(best_match.id)

        return matches

    def _compute_match_score(
        self,
        e1: Entity,
        e2: Entity,
    ) -> Tuple[float, str]:
        """Compute match score between two entities."""
        scores = []

        # Exact name match
        if e1.name.lower() == e2.name.lower():
            return 1.0, "exact_name"

        # Type must match for high confidence
        if e1.type != e2.type:
            return 0.0, "type_mismatch"

        if self.strategy in (ResolutionStrategy.FUZZY, ResolutionStrategy.HYBRID):
            # Alias matching
            e1_names = {e1.name.lower()} | {a.lower() for a in e1.aliases}
            e2_names = {e2.name.lower()} | {a.lower() for a in e2.aliases}

            if e1_names & e2_names:
                return 0.95, "alias_match"

            # Fuzzy name matching
            fuzzy_score = self._fuzzy_name_match(e1.name, e2.name)
            if fuzzy_score > 0:
                scores.append((fuzzy_score, "fuzzy_name"))

        if self.strategy in (ResolutionStrategy.SEMANTIC, ResolutionStrategy.HYBRID):
            # Semantic similarity
            semantic_score = self._semantic_similarity(e1, e2)
            if semantic_score > 0:
                scores.append((semantic_score, "semantic"))

        if not scores:
            return 0.0, "no_match"

        # Return highest score
        return max(scores, key=lambda x: x[0])

    def _fuzzy_name_match(self, name1: str, name2: str) -> float:
        """Compute fuzzy match score between names."""
        # Simple character overlap
        n1 = name1.lower().replace(" ", "")
        n2 = name2.lower().replace(" ", "")

        if not n1 or not n2:
            return 0.0

        # Levenshtein-like scoring
        common = sum(1 for c in n1 if c in n2)
        max_len = max(len(n1), len(n2))

        return common / max_len

    def _semantic_similarity(self, e1: Entity, e2: Entity) -> float:
        """Compute semantic similarity between entities."""
        # Build context strings
        ctx1 = f"{e1.name} {' '.join(e1.aliases)} {' '.join(str(v) for v in e1.attributes.values())}"
        ctx2 = f"{e2.name} {' '.join(e2.aliases)} {' '.join(str(v) for v in e2.attributes.values())}"

        emb1 = self.embed_fn(ctx1)
        emb2 = self.embed_fn(ctx2)

        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def merge_entities(
        self,
        e1: Entity,
        e2: Entity,
        prefer_first: bool = True,
    ) -> Entity:
        """Merge two matched entities into one.

        Args:
            e1: First entity.
            e2: Second entity.
            prefer_first: Whether to prefer first entity's values.

        Returns:
            Merged entity.
        """
        # Use primary entity's ID and name
        primary, secondary = (e1, e2) if prefer_first else (e2, e1)

        # Merge aliases
        merged_aliases = primary.aliases | secondary.aliases
        merged_aliases.add(secondary.name)  # Add other's name as alias
        merged_aliases.discard(primary.name)  # Don't include primary name as alias

        # Merge attributes
        merged_attrs = dict(secondary.attributes)
        merged_attrs.update(primary.attributes)  # Primary takes precedence

        # Merge mentions
        merged_mentions = primary.mentions + secondary.mentions

        # Take higher importance
        merged_importance = max(primary.importance_score, secondary.importance_score)

        return Entity(
            id=primary.id,
            name=primary.name,
            type=primary.type,
            aliases=merged_aliases,
            attributes=merged_attrs,
            mentions=merged_mentions,
            importance_score=merged_importance,
        )

    def resolve_all(
        self,
        entity_sets: List[List[Entity]],
    ) -> Tuple[List[Entity], List[EntityMatch]]:
        """Resolve entities across multiple sets.

        Args:
            entity_sets: List of entity lists to resolve.

        Returns:
            Tuple of (merged_entities, all_matches).
        """
        if not entity_sets:
            return [], []

        # Start with first set
        merged = {e.id: e for e in entity_sets[0]}
        all_matches = []

        # Progressively merge with remaining sets
        for entities in entity_sets[1:]:
            matches = self.find_matches(list(merged.values()), entities)
            all_matches.extend(matches)

            matched_ids = set()

            # Merge matched entities
            for match in matches:
                e1 = merged[match.entity1_id]
                e2_list = [e for e in entities if e.id == match.entity2_id]
                if e2_list:
                    e2 = e2_list[0]
                    merged[match.entity1_id] = self.merge_entities(e1, e2)
                    matched_ids.add(match.entity2_id)
                    match.merged_entity = merged[match.entity1_id]

            # Add unmatched entities
            for entity in entities:
                if entity.id not in matched_ids:
                    # Generate new ID to avoid conflicts
                    new_id = f"{entity.id}_merged"
                    while new_id in merged:
                        new_id += "_"

                    new_entity = Entity(
                        id=new_id,
                        name=entity.name,
                        type=entity.type,
                        aliases=entity.aliases,
                        attributes=entity.attributes,
                        mentions=entity.mentions,
                        importance_score=entity.importance_score,
                    )
                    merged[new_id] = new_entity

        return list(merged.values()), all_matches


class EntityCluster:
    """Cluster of entities referring to the same real-world entity."""

    def __init__(self, canonical_id: str) -> None:
        """Initialize cluster with canonical ID."""
        self.canonical_id = canonical_id
        self.members: List[Entity] = []
        self.canonical_entity: Optional[Entity] = None

    def add_member(self, entity: Entity) -> None:
        """Add entity to cluster."""
        self.members.append(entity)

    def compute_canonical(self) -> Entity:
        """Compute canonical entity from cluster members."""
        if not self.members:
            raise ValueError("Cannot compute canonical from empty cluster")

        if len(self.members) == 1:
            self.canonical_entity = self.members[0]
            return self.canonical_entity

        # Use most mentioned name as canonical
        name_counts: Dict[str, int] = {}
        all_aliases: Set[str] = set()
        all_attrs: Dict[str, Any] = {}
        total_importance = 0.0
        total_mentions: List[int] = []

        for entity in self.members:
            # Count name occurrences
            name_counts[entity.name] = name_counts.get(entity.name, 0) + len(entity.mentions)

            # Collect all aliases
            all_aliases.update(entity.aliases)
            all_aliases.add(entity.name)

            # Merge attributes
            all_attrs.update(entity.attributes)

            # Aggregate importance
            total_importance = max(total_importance, entity.importance_score)

            # Collect mentions
            total_mentions.extend(entity.mentions)

        # Choose most common name as canonical
        canonical_name = max(name_counts, key=name_counts.get)
        all_aliases.discard(canonical_name)

        # Use first member's type
        canonical_type = self.members[0].type

        self.canonical_entity = Entity(
            id=self.canonical_id,
            name=canonical_name,
            type=canonical_type,
            aliases=all_aliases,
            attributes=all_attrs,
            mentions=sorted(set(total_mentions)),
            importance_score=total_importance,
        )

        return self.canonical_entity
