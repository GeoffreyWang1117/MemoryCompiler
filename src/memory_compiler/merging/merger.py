"""Main memory merging module for consolidating multiple sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from memory_compiler.ir.entities import Entity
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.events import Event
from memory_compiler.ir.relations import Relation
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.merging.entity_resolver import EntityResolver, EntityMatch, ResolutionStrategy
from memory_compiler.merging.conflict import ConflictResolver, Conflict, ResolutionMethod


class MergeStrategy(Enum):
    """Strategy for merging memories."""

    UNION = "union"  # Keep all unique items
    INTERSECTION = "intersection"  # Keep only common items
    LATEST = "latest"  # Prefer latest session's items
    WEIGHTED = "weighted"  # Weight by importance scores
    SMART = "smart"  # Entity resolution + conflict handling


@dataclass
class MergeConfig:
    """Configuration for memory merging.

    Attributes:
        strategy: Overall merge strategy.
        entity_resolution: Entity resolution strategy.
        similarity_threshold: Threshold for entity matching.
        resolve_conflicts: Whether to automatically resolve conflicts.
        conflict_resolution: Method for resolving conflicts.
        preserve_timeline: Whether to maintain temporal ordering.
        deduplicate_facts: Whether to remove duplicate facts.
        max_entities: Maximum entities in merged result.
        max_facts: Maximum facts in merged result.
    """

    strategy: MergeStrategy = MergeStrategy.SMART
    entity_resolution: ResolutionStrategy = ResolutionStrategy.HYBRID
    similarity_threshold: float = 0.8
    resolve_conflicts: bool = True
    conflict_resolution: ResolutionMethod = ResolutionMethod.KEEP_LATEST
    preserve_timeline: bool = True
    deduplicate_facts: bool = True
    max_entities: Optional[int] = None
    max_facts: Optional[int] = None


@dataclass
class MergeResult:
    """Result of merging multiple Memory IRs.

    Attributes:
        merged_ir: The merged Memory IR.
        source_count: Number of source IRs merged.
        entity_matches: Matches found during entity resolution.
        conflicts: Conflicts detected and resolved.
        stats: Statistics about the merge.
    """

    merged_ir: MemoryIR
    source_count: int
    entity_matches: List[EntityMatch] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_count": self.source_count,
            "entity_matches": [m.to_dict() for m in self.entity_matches],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "stats": self.stats,
        }


class MemoryMerger:
    """Merge multiple Memory IRs into a unified representation.

    Handles entity resolution, fact deduplication, and conflict
    resolution to create a coherent merged memory.

    Example:
        >>> merger = MemoryMerger()
        >>> result = merger.merge([ir1, ir2, ir3])
        >>> print(f"Merged {result.source_count} memories")
    """

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        embed_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize merger.

        Args:
            config: Merge configuration.
            embed_fn: Optional embedding function for semantic matching.
        """
        self.config = config or MergeConfig()
        self.embed_fn = embed_fn

        self.entity_resolver = EntityResolver(
            strategy=self.config.entity_resolution,
            similarity_threshold=self.config.similarity_threshold,
            embed_fn=embed_fn,
        )

        self.conflict_resolver = ConflictResolver(
            default_resolution=self.config.conflict_resolution,
        )

    def merge(
        self,
        memories: List[MemoryIR],
        session_id: Optional[str] = None,
    ) -> MergeResult:
        """Merge multiple Memory IRs.

        Args:
            memories: List of Memory IRs to merge.
            session_id: Session ID for merged result.

        Returns:
            MergeResult with merged IR and metadata.
        """
        if not memories:
            return MergeResult(
                merged_ir=MemoryIR(session_id=session_id or "merged"),
                source_count=0,
            )

        if len(memories) == 1:
            return MergeResult(
                merged_ir=memories[0],
                source_count=1,
            )

        logger.info(f"Merging {len(memories)} Memory IRs")

        if self.config.strategy == MergeStrategy.SMART:
            return self._smart_merge(memories, session_id)
        elif self.config.strategy == MergeStrategy.UNION:
            return self._union_merge(memories, session_id)
        elif self.config.strategy == MergeStrategy.LATEST:
            return self._latest_merge(memories, session_id)
        else:
            return self._union_merge(memories, session_id)

    def _smart_merge(
        self,
        memories: List[MemoryIR],
        session_id: Optional[str],
    ) -> MergeResult:
        """Smart merge with entity resolution and conflict handling."""
        merged_ir = MemoryIR(session_id=session_id or "merged")

        # Collect all items
        all_entities = [list(ir.iter_entities()) for ir in memories]
        all_facts = [list(ir.iter_facts()) for ir in memories]
        all_events = [list(ir.iter_events()) for ir in memories]

        # Step 1: Entity resolution
        merged_entities, entity_matches = self.entity_resolver.resolve_all(all_entities)

        # Build entity ID mapping (old ID -> new ID)
        entity_id_map = self._build_entity_id_map(entity_matches, memories)

        # Add merged entities
        for entity in merged_entities:
            merged_ir.add_entity(entity)

        # Step 2: Merge facts with ID remapping
        flat_facts = [f for facts in all_facts for f in facts]
        remapped_facts = self._remap_fact_entities(flat_facts, entity_id_map)

        # Detect and resolve fact conflicts
        conflicts = []
        if self.config.resolve_conflicts and len(memories) > 1:
            for i in range(len(all_facts)):
                for j in range(i + 1, len(all_facts)):
                    fact_conflicts = self.conflict_resolver.detect_fact_conflicts(
                        all_facts[i], all_facts[j]
                    )
                    conflicts.extend(fact_conflicts)

            conflicts = self.conflict_resolver.resolve_all(conflicts)

        # Add facts (deduplicated)
        if self.config.deduplicate_facts:
            unique_facts = self._deduplicate_facts(remapped_facts)
        else:
            unique_facts = remapped_facts

        for fact in unique_facts:
            merged_ir.add_fact(fact)

        # Step 3: Merge events
        flat_events = [e for events in all_events for e in events]
        for event in flat_events:
            # Remap participant names if needed
            remapped_participants = [
                entity_id_map.get(p, p) for p in event.participants
            ]
            new_event = Event(
                id=event.id,
                description=event.description,
                participants=remapped_participants,
                timestamp=event.timestamp,
                importance_score=event.importance_score,
            )
            merged_ir.add_event(new_event)

        # Step 4: Merge relations (with remapping)
        for memory in memories:
            for relation in memory.relations:
                new_source = entity_id_map.get(relation.source_id, relation.source_id)
                new_target = entity_id_map.get(relation.target_id, relation.target_id)

                new_relation = Relation(
                    source_id=new_source,
                    target_id=new_target,
                    relation_type=relation.relation_type,
                    weight=relation.weight,
                )
                merged_ir.add_relation(new_relation)

        # Step 5: Merge summaries
        for memory in memories:
            for summary in memory.summaries:
                if summary not in merged_ir.summaries:
                    merged_ir.summaries.append(summary)

        # Apply limits if configured
        if self.config.max_entities:
            self._limit_entities(merged_ir, self.config.max_entities)
        if self.config.max_facts:
            self._limit_facts(merged_ir, self.config.max_facts)

        # Compute stats
        stats = {
            "input_entities": sum(len(list(ir.iter_entities())) for ir in memories),
            "output_entities": len(list(merged_ir.iter_entities())),
            "input_facts": sum(len(list(ir.iter_facts())) for ir in memories),
            "output_facts": len(list(merged_ir.iter_facts())),
            "entity_matches": len(entity_matches),
            "conflicts_found": len(conflicts),
        }

        return MergeResult(
            merged_ir=merged_ir,
            source_count=len(memories),
            entity_matches=entity_matches,
            conflicts=conflicts,
            stats=stats,
        )

    def _union_merge(
        self,
        memories: List[MemoryIR],
        session_id: Optional[str],
    ) -> MergeResult:
        """Simple union merge - keep all unique items."""
        merged_ir = MemoryIR(session_id=session_id or "merged")

        seen_entity_names: Set[str] = set()
        seen_fact_keys: Set[str] = set()

        for memory in memories:
            # Add entities
            for entity in memory.iter_entities():
                key = entity.name.lower()
                if key not in seen_entity_names:
                    merged_ir.add_entity(entity)
                    seen_entity_names.add(key)

            # Add facts
            for fact in memory.iter_facts():
                key = f"{fact.subject}|{fact.predicate}|{fact.object}".lower()
                if key not in seen_fact_keys:
                    merged_ir.add_fact(fact)
                    seen_fact_keys.add(key)

            # Add events
            for event in memory.iter_events():
                merged_ir.add_event(event)

            # Add relations
            for relation in memory.relations:
                merged_ir.add_relation(relation)

            # Add summaries
            for summary in memory.summaries:
                if summary not in merged_ir.summaries:
                    merged_ir.summaries.append(summary)

        return MergeResult(
            merged_ir=merged_ir,
            source_count=len(memories),
            stats={
                "output_entities": len(list(merged_ir.iter_entities())),
                "output_facts": len(list(merged_ir.iter_facts())),
            },
        )

    def _latest_merge(
        self,
        memories: List[MemoryIR],
        session_id: Optional[str],
    ) -> MergeResult:
        """Merge preferring latest memory's values."""
        # Process in reverse order so latest values win
        reversed_memories = list(reversed(memories))
        return self._union_merge(reversed_memories, session_id)

    def _build_entity_id_map(
        self,
        matches: List[EntityMatch],
        memories: List[MemoryIR],
    ) -> Dict[str, str]:
        """Build mapping from old entity IDs to merged IDs."""
        id_map: Dict[str, str] = {}

        for match in matches:
            # Map second entity's ID to first (merged) entity's ID
            id_map[match.entity2_id] = match.entity1_id

        return id_map

    def _remap_fact_entities(
        self,
        facts: List[Fact],
        id_map: Dict[str, str],
    ) -> List[Fact]:
        """Remap entity references in facts."""
        remapped = []

        for fact in facts:
            new_subject = id_map.get(fact.subject, fact.subject)
            new_object = id_map.get(fact.object, fact.object)

            remapped_fact = Fact(
                id=fact.id,
                subject=new_subject,
                predicate=fact.predicate,
                object=new_object,
                confidence=fact.confidence,
                context=fact.context,
                source_turn=fact.source_turn,
                importance_score=fact.importance_score,
            )
            remapped.append(remapped_fact)

        return remapped

    def _deduplicate_facts(self, facts: List[Fact]) -> List[Fact]:
        """Remove duplicate facts, keeping highest confidence version."""
        unique: Dict[str, Fact] = {}

        for fact in facts:
            key = f"{fact.subject.lower()}|{fact.predicate.lower()}|{fact.object.lower()}"

            if key not in unique:
                unique[key] = fact
            else:
                # Keep version with higher confidence
                if fact.confidence > unique[key].confidence:
                    unique[key] = fact

        return list(unique.values())

    def _limit_entities(self, ir: MemoryIR, max_count: int) -> None:
        """Limit entities by importance score."""
        entities = sorted(
            ir.iter_entities(),
            key=lambda e: e.importance_score,
            reverse=True,
        )

        to_keep = {e.id for e in entities[:max_count]}

        ir.entities = {
            eid: e for eid, e in ir.entities.items()
            if eid in to_keep
        }

    def _limit_facts(self, ir: MemoryIR, max_count: int) -> None:
        """Limit facts by importance score."""
        facts = sorted(
            ir.iter_facts(),
            key=lambda f: f.importance_score,
            reverse=True,
        )

        to_keep = {f.id for f in facts[:max_count]}

        ir.facts = {
            fid: f for fid, f in ir.facts.items()
            if fid in to_keep
        }

    def merge_into_existing(
        self,
        base_ir: MemoryIR,
        new_ir: MemoryIR,
    ) -> MergeResult:
        """Merge new IR into an existing base IR.

        Args:
            base_ir: Existing base Memory IR.
            new_ir: New Memory IR to merge in.

        Returns:
            MergeResult with updated base IR.
        """
        return self.merge([base_ir, new_ir], session_id=base_ir.session_id)

    def incremental_merge(
        self,
        base_ir: MemoryIR,
        new_dialogue: str,
        extractor: Optional[Callable] = None,
    ) -> MergeResult:
        """Incrementally merge new dialogue into existing memory.

        Args:
            base_ir: Existing Memory IR.
            new_dialogue: New dialogue to extract and merge.
            extractor: Function to extract IR from dialogue.

        Returns:
            MergeResult with updated IR.
        """
        if extractor is None:
            from memory_compiler.pipeline import MemoryCompiler
            compiler = MemoryCompiler(use_mock=True)
            new_ir = compiler.extract(new_dialogue)
        else:
            new_ir = extractor(new_dialogue)

        return self.merge_into_existing(base_ir, new_ir)
