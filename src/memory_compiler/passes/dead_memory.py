"""Dead Memory Elimination pass.

This pass identifies and removes memory units that are unlikely to be
referenced in future queries, similar to dead code elimination in compilers.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from memory_compiler.ir.entities import Entity
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import OptimizationPass, PassResult


class DeadMemoryEliminationPass(OptimizationPass):
    """Eliminates memory units unlikely to be referenced.

    This pass uses several heuristics to identify "dead" memory:
    1. Entities mentioned only once and not in recent turns
    2. Facts with low confidence that aren't referenced by other facts
    3. Events that are isolated and not part of important sequences
    4. Information that is superseded by later updates

    Configuration options:
        recency_threshold: Turns must be within this many turns of the end
            to be considered "recent" (default: 10).
        min_entity_mentions: Minimum mentions for an entity to be kept
            (default: 1).
        min_fact_confidence: Minimum confidence for standalone facts
            (default: 0.5).
        remove_superseded: Whether to remove facts superseded by updates
            (default: True).
    """

    def __init__(
        self,
        recency_threshold: int = 10,
        min_entity_mentions: int = 1,
        min_fact_confidence: float = 0.5,
        remove_superseded: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recency_threshold = recency_threshold
        self.min_entity_mentions = min_entity_mentions
        self.min_fact_confidence = min_fact_confidence
        self.remove_superseded = remove_superseded

    def run(self, ir: MemoryIR) -> PassResult:
        """Run dead memory elimination on the IR."""
        result = PassResult()
        num_turns = len(ir.dialogue_turns)

        # Track what to remove
        entities_to_remove: set[str] = set()
        facts_to_remove: set[str] = set()
        events_to_remove: set[str] = set()

        # Step 1: Identify dead entities
        for entity_id, entity in ir.entities.items():
            if self._is_entity_dead(entity, num_turns, ir):
                entities_to_remove.add(entity_id)
                logger.debug(f"Marking entity as dead: {entity.name}")

        # Step 2: Identify dead facts
        for fact_id, fact in ir.facts.items():
            if self._is_fact_dead(fact, num_turns, ir, entities_to_remove):
                facts_to_remove.add(fact_id)
                logger.debug(f"Marking fact as dead: {fact.natural_language}")

        # Step 3: Identify superseded facts
        if self.remove_superseded:
            superseded = self._find_superseded_facts(ir)
            facts_to_remove.update(superseded)
            result.stats["superseded_facts"] = len(superseded)

        # Step 4: Identify dead events
        for event_id, event in ir.events.items():
            if self._is_event_dead(event, num_turns, ir, facts_to_remove):
                events_to_remove.add(event_id)

        # Step 5: Remove dead memory units
        for entity_id in entities_to_remove:
            ir.remove_entity(entity_id)
            result.entities_removed += 1

        for fact_id in facts_to_remove:
            if fact_id in ir.facts:  # May have been removed with entity
                ir.remove_fact(fact_id)
                result.facts_removed += 1

        for event_id in events_to_remove:
            ir.remove_event(event_id)
            result.events_removed += 1

        result.modified = (
            result.entities_removed > 0
            or result.facts_removed > 0
            or result.events_removed > 0
        )

        logger.info(
            f"DeadMemoryElimination: removed {result.entities_removed} entities, "
            f"{result.facts_removed} facts, {result.events_removed} events"
        )

        return result

    def _is_entity_dead(
        self, entity: Entity, num_turns: int, ir: MemoryIR
    ) -> bool:
        """Check if an entity is dead (unlikely to be referenced)."""
        # Keep entities mentioned multiple times
        if entity.reference_count > self.min_entity_mentions:
            return False

        # Keep entities mentioned recently
        if entity.last_mention_turn is not None:
            if num_turns - entity.last_mention_turn <= self.recency_threshold:
                return False

        # Keep entities that are subjects of important facts
        facts_about = ir.get_facts_about(entity.name)
        if any(f.importance_score > 0.7 for f in facts_about):
            return False

        # Keep entities of certain important types
        if entity.type.value in {"person", "organization", "project"}:
            # Higher bar for removing these
            if entity.reference_count >= 1:
                return False

        return True

    def _is_fact_dead(
        self,
        fact: Fact,
        num_turns: int,
        ir: MemoryIR,
        dead_entities: set[str],
    ) -> bool:
        """Check if a fact is dead."""
        # Remove facts about dead entities
        subj_entity = ir.get_entity_by_name(fact.subject)
        obj_entity = ir.get_entity_by_name(fact.object)

        if subj_entity and subj_entity.id in dead_entities:
            return True
        if obj_entity and obj_entity.id in dead_entities:
            return True

        # Remove low-confidence facts not mentioned recently
        if fact.confidence < self.min_fact_confidence:
            if not fact.source_turns:
                return True
            latest_turn = max(fact.source_turns)
            if num_turns - latest_turn > self.recency_threshold:
                return True

        # Keep facts that are referenced by relations
        related_facts = ir.get_related_facts(fact.id)
        if related_facts:
            return False

        return False

    def _is_event_dead(
        self,
        event: Any,
        num_turns: int,
        ir: MemoryIR,
        dead_facts: set[str],
    ) -> bool:
        """Check if an event is dead."""
        # Remove events whose related facts are all dead
        if event.related_facts:
            if all(f in dead_facts for f in event.related_facts):
                return True

        # Remove old events with low importance
        if event.importance_score < 0.3:
            if num_turns - event.turn_index > self.recency_threshold * 2:
                return True

        return False

    def _find_superseded_facts(self, ir: MemoryIR) -> set[str]:
        """Find facts that have been superseded by later updates."""
        superseded: set[str] = set()

        # Group facts by subject-predicate
        fact_groups: dict[tuple[str, str], list[Fact]] = {}
        for fact in ir.iter_facts():
            key = (fact.subject.lower(), fact.predicate.lower())
            if key not in fact_groups:
                fact_groups[key] = []
            fact_groups[key].append(fact)

        # For functional predicates, keep only the latest
        functional_predicates = {
            "is",
            "has name",
            "lives in",
            "works at",
            "is located in",
            "has status",
            "has state",
        }

        for (subj, pred), facts in fact_groups.items():
            if pred in functional_predicates and len(facts) > 1:
                # Sort by source turn (latest first)
                sorted_facts = sorted(
                    facts,
                    key=lambda f: max(f.source_turns) if f.source_turns else 0,
                    reverse=True,
                )

                # Mark all but the latest as superseded
                for fact in sorted_facts[1:]:
                    superseded.add(fact.id)
                    logger.debug(
                        f"Fact superseded: {fact.natural_language} "
                        f"(by {sorted_facts[0].natural_language})"
                    )

        return superseded


class QueryPredictionDeadMemoryPass(OptimizationPass):
    """Advanced dead memory elimination using query prediction.

    This pass uses a model to predict likely future queries and keeps
    only memory that is relevant to predicted queries.

    This is more sophisticated than the basic heuristic approach but
    requires a trained query predictor.
    """

    def __init__(
        self,
        query_predictor: Any = None,
        relevance_threshold: float = 0.3,
        num_predicted_queries: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.query_predictor = query_predictor
        self.relevance_threshold = relevance_threshold
        self.num_predicted_queries = num_predicted_queries

    def run(self, ir: MemoryIR) -> PassResult:
        """Run query-prediction-based dead memory elimination."""
        result = PassResult()

        if self.query_predictor is None:
            logger.warning("No query predictor provided, skipping pass")
            return result

        # Predict likely future queries
        context = self._build_context(ir)
        predicted_queries = self.query_predictor.predict(
            context, num_queries=self.num_predicted_queries
        )

        # Score each fact by relevance to predicted queries
        fact_relevance: dict[str, float] = {}
        for fact in ir.iter_facts():
            max_relevance = 0.0
            for query in predicted_queries:
                relevance = self._compute_relevance(fact, query)
                max_relevance = max(max_relevance, relevance)
            fact_relevance[fact.id] = max_relevance

        # Remove facts with low relevance
        facts_to_remove = [
            fid for fid, rel in fact_relevance.items() if rel < self.relevance_threshold
        ]

        for fact_id in facts_to_remove:
            ir.remove_fact(fact_id)
            result.facts_removed += 1

        result.modified = result.facts_removed > 0
        result.stats["predicted_queries"] = len(predicted_queries)

        return result

    def _build_context(self, ir: MemoryIR) -> str:
        """Build context string for query prediction."""
        lines = []
        for turn in ir.dialogue_turns[-10:]:  # Last 10 turns
            lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(lines)

    def _compute_relevance(self, fact: Fact, query: str) -> float:
        """Compute relevance of a fact to a query."""
        # Simple keyword overlap for now
        query_words = set(query.lower().split())
        fact_words = set(fact.natural_language.lower().split())
        overlap = len(query_words & fact_words)
        return overlap / max(len(query_words), 1)
