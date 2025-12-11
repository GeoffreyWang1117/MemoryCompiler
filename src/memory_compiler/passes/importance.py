"""Importance Pruning pass.

This pass selects the most important memory units to keep within a token budget,
using a knapsack-style optimization approach.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from memory_compiler.ir.entities import Entity
from memory_compiler.ir.events import Event
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import OptimizationPass, PassResult


class ImportancePruningPass(OptimizationPass):
    """Prunes memory units based on importance scores.

    This pass treats memory selection as a knapsack problem: given a token
    budget, select the subset of memory units that maximizes total importance.

    Importance is computed from multiple signals:
    - Reference frequency: How often is this information referenced?
    - Temporal recency: When was this information last mentioned?
    - Semantic centrality: How connected is this to other information?
    - Emotional/sentiment weight: Does this have emotional significance?

    Configuration options:
        token_budget: Maximum tokens to keep (default: 2048).
        min_importance: Minimum importance to keep a unit (default: 0.1).
        recency_weight: Weight for recency in importance calculation
            (default: 0.3).
        frequency_weight: Weight for reference frequency (default: 0.3).
        centrality_weight: Weight for semantic centrality (default: 0.2).
        type_weight: Weight for entity/fact type importance (default: 0.2).
        use_knapsack: Whether to use knapsack optimization (default: True).
    """

    def __init__(
        self,
        token_budget: int = 2048,
        min_importance: float = 0.1,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.3,
        centrality_weight: float = 0.2,
        type_weight: float = 0.2,
        use_knapsack: bool = True,
        tokens_per_entity: int = 15,
        tokens_per_fact: int = 20,
        tokens_per_event: int = 25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.token_budget = token_budget
        self.min_importance = min_importance
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.centrality_weight = centrality_weight
        self.type_weight = type_weight
        self.use_knapsack = use_knapsack
        self.tokens_per_entity = tokens_per_entity
        self.tokens_per_fact = tokens_per_fact
        self.tokens_per_event = tokens_per_event

    def run(self, ir: MemoryIR) -> PassResult:
        """Run importance pruning on the IR."""
        result = PassResult()

        # Step 1: Compute importance scores for all memory units
        self._compute_importance_scores(ir)

        # Step 2: Estimate token costs
        unit_costs = self._estimate_token_costs(ir)
        total_cost = sum(unit_costs.values())

        result.stats["total_tokens_before"] = total_cost
        result.stats["token_budget"] = self.token_budget

        # If under budget, no pruning needed
        if total_cost <= self.token_budget:
            logger.info(f"ImportancePruning: already under budget ({total_cost} <= {self.token_budget})")
            return result

        # Step 3: Select memory units to keep
        if self.use_knapsack:
            selected = self._knapsack_selection(ir, unit_costs)
        else:
            selected = self._greedy_selection(ir, unit_costs)

        result.stats["units_selected"] = len(selected)

        # Step 4: Remove non-selected units
        all_entities = set(ir.entities.keys())
        all_facts = set(ir.facts.keys())
        all_events = set(ir.events.keys())

        selected_entities = {uid for uid in selected if uid.startswith("ent_")}
        selected_facts = {uid for uid in selected if uid.startswith("fact_")}
        selected_events = {uid for uid in selected if uid.startswith("evt_")}

        # Remove entities not selected
        for entity_id in all_entities - selected_entities:
            ir.remove_entity(entity_id)
            result.entities_removed += 1

        # Remove facts not selected
        for fact_id in all_facts - selected_facts:
            if fact_id in ir.facts:  # May have been removed with entity
                ir.remove_fact(fact_id)
                result.facts_removed += 1

        # Remove events not selected
        for event_id in all_events - selected_events:
            ir.remove_event(event_id)
            result.events_removed += 1

        # Compute final token count
        final_cost = sum(
            unit_costs[uid] for uid in selected if uid in unit_costs
        )
        result.stats["total_tokens_after"] = final_cost

        result.modified = (
            result.entities_removed > 0
            or result.facts_removed > 0
            or result.events_removed > 0
        )

        logger.info(
            f"ImportancePruning: removed {result.entities_removed} entities, "
            f"{result.facts_removed} facts, {result.events_removed} events. "
            f"Tokens: {total_cost} -> {final_cost}"
        )

        return result

    def _compute_importance_scores(self, ir: MemoryIR) -> None:
        """Compute importance scores for all memory units."""
        num_turns = len(ir.dialogue_turns)

        # Compute centrality scores
        try:
            centrality = ir.compute_entity_centrality()
        except Exception:
            centrality = {}

        # Score entities
        for entity in ir.iter_entities():
            score = self._compute_entity_importance(entity, num_turns, centrality)
            entity.importance_score = score

        # Score facts
        for fact in ir.iter_facts():
            score = self._compute_fact_importance(fact, num_turns, ir)
            fact.importance_score = score

        # Score events
        for event in ir.iter_events():
            score = self._compute_event_importance(event, num_turns)
            event.importance_score = score

    def _compute_entity_importance(
        self,
        entity: Entity,
        num_turns: int,
        centrality: dict[str, float],
    ) -> float:
        """Compute importance score for an entity."""
        # Reference frequency score
        freq_score = min(1.0, entity.reference_count / 10)

        # Recency score
        if entity.last_mention_turn is not None and num_turns > 0:
            recency_score = entity.last_mention_turn / num_turns
        else:
            recency_score = 0.0

        # Centrality score
        centrality_score = centrality.get(entity.id, 0.0)

        # Type score (some types are more important)
        type_scores = {
            "person": 1.0,
            "organization": 0.9,
            "project": 0.85,
            "concept": 0.7,
            "location": 0.6,
            "product": 0.7,
            "event": 0.65,
            "time": 0.4,
            "quantity": 0.3,
            "other": 0.5,
        }
        type_score = type_scores.get(entity.type.value, 0.5)

        # Weighted combination
        importance = (
            self.frequency_weight * freq_score
            + self.recency_weight * recency_score
            + self.centrality_weight * centrality_score
            + self.type_weight * type_score
        )

        return importance

    def _compute_fact_importance(
        self, fact: Fact, num_turns: int, ir: MemoryIR
    ) -> float:
        """Compute importance score for a fact."""
        # Base confidence score
        base_score = fact.confidence

        # Frequency score (repeated facts are more important)
        freq_score = min(1.0, len(fact.source_turns) / 5)

        # Recency score
        if fact.source_turns and num_turns > 0:
            latest_turn = max(fact.source_turns)
            recency_score = latest_turn / num_turns
        else:
            recency_score = 0.0

        # Centrality score (facts about important entities)
        subj_entity = ir.get_entity_by_name(fact.subject)
        obj_entity = ir.get_entity_by_name(fact.object)
        centrality_score = 0.0
        if subj_entity:
            centrality_score = max(centrality_score, subj_entity.importance_score)
        if obj_entity:
            centrality_score = max(centrality_score, obj_entity.importance_score)

        # Type score
        type_scores = {
            "attribute": 0.8,
            "relation": 0.85,
            "action": 0.7,
            "state": 0.6,
            "preference": 0.9,
            "belief": 0.85,
            "intention": 0.8,
            "experience": 0.75,
            "possession": 0.6,
            "membership": 0.7,
            "other": 0.5,
        }
        type_score = type_scores.get(fact.fact_type.value, 0.5)

        # Weighted combination
        importance = (
            0.2 * base_score
            + self.frequency_weight * freq_score
            + self.recency_weight * recency_score
            + self.centrality_weight * centrality_score
            + self.type_weight * type_score
        )

        return importance

    def _compute_event_importance(self, event: Event, num_turns: int) -> float:
        """Compute importance score for an event."""
        # Recency score
        if num_turns > 0:
            recency_score = event.turn_index / num_turns
        else:
            recency_score = 0.0

        # Participant score (more participants = more important)
        participant_score = min(1.0, len(event.participants) / 3)

        # Type score
        type_scores = {
            "action": 0.7,
            "state_change": 0.6,
            "decision": 0.9,
            "discovery": 0.85,
            "error": 0.7,
            "resolution": 0.95,
            "question": 0.5,
            "answer": 0.6,
            "request": 0.65,
            "completion": 0.8,
            "other": 0.5,
        }
        type_score = type_scores.get(event.event_type.value, 0.5)

        # Weighted combination
        importance = (
            self.recency_weight * recency_score
            + self.frequency_weight * participant_score
            + self.type_weight * type_score
        )

        return importance

    def _estimate_token_costs(self, ir: MemoryIR) -> dict[str, int]:
        """Estimate token cost for each memory unit."""
        costs = {}

        for entity in ir.iter_entities():
            # Estimate based on name length and attributes
            cost = self.tokens_per_entity
            cost += len(entity.name.split())
            cost += len(entity.aliases) * 2
            cost += len(entity.attributes) * 3
            costs[entity.id] = cost

        for fact in ir.iter_facts():
            # Estimate based on triple length
            cost = self.tokens_per_fact
            cost += len(fact.subject.split())
            cost += len(fact.predicate.split())
            cost += len(fact.object.split())
            costs[fact.id] = cost

        for event in ir.iter_events():
            # Estimate based on description length
            cost = self.tokens_per_event
            cost += len(event.description.split())
            costs[event.id] = cost

        return costs

    def _knapsack_selection(
        self, ir: MemoryIR, unit_costs: dict[str, int]
    ) -> set[str]:
        """Select memory units using 0-1 knapsack optimization."""
        # Collect all items with their values and weights
        items = []  # (id, value, weight)

        for entity in ir.iter_entities():
            items.append((entity.id, entity.importance_score, unit_costs.get(entity.id, 0)))

        for fact in ir.iter_facts():
            items.append((fact.id, fact.importance_score, unit_costs.get(fact.id, 0)))

        for event in ir.iter_events():
            items.append((event.id, event.importance_score, unit_costs.get(event.id, 0)))

        # Filter out items below minimum importance
        items = [(uid, val, wt) for uid, val, wt in items if val >= self.min_importance]

        if not items:
            return set()

        # Use dynamic programming for small instances, greedy for large
        if len(items) * self.token_budget < 1_000_000:
            selected = self._dp_knapsack(items, self.token_budget)
        else:
            # Fall back to greedy for large instances
            selected = self._greedy_knapsack(items, self.token_budget)

        return selected

    def _dp_knapsack(
        self, items: list[tuple[str, float, int]], capacity: int
    ) -> set[str]:
        """Dynamic programming solution to 0-1 knapsack."""
        n = len(items)
        if n == 0 or capacity == 0:
            return set()

        # Scale values to integers for DP
        scale = 1000
        scaled_values = [int(val * scale) for _, val, _ in items]
        weights = [wt for _, _, wt in items]
        ids = [uid for uid, _, _ in items]

        # DP table
        dp = np.zeros((n + 1, capacity + 1), dtype=np.int64)

        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i, w] = max(
                        dp[i - 1, w],
                        dp[i - 1, w - weights[i - 1]] + scaled_values[i - 1],
                    )
                else:
                    dp[i, w] = dp[i - 1, w]

        # Backtrack to find selected items
        selected = set()
        w = capacity
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i - 1, w]:
                selected.add(ids[i - 1])
                w -= weights[i - 1]

        return selected

    def _greedy_knapsack(
        self, items: list[tuple[str, float, int]], capacity: int
    ) -> set[str]:
        """Greedy approximation for knapsack (by value/weight ratio)."""
        # Sort by value/weight ratio
        items_with_ratio = [
            (uid, val, wt, val / max(wt, 1)) for uid, val, wt in items
        ]
        items_with_ratio.sort(key=lambda x: x[3], reverse=True)

        selected = set()
        total_weight = 0

        for uid, val, wt, _ in items_with_ratio:
            if total_weight + wt <= capacity:
                selected.add(uid)
                total_weight += wt

        return selected

    def _greedy_selection(
        self, ir: MemoryIR, unit_costs: dict[str, int]
    ) -> set[str]:
        """Simple greedy selection by importance."""
        # Collect all items
        items = []

        for entity in ir.iter_entities():
            if entity.importance_score >= self.min_importance:
                items.append((entity.id, entity.importance_score, unit_costs.get(entity.id, 0)))

        for fact in ir.iter_facts():
            if fact.importance_score >= self.min_importance:
                items.append((fact.id, fact.importance_score, unit_costs.get(fact.id, 0)))

        for event in ir.iter_events():
            if event.importance_score >= self.min_importance:
                items.append((event.id, event.importance_score, unit_costs.get(event.id, 0)))

        # Sort by importance
        items.sort(key=lambda x: x[1], reverse=True)

        selected = set()
        total_cost = 0

        for uid, _, cost in items:
            if total_cost + cost <= self.token_budget:
                selected.add(uid)
                total_cost += cost

        return selected


class AdaptiveBudgetPruningPass(OptimizationPass):
    """Pruning with adaptive budget based on dialogue complexity.

    This pass adjusts the token budget based on the complexity and
    importance distribution of the dialogue content.
    """

    def __init__(
        self,
        base_budget: int = 2048,
        min_budget_ratio: float = 0.5,
        max_budget_ratio: float = 1.5,
        complexity_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_budget = base_budget
        self.min_budget_ratio = min_budget_ratio
        self.max_budget_ratio = max_budget_ratio
        self.complexity_threshold = complexity_threshold

    def run(self, ir: MemoryIR) -> PassResult:
        """Run adaptive budget pruning."""
        # Compute dialogue complexity
        complexity = self._compute_complexity(ir)

        # Adjust budget based on complexity
        if complexity > self.complexity_threshold:
            ratio = min(self.max_budget_ratio, 1 + (complexity - self.complexity_threshold))
        else:
            ratio = max(self.min_budget_ratio, complexity / self.complexity_threshold)

        adjusted_budget = int(self.base_budget * ratio)

        logger.info(
            f"AdaptiveBudget: complexity={complexity:.2f}, "
            f"budget adjusted from {self.base_budget} to {adjusted_budget}"
        )

        # Run importance pruning with adjusted budget
        pruning_pass = ImportancePruningPass(token_budget=adjusted_budget)
        return pruning_pass.run(ir)

    def _compute_complexity(self, ir: MemoryIR) -> float:
        """Compute dialogue complexity score (0-1)."""
        stats = ir.get_stats()

        # Factors contributing to complexity
        entity_factor = min(1.0, stats.num_entities / 20)
        fact_factor = min(1.0, stats.num_facts / 50)
        relation_factor = min(1.0, stats.num_relations / 30)
        event_factor = min(1.0, stats.num_events / 40)

        # Entity type diversity
        type_diversity = len(stats.entity_types) / 10

        # Weighted combination
        complexity = (
            0.25 * entity_factor
            + 0.30 * fact_factor
            + 0.20 * relation_factor
            + 0.15 * event_factor
            + 0.10 * type_diversity
        )

        return min(1.0, complexity)
