"""Context window optimizer for LLM prompt construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.entities import Entity
from memory_compiler.ir.facts import Fact


class PackingStrategy(Enum):
    """Strategy for packing context."""

    GREEDY = "greedy"  # Greedy selection by importance
    KNAPSACK = "knapsack"  # Optimal packing with DP
    QUERY_FOCUSED = "query_focused"  # Prioritize query-relevant items
    BALANCED = "balanced"  # Balance recency, importance, relevance
    HIERARCHICAL = "hierarchical"  # Hierarchical summary structure


@dataclass
class OptimizationConfig:
    """Configuration for context optimization.

    Attributes:
        max_tokens: Maximum context tokens.
        strategy: Packing strategy.
        reserve_tokens: Tokens to reserve for response.
        include_system_prompt: Include space for system prompt.
        system_prompt_tokens: Estimated system prompt size.
        importance_weight: Weight for importance in scoring.
        recency_weight: Weight for recency in scoring.
        relevance_weight: Weight for query relevance.
    """

    max_tokens: int = 4000
    strategy: PackingStrategy = PackingStrategy.BALANCED
    reserve_tokens: int = 500
    include_system_prompt: bool = True
    system_prompt_tokens: int = 200
    importance_weight: float = 0.4
    recency_weight: float = 0.3
    relevance_weight: float = 0.3


@dataclass
class ContextItem:
    """An item that can be included in context.

    Attributes:
        id: Unique identifier.
        content: Text content.
        tokens: Estimated token count.
        importance: Importance score (0-1).
        recency: Recency score (0-1).
        relevance: Query relevance score (0-1).
        item_type: Type of item (entity/fact/event/summary).
        metadata: Additional metadata.
    """

    id: str
    content: str
    tokens: int
    importance: float = 0.5
    recency: float = 0.5
    relevance: float = 0.5
    item_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def combined_score(self) -> float:
        """Get combined score with default weights."""
        return (self.importance + self.recency + self.relevance) / 3

    def score(
        self,
        importance_weight: float = 0.4,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.3,
    ) -> float:
        """Get weighted score."""
        return (
            importance_weight * self.importance
            + recency_weight * self.recency
            + relevance_weight * self.relevance
        )


class ContextOptimizer:
    """Optimize Memory IR for context window packing.

    Selects and arranges memory items to maximize information
    utility within token constraints.

    Example:
        >>> optimizer = ContextOptimizer(OptimizationConfig(max_tokens=4000))
        >>> packed = optimizer.optimize(memory_ir, query="What's the user's job?")
        >>> context_str = packed.to_string()
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            config: Optimization configuration.
            embed_fn: Optional embedding function for relevance.
        """
        self.config = config or OptimizationConfig()
        self.embed_fn = embed_fn

    def optimize(
        self,
        ir: MemoryIR,
        query: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> "PackedContext":
        """Optimize Memory IR for context packing.

        Args:
            ir: Memory IR to optimize.
            query: Optional query for relevance scoring.
            additional_context: Optional extra context to include.

        Returns:
            PackedContext with selected items.
        """
        # Calculate available tokens
        available = self.config.max_tokens - self.config.reserve_tokens
        if self.config.include_system_prompt:
            available -= self.config.system_prompt_tokens

        # Handle additional context
        if additional_context:
            additional_tokens = self._estimate_tokens(additional_context)
            available -= additional_tokens

        # Convert IR to context items
        items = self._ir_to_items(ir, query)

        # Apply packing strategy
        if self.config.strategy == PackingStrategy.GREEDY:
            selected = self._greedy_pack(items, available)
        elif self.config.strategy == PackingStrategy.KNAPSACK:
            selected = self._knapsack_pack(items, available)
        elif self.config.strategy == PackingStrategy.QUERY_FOCUSED:
            selected = self._query_focused_pack(items, available, query)
        elif self.config.strategy == PackingStrategy.HIERARCHICAL:
            selected = self._hierarchical_pack(items, available)
        else:  # BALANCED
            selected = self._balanced_pack(items, available)

        from memory_compiler.context.packer import PackedContext

        return PackedContext(
            items=selected,
            total_tokens=sum(i.tokens for i in selected),
            max_tokens=self.config.max_tokens,
            query=query,
            additional_context=additional_context,
        )

    def _ir_to_items(
        self,
        ir: MemoryIR,
        query: Optional[str] = None,
    ) -> List[ContextItem]:
        """Convert Memory IR to context items."""
        items = []

        # Get query embedding for relevance scoring
        query_embedding = None
        if query and self.embed_fn:
            query_embedding = self.embed_fn(query)

        # Process entities
        entities = list(ir.iter_entities())
        max_turn = self._get_max_turn(ir)

        for entity in entities:
            content = self._format_entity(entity)
            tokens = self._estimate_tokens(content)

            recency = self._compute_recency(entity.mentions, max_turn)
            relevance = self._compute_relevance(content, query, query_embedding)

            items.append(
                ContextItem(
                    id=f"entity:{entity.id}",
                    content=content,
                    tokens=tokens,
                    importance=entity.importance_score,
                    recency=recency,
                    relevance=relevance,
                    item_type="entity",
                    metadata={"name": entity.name, "type": entity.type.value},
                )
            )

        # Process facts
        for fact in ir.iter_facts():
            content = self._format_fact(fact)
            tokens = self._estimate_tokens(content)

            recency = fact.source_turn / max_turn if max_turn > 0 and fact.source_turn else 0.5
            relevance = self._compute_relevance(content, query, query_embedding)

            items.append(
                ContextItem(
                    id=f"fact:{fact.id}",
                    content=content,
                    tokens=tokens,
                    importance=fact.importance_score,
                    recency=recency,
                    relevance=relevance,
                    item_type="fact",
                )
            )

        # Process summaries
        for i, summary in enumerate(ir.summaries):
            tokens = self._estimate_tokens(summary)
            relevance = self._compute_relevance(summary, query, query_embedding)

            items.append(
                ContextItem(
                    id=f"summary:{i}",
                    content=summary,
                    tokens=tokens,
                    importance=0.8,  # Summaries are important
                    recency=0.9,  # Treat as recent
                    relevance=relevance,
                    item_type="summary",
                )
            )

        return items

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text.split()) * 1.3)

    def _format_entity(self, entity: Entity) -> str:
        """Format entity for context."""
        parts = [f"{entity.name} ({entity.type.value})"]

        if entity.aliases:
            parts.append(f"Also known as: {', '.join(entity.aliases)}")

        if entity.attributes:
            attrs = [f"{k}: {v}" for k, v in entity.attributes.items()]
            parts.append(f"Attributes: {'; '.join(attrs)}")

        return " - ".join(parts)

    def _format_fact(self, fact: Fact) -> str:
        """Format fact for context."""
        return f"{fact.subject} {fact.predicate} {fact.object}"

    def _get_max_turn(self, ir: MemoryIR) -> int:
        """Get maximum turn number from IR."""
        max_turn = 0

        for entity in ir.iter_entities():
            if entity.mentions:
                max_turn = max(max_turn, max(entity.mentions))

        for fact in ir.iter_facts():
            if fact.source_turn:
                max_turn = max(max_turn, fact.source_turn)

        return max_turn or 1

    def _compute_recency(self, mentions: List[int], max_turn: int) -> float:
        """Compute recency score from mentions."""
        if not mentions or max_turn == 0:
            return 0.5

        max_mention = max(mentions)
        return max_mention / max_turn

    def _compute_relevance(
        self,
        content: str,
        query: Optional[str],
        query_embedding: Optional[np.ndarray],
    ) -> float:
        """Compute relevance to query."""
        if not query:
            return 0.5

        # Keyword overlap
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        keyword_overlap = len(query_words & content_words) / (len(query_words) + 1)

        # Embedding similarity if available
        if query_embedding is not None and self.embed_fn:
            content_embedding = self.embed_fn(content)
            sim = np.dot(query_embedding, content_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding) + 1e-10
            )
            return 0.5 * keyword_overlap + 0.5 * max(0, sim)

        return keyword_overlap

    def _greedy_pack(
        self,
        items: List[ContextItem],
        available_tokens: int,
    ) -> List[ContextItem]:
        """Greedy packing by importance."""
        sorted_items = sorted(
            items,
            key=lambda x: x.importance,
            reverse=True,
        )

        selected = []
        used_tokens = 0

        for item in sorted_items:
            if used_tokens + item.tokens <= available_tokens:
                selected.append(item)
                used_tokens += item.tokens

        return selected

    def _knapsack_pack(
        self,
        items: List[ContextItem],
        available_tokens: int,
    ) -> List[ContextItem]:
        """Optimal packing using dynamic programming."""
        n = len(items)
        if n == 0:
            return []

        # Scale values to integers for DP
        values = [int(item.combined_score * 1000) for item in items]
        weights = [item.tokens for item in items]

        # DP table
        dp = [[0] * (available_tokens + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(available_tokens + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - weights[i - 1]] + values[i - 1],
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        # Backtrack to find selected items
        selected = []
        w = available_tokens

        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(items[i - 1])
                w -= weights[i - 1]

        return selected

    def _query_focused_pack(
        self,
        items: List[ContextItem],
        available_tokens: int,
        query: Optional[str],
    ) -> List[ContextItem]:
        """Pack prioritizing query relevance."""
        sorted_items = sorted(
            items,
            key=lambda x: (x.relevance, x.importance),
            reverse=True,
        )

        selected = []
        used_tokens = 0

        for item in sorted_items:
            if used_tokens + item.tokens <= available_tokens:
                selected.append(item)
                used_tokens += item.tokens

        return selected

    def _balanced_pack(
        self,
        items: List[ContextItem],
        available_tokens: int,
    ) -> List[ContextItem]:
        """Balanced packing with weighted scoring."""
        sorted_items = sorted(
            items,
            key=lambda x: x.score(
                self.config.importance_weight,
                self.config.recency_weight,
                self.config.relevance_weight,
            ),
            reverse=True,
        )

        selected = []
        used_tokens = 0

        for item in sorted_items:
            if used_tokens + item.tokens <= available_tokens:
                selected.append(item)
                used_tokens += item.tokens

        return selected

    def _hierarchical_pack(
        self,
        items: List[ContextItem],
        available_tokens: int,
    ) -> List[ContextItem]:
        """Hierarchical packing - summaries first, then details."""
        # Group by type
        by_type: Dict[str, List[ContextItem]] = {}
        for item in items:
            by_type.setdefault(item.item_type, []).append(item)

        selected = []
        used_tokens = 0

        # Priority order: summaries -> entities -> facts
        type_order = ["summary", "entity", "fact", "event"]

        for item_type in type_order:
            type_items = by_type.get(item_type, [])
            type_items.sort(key=lambda x: x.importance, reverse=True)

            for item in type_items:
                if used_tokens + item.tokens <= available_tokens:
                    selected.append(item)
                    used_tokens += item.tokens

        return selected
