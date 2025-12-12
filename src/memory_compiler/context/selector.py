"""Context selection criteria and filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from memory_compiler.context.optimizer import ContextItem


class SelectionCriterion(Enum):
    """Criteria for selecting context items."""

    IMPORTANCE = "importance"
    RECENCY = "recency"
    RELEVANCE = "relevance"
    TYPE = "type"
    CUSTOM = "custom"


@dataclass
class SelectionCriteria:
    """Collection of selection criteria for filtering context.

    Attributes:
        min_importance: Minimum importance score.
        min_recency: Minimum recency score.
        min_relevance: Minimum relevance score.
        include_types: Item types to include.
        exclude_types: Item types to exclude.
        must_include_ids: IDs that must be included.
        custom_filter: Custom filter function.
    """

    min_importance: float = 0.0
    min_recency: float = 0.0
    min_relevance: float = 0.0
    include_types: Optional[Set[str]] = None
    exclude_types: Optional[Set[str]] = None
    must_include_ids: Set[str] = field(default_factory=set)
    custom_filter: Optional[Callable[[ContextItem], bool]] = None


class ContextSelector:
    """Select context items based on criteria.

    Provides flexible filtering and selection of context items
    before packing into the context window.

    Example:
        >>> selector = ContextSelector()
        >>> criteria = SelectionCriteria(min_importance=0.3, include_types={"entity", "fact"})
        >>> filtered = selector.apply(items, criteria)
    """

    def __init__(self) -> None:
        """Initialize selector."""
        self._criteria_history: List[SelectionCriteria] = []

    def apply(
        self,
        items: List[ContextItem],
        criteria: SelectionCriteria,
    ) -> List[ContextItem]:
        """Apply selection criteria to items.

        Args:
            items: Items to filter.
            criteria: Selection criteria.

        Returns:
            Filtered items.
        """
        self._criteria_history.append(criteria)

        filtered = []
        must_include = set()

        # First pass: identify must-include items
        for item in items:
            if item.id in criteria.must_include_ids:
                must_include.add(item.id)
                filtered.append(item)

        # Second pass: apply criteria
        for item in items:
            if item.id in must_include:
                continue

            if not self._passes_criteria(item, criteria):
                continue

            filtered.append(item)

        logger.debug(f"Selected {len(filtered)}/{len(items)} items")
        return filtered

    def _passes_criteria(
        self,
        item: ContextItem,
        criteria: SelectionCriteria,
    ) -> bool:
        """Check if item passes all criteria."""
        # Importance threshold
        if item.importance < criteria.min_importance:
            return False

        # Recency threshold
        if item.recency < criteria.min_recency:
            return False

        # Relevance threshold
        if item.relevance < criteria.min_relevance:
            return False

        # Type inclusion
        if criteria.include_types and item.item_type not in criteria.include_types:
            return False

        # Type exclusion
        if criteria.exclude_types and item.item_type in criteria.exclude_types:
            return False

        # Custom filter
        if criteria.custom_filter and not criteria.custom_filter(item):
            return False

        return True

    def select_top_k(
        self,
        items: List[ContextItem],
        k: int,
        criterion: SelectionCriterion = SelectionCriterion.IMPORTANCE,
    ) -> List[ContextItem]:
        """Select top K items by a criterion.

        Args:
            items: Items to select from.
            k: Number of items to select.
            criterion: Criterion for ranking.

        Returns:
            Top K items.
        """
        if criterion == SelectionCriterion.IMPORTANCE:
            key_fn = lambda x: x.importance
        elif criterion == SelectionCriterion.RECENCY:
            key_fn = lambda x: x.recency
        elif criterion == SelectionCriterion.RELEVANCE:
            key_fn = lambda x: x.relevance
        else:
            key_fn = lambda x: x.combined_score

        sorted_items = sorted(items, key=key_fn, reverse=True)
        return sorted_items[:k]

    def select_diverse(
        self,
        items: List[ContextItem],
        k: int,
        type_weights: Optional[Dict[str, float]] = None,
    ) -> List[ContextItem]:
        """Select diverse items across types.

        Args:
            items: Items to select from.
            k: Number of items to select.
            type_weights: Optional weights per type.

        Returns:
            Diverse selection of items.
        """
        type_weights = type_weights or {}

        # Group by type
        by_type: Dict[str, List[ContextItem]] = {}
        for item in items:
            by_type.setdefault(item.item_type, []).append(item)

        # Sort each group by importance
        for item_type in by_type:
            by_type[item_type].sort(key=lambda x: x.importance, reverse=True)

        # Round-robin selection with weights
        selected = []
        type_indices: Dict[str, int] = {t: 0 for t in by_type}

        while len(selected) < k:
            added_this_round = False

            for item_type, type_items in by_type.items():
                if len(selected) >= k:
                    break

                idx = type_indices[item_type]
                if idx < len(type_items):
                    # Check weight
                    weight = type_weights.get(item_type, 1.0)
                    if weight > 0 and (len(selected) == 0 or weight >= 0.5):
                        selected.append(type_items[idx])
                        type_indices[item_type] = idx + 1
                        added_this_round = True

            if not added_this_round:
                break

        return selected

    def select_by_query_terms(
        self,
        items: List[ContextItem],
        query: str,
        min_overlap: int = 1,
    ) -> List[ContextItem]:
        """Select items that match query terms.

        Args:
            items: Items to select from.
            query: Query string.
            min_overlap: Minimum word overlap required.

        Returns:
            Items matching query terms.
        """
        query_words = set(query.lower().split())

        selected = []
        for item in items:
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)

            if overlap >= min_overlap:
                selected.append(item)

        return selected

    def deduplicate(
        self,
        items: List[ContextItem],
        similarity_threshold: float = 0.8,
    ) -> List[ContextItem]:
        """Remove near-duplicate items.

        Args:
            items: Items to deduplicate.
            similarity_threshold: Threshold for considering duplicates.

        Returns:
            Deduplicated items.
        """
        if not items:
            return []

        selected = [items[0]]

        for item in items[1:]:
            is_duplicate = False

            for existing in selected:
                sim = self._compute_similarity(item.content, existing.content)
                if sim >= similarity_threshold:
                    is_duplicate = True
                    # Keep higher importance version
                    if item.importance > existing.importance:
                        selected.remove(existing)
                        selected.append(item)
                    break

            if not is_duplicate:
                selected.append(item)

        return selected

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class AdaptiveSelector(ContextSelector):
    """Selector that adapts based on query characteristics."""

    def auto_select(
        self,
        items: List[ContextItem],
        query: str,
        token_budget: int,
    ) -> List[ContextItem]:
        """Automatically select items based on query.

        Args:
            items: Items to select from.
            query: User query.
            token_budget: Available token budget.

        Returns:
            Selected items.
        """
        # Analyze query
        query_type = self._analyze_query(query)

        # Build criteria based on query type
        criteria = self._build_criteria_for_query(query_type)

        # Apply criteria
        filtered = self.apply(items, criteria)

        # Select top items within budget
        filtered.sort(key=lambda x: x.combined_score, reverse=True)

        selected = []
        used_tokens = 0

        for item in filtered:
            if used_tokens + item.tokens <= token_budget:
                selected.append(item)
                used_tokens += item.tokens

        return selected

    def _analyze_query(self, query: str) -> str:
        """Analyze query to determine type."""
        query_lower = query.lower()

        # Question word detection
        if any(w in query_lower for w in ["who", "name", "person"]):
            return "person_focused"
        elif any(w in query_lower for w in ["when", "date", "time"]):
            return "temporal"
        elif any(w in query_lower for w in ["where", "location", "place"]):
            return "location_focused"
        elif any(w in query_lower for w in ["what", "how", "why"]):
            return "explanatory"
        else:
            return "general"

    def _build_criteria_for_query(self, query_type: str) -> SelectionCriteria:
        """Build selection criteria based on query type."""
        if query_type == "person_focused":
            return SelectionCriteria(
                min_importance=0.2,
                include_types={"entity", "fact"},
            )
        elif query_type == "temporal":
            return SelectionCriteria(
                min_recency=0.3,
                include_types={"event", "fact"},
            )
        elif query_type == "location_focused":
            return SelectionCriteria(
                min_importance=0.2,
                include_types={"entity", "fact"},
            )
        else:
            return SelectionCriteria(
                min_importance=0.1,
            )
