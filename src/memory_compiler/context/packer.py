"""Context packing and formatting for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from memory_compiler.context.optimizer import ContextItem


class ContextFormat(Enum):
    """Format for packed context output."""

    PLAIN = "plain"  # Plain text
    STRUCTURED = "structured"  # With headers and sections
    XML = "xml"  # XML-style tags
    MARKDOWN = "markdown"  # Markdown formatting
    JSON = "json"  # JSON structure


@dataclass
class PackedContext:
    """Packed context ready for prompt injection.

    Attributes:
        items: Selected context items.
        total_tokens: Total token count.
        max_tokens: Maximum token budget.
        query: Optional query that influenced selection.
        additional_context: Any additional context.
        metadata: Additional metadata.
    """

    items: List[ContextItem]
    total_tokens: int
    max_tokens: int
    query: Optional[str] = None
    additional_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Get token utilization (0-1)."""
        return self.total_tokens / self.max_tokens if self.max_tokens > 0 else 0

    @property
    def item_count(self) -> int:
        """Get number of items."""
        return len(self.items)

    def to_string(
        self,
        format: ContextFormat = ContextFormat.STRUCTURED,
        include_metadata: bool = False,
    ) -> str:
        """Convert to string for prompt injection.

        Args:
            format: Output format.
            include_metadata: Whether to include metadata.

        Returns:
            Formatted context string.
        """
        if format == ContextFormat.PLAIN:
            return self._format_plain()
        elif format == ContextFormat.STRUCTURED:
            return self._format_structured()
        elif format == ContextFormat.XML:
            return self._format_xml()
        elif format == ContextFormat.MARKDOWN:
            return self._format_markdown()
        elif format == ContextFormat.JSON:
            return self._format_json()
        else:
            return self._format_plain()

    def _format_plain(self) -> str:
        """Format as plain text."""
        parts = []

        if self.additional_context:
            parts.append(self.additional_context)

        for item in self.items:
            parts.append(item.content)

        return "\n".join(parts)

    def _format_structured(self) -> str:
        """Format with structure and headers."""
        parts = ["=== Relevant Context ==="]

        if self.additional_context:
            parts.append(f"\n{self.additional_context}")

        # Group by type
        by_type: Dict[str, List[ContextItem]] = {}
        for item in self.items:
            by_type.setdefault(item.item_type, []).append(item)

        type_headers = {
            "entity": "Key People and Things",
            "fact": "Important Facts",
            "event": "Events",
            "summary": "Summary",
        }

        for item_type, items in by_type.items():
            header = type_headers.get(item_type, item_type.title())
            parts.append(f"\n--- {header} ---")

            for item in items:
                parts.append(f"• {item.content}")

        parts.append("\n=== End Context ===")
        return "\n".join(parts)

    def _format_xml(self) -> str:
        """Format with XML-style tags."""
        parts = ["<context>"]

        if self.additional_context:
            parts.append(f"  <additional>{self.additional_context}</additional>")

        # Group by type
        by_type: Dict[str, List[ContextItem]] = {}
        for item in self.items:
            by_type.setdefault(item.item_type, []).append(item)

        for item_type, items in by_type.items():
            parts.append(f"  <{item_type}s>")
            for item in items:
                escaped = item.content.replace("<", "&lt;").replace(">", "&gt;")
                parts.append(f"    <{item_type}>{escaped}</{item_type}>")
            parts.append(f"  </{item_type}s>")

        parts.append("</context>")
        return "\n".join(parts)

    def _format_markdown(self) -> str:
        """Format as Markdown."""
        parts = ["## Relevant Context\n"]

        if self.additional_context:
            parts.append(f"{self.additional_context}\n")

        # Group by type
        by_type: Dict[str, List[ContextItem]] = {}
        for item in self.items:
            by_type.setdefault(item.item_type, []).append(item)

        type_headers = {
            "entity": "### Key Entities",
            "fact": "### Facts",
            "event": "### Events",
            "summary": "### Summary",
        }

        for item_type, items in by_type.items():
            header = type_headers.get(item_type, f"### {item_type.title()}")
            parts.append(header)

            for item in items:
                parts.append(f"- {item.content}")

            parts.append("")

        return "\n".join(parts)

    def _format_json(self) -> str:
        """Format as JSON string."""
        import json

        data = {
            "context": {
                "additional": self.additional_context,
                "items": [
                    {
                        "type": item.item_type,
                        "content": item.content,
                        "importance": item.importance,
                    }
                    for item in self.items
                ],
            }
        }

        return json.dumps(data, indent=2)

    def get_items_by_type(self, item_type: str) -> List[ContextItem]:
        """Get items of a specific type."""
        return [i for i in self.items if i.item_type == item_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.utilization,
            "item_count": self.item_count,
            "query": self.query,
            "items": [
                {
                    "id": item.id,
                    "type": item.item_type,
                    "content": item.content,
                    "tokens": item.tokens,
                    "importance": item.importance,
                    "recency": item.recency,
                    "relevance": item.relevance,
                }
                for item in self.items
            ],
        }


class ContextPacker:
    """Pack context items into formatted output.

    Provides utilities for arranging and formatting context
    for optimal LLM consumption.

    Example:
        >>> packer = ContextPacker()
        >>> packed = PackedContext(items=items, total_tokens=1000, max_tokens=4000)
        >>> formatted = packer.format(packed, ContextFormat.STRUCTURED)
    """

    def __init__(
        self,
        default_format: ContextFormat = ContextFormat.STRUCTURED,
        max_item_length: int = 500,
    ) -> None:
        """Initialize packer.

        Args:
            default_format: Default output format.
            max_item_length: Maximum characters per item.
        """
        self.default_format = default_format
        self.max_item_length = max_item_length

    def format(
        self,
        packed: PackedContext,
        format: Optional[ContextFormat] = None,
    ) -> str:
        """Format packed context.

        Args:
            packed: Packed context to format.
            format: Optional format override.

        Returns:
            Formatted string.
        """
        fmt = format or self.default_format
        return packed.to_string(fmt)

    def truncate_items(
        self,
        items: List[ContextItem],
        max_length: Optional[int] = None,
    ) -> List[ContextItem]:
        """Truncate item content to max length.

        Args:
            items: Items to truncate.
            max_length: Maximum content length.

        Returns:
            Items with truncated content.
        """
        max_len = max_length or self.max_item_length

        truncated = []
        for item in items:
            if len(item.content) > max_len:
                new_content = item.content[:max_len - 3] + "..."
                new_item = ContextItem(
                    id=item.id,
                    content=new_content,
                    tokens=len(new_content.split()),
                    importance=item.importance,
                    recency=item.recency,
                    relevance=item.relevance,
                    item_type=item.item_type,
                    metadata=item.metadata,
                )
                truncated.append(new_item)
            else:
                truncated.append(item)

        return truncated

    def reorder_for_llm(
        self,
        items: List[ContextItem],
        strategy: str = "importance_first",
    ) -> List[ContextItem]:
        """Reorder items for optimal LLM processing.

        Args:
            items: Items to reorder.
            strategy: Ordering strategy.

        Returns:
            Reordered items.
        """
        if strategy == "importance_first":
            return sorted(items, key=lambda x: x.importance, reverse=True)

        elif strategy == "recency_first":
            return sorted(items, key=lambda x: x.recency, reverse=True)

        elif strategy == "type_grouped":
            # Group by type, then by importance within groups
            by_type: Dict[str, List[ContextItem]] = {}
            for item in items:
                by_type.setdefault(item.item_type, []).append(item)

            ordered = []
            type_order = ["summary", "entity", "fact", "event"]

            for item_type in type_order:
                type_items = by_type.get(item_type, [])
                type_items.sort(key=lambda x: x.importance, reverse=True)
                ordered.extend(type_items)

            # Add any remaining types
            for item_type, type_items in by_type.items():
                if item_type not in type_order:
                    type_items.sort(key=lambda x: x.importance, reverse=True)
                    ordered.extend(type_items)

            return ordered

        elif strategy == "interleaved":
            # Interleave high and low importance for attention balance
            sorted_items = sorted(items, key=lambda x: x.importance, reverse=True)
            high = sorted_items[:len(sorted_items) // 2]
            low = sorted_items[len(sorted_items) // 2:]

            interleaved = []
            for h, l in zip(high, reversed(low)):
                interleaved.extend([h, l])

            # Handle odd lengths
            if len(high) > len(low):
                interleaved.extend(high[len(low):])

            return interleaved

        else:
            return items

    def create_prompt_context(
        self,
        packed: PackedContext,
        system_prompt: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> str:
        """Create full prompt context with system prompt and query.

        Args:
            packed: Packed context.
            system_prompt: Optional system prompt.
            user_query: Optional user query.

        Returns:
            Full prompt string.
        """
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}")

        context_str = packed.to_string(self.default_format)
        parts.append(f"\n{context_str}")

        if user_query:
            parts.append(f"\nUser: {user_query}")

        return "\n".join(parts)
