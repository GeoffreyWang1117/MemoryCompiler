"""Buffering strategies for streaming dialogue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Generic, Iterator, List, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class BufferItem(Generic[T]):
    """An item in the buffer with metadata.

    Attributes:
        data: The buffered data.
        timestamp: When item was added.
        priority: Optional priority for ordering.
        metadata: Additional metadata.
    """

    data: T
    timestamp: float = 0.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TurnBuffer:
    """Fixed-size buffer for dialogue turns.

    Maintains a fixed-size window of recent turns,
    automatically evicting oldest when full.

    Example:
        >>> buffer = TurnBuffer(max_size=10)
        >>> buffer.add("Hello!")
        >>> buffer.add("Hi there!")
        >>> recent = buffer.get_recent(5)
    """

    def __init__(self, max_size: int = 20) -> None:
        """Initialize buffer.

        Args:
            max_size: Maximum number of items.
        """
        self.max_size = max_size
        self._buffer: Deque[Any] = deque(maxlen=max_size)
        self._total_added = 0

    def add(self, item: Any) -> Optional[Any]:
        """Add an item to the buffer.

        Args:
            item: Item to add.

        Returns:
            Evicted item if buffer was full, None otherwise.
        """
        evicted = None

        if len(self._buffer) >= self.max_size:
            evicted = self._buffer[0]

        self._buffer.append(item)
        self._total_added += 1

        return evicted

    def get_recent(self, n: int) -> List[Any]:
        """Get n most recent items.

        Args:
            n: Number of items to retrieve.

        Returns:
            List of recent items (oldest to newest).
        """
        return list(self._buffer)[-n:]

    def get_all(self) -> List[Any]:
        """Get all items in buffer."""
        return list(self._buffer)

    def clear(self) -> List[Any]:
        """Clear buffer and return contents.

        Returns:
            All items that were in the buffer.
        """
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def peek(self) -> Optional[Any]:
        """Peek at most recent item without removing."""
        if self._buffer:
            return self._buffer[-1]
        return None

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._buffer)

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.max_size

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    @property
    def total_added(self) -> int:
        """Get total number of items ever added."""
        return self._total_added


class WindowedBuffer:
    """Sliding window buffer with token-based sizing.

    Maintains a window of items based on token budget rather
    than item count, useful for LLM context management.

    Example:
        >>> buffer = WindowedBuffer(max_tokens=2000)
        >>> buffer.add("Short message", tokens=10)
        >>> buffer.add("Longer message here", tokens=25)
        >>> context = buffer.get_context()
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        overlap_tokens: int = 200,
    ) -> None:
        """Initialize windowed buffer.

        Args:
            max_tokens: Maximum tokens in window.
            overlap_tokens: Overlap when sliding window.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self._items: List[BufferItem] = []
        self._current_tokens = 0

    def add(
        self,
        content: str,
        tokens: Optional[int] = None,
        timestamp: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[BufferItem]:
        """Add content to buffer.

        Args:
            content: Text content to add.
            tokens: Token count (estimated if not provided).
            timestamp: When item was added.
            metadata: Additional metadata.

        Returns:
            List of evicted items.
        """
        import time

        if tokens is None:
            tokens = len(content.split()) * 1.3

        item = BufferItem(
            data=content,
            timestamp=timestamp or time.time(),
            metadata=metadata or {"tokens": tokens},
        )
        item.metadata["tokens"] = tokens

        evicted = []

        # Check if we need to evict
        while self._current_tokens + tokens > self.max_tokens and self._items:
            old_item = self._items.pop(0)
            old_tokens = old_item.metadata.get("tokens", 0)
            self._current_tokens -= old_tokens
            evicted.append(old_item)

        self._items.append(item)
        self._current_tokens += tokens

        return evicted

    def get_context(self) -> str:
        """Get all content as context string.

        Returns:
            Concatenated context.
        """
        return "\n".join(item.data for item in self._items)

    def get_recent_context(self, max_tokens: int) -> str:
        """Get recent context up to token limit.

        Args:
            max_tokens: Maximum tokens to return.

        Returns:
            Recent context string.
        """
        result = []
        total = 0

        for item in reversed(self._items):
            item_tokens = item.metadata.get("tokens", 0)

            if total + item_tokens > max_tokens:
                break

            result.append(item.data)
            total += item_tokens

        return "\n".join(reversed(result))

    def slide_window(self) -> List[BufferItem]:
        """Slide window to evict old items.

        Evicts items to create room for overlap_tokens.

        Returns:
            Evicted items.
        """
        target = self.max_tokens - self.overlap_tokens
        evicted = []

        while self._current_tokens > target and self._items:
            item = self._items.pop(0)
            item_tokens = item.metadata.get("tokens", 0)
            self._current_tokens -= item_tokens
            evicted.append(item)

        return evicted

    def clear(self) -> List[BufferItem]:
        """Clear buffer.

        Returns:
            All evicted items.
        """
        items = self._items
        self._items = []
        self._current_tokens = 0
        return items

    @property
    def current_tokens(self) -> int:
        """Get current token count."""
        return self._current_tokens

    @property
    def remaining_tokens(self) -> int:
        """Get remaining token capacity."""
        return max(0, self.max_tokens - self._current_tokens)

    @property
    def utilization(self) -> float:
        """Get buffer utilization (0-1)."""
        return self._current_tokens / self.max_tokens if self.max_tokens > 0 else 0

    def __len__(self) -> int:
        return len(self._items)


class PriorityBuffer:
    """Priority-based buffer that keeps most important items.

    Items with higher priority are kept when buffer is full.
    Useful for importance-based memory retention.

    Example:
        >>> buffer = PriorityBuffer(max_size=100)
        >>> buffer.add("Important fact", priority=10)
        >>> buffer.add("Less important", priority=1)
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize priority buffer.

        Args:
            max_size: Maximum number of items.
        """
        self.max_size = max_size
        self._items: List[BufferItem] = []

    def add(
        self,
        content: str,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[BufferItem]:
        """Add item with priority.

        Args:
            content: Content to add.
            priority: Priority score (higher = more important).
            metadata: Additional metadata.

        Returns:
            Evicted item if any.
        """
        import time

        item = BufferItem(
            data=content,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {},
        )

        evicted = None

        if len(self._items) >= self.max_size:
            # Find lowest priority item
            min_idx = min(range(len(self._items)), key=lambda i: self._items[i].priority)

            if self._items[min_idx].priority < priority:
                evicted = self._items.pop(min_idx)
            else:
                # New item has lower priority, don't add
                return item

        self._items.append(item)
        return evicted

    def get_by_priority(self, top_k: Optional[int] = None) -> List[BufferItem]:
        """Get items sorted by priority.

        Args:
            top_k: Optional limit.

        Returns:
            Items sorted by priority (highest first).
        """
        sorted_items = sorted(self._items, key=lambda x: x.priority, reverse=True)

        if top_k:
            return sorted_items[:top_k]
        return sorted_items

    def update_priority(self, content: str, new_priority: int) -> bool:
        """Update priority of an existing item.

        Args:
            content: Content to find.
            new_priority: New priority value.

        Returns:
            True if item was found and updated.
        """
        for item in self._items:
            if item.data == content:
                item.priority = new_priority
                return True
        return False

    def clear(self) -> List[BufferItem]:
        """Clear buffer."""
        items = self._items
        self._items = []
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[BufferItem]:
        return iter(self._items)
