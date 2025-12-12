"""Incremental compression for streaming dialogue."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.entities import Entity
from memory_compiler.ir.facts import Fact
from memory_compiler.pipeline import MemoryCompiler
from memory_compiler.streaming.buffer import WindowedBuffer


class CompressionEventType(Enum):
    """Types of compression events."""

    ENTITY_ADDED = "entity_added"
    ENTITY_UPDATED = "entity_updated"
    FACT_ADDED = "fact_added"
    FACT_REMOVED = "fact_removed"
    COMPRESSION_TRIGGERED = "compression_triggered"
    CONTEXT_UPDATED = "context_updated"


@dataclass
class CompressionEvent:
    """Event generated during incremental compression.

    Attributes:
        event_type: Type of event.
        data: Event data.
        timestamp: When event occurred.
        metadata: Additional metadata.
    """

    event_type: CompressionEventType
    data: Any
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncrementalCompressor:
    """Incrementally compress dialogue as it arrives.

    Maintains a running Memory IR and updates it incrementally
    with each new dialogue turn, only doing full compression
    when necessary.

    Example:
        >>> compressor = IncrementalCompressor(token_budget=2000)
        >>>
        >>> async for event in compressor.process_turn("User: Hello!"):
        ...     print(f"Event: {event.event_type}")
        >>>
        >>> context = compressor.get_context()
    """

    def __init__(
        self,
        token_budget: int = 2000,
        compression_ratio_target: float = 0.5,
        incremental_extraction: bool = True,
        compiler: Optional[MemoryCompiler] = None,
    ) -> None:
        """Initialize compressor.

        Args:
            token_budget: Target token budget.
            compression_ratio_target: Target compression ratio.
            incremental_extraction: Whether to extract incrementally.
            compiler: MemoryCompiler instance.
        """
        self.token_budget = token_budget
        self.compression_ratio_target = compression_ratio_target
        self.incremental_extraction = incremental_extraction
        self.compiler = compiler or MemoryCompiler(use_mock=True)

        self._ir = MemoryIR(session_id="incremental")
        self._buffer = WindowedBuffer(max_tokens=token_budget * 2)
        self._raw_history: List[str] = []
        self._event_listeners: List[Callable[[CompressionEvent], None]] = []
        self._compression_count = 0

    @property
    def current_ir(self) -> MemoryIR:
        """Get current Memory IR."""
        return self._ir

    def add_event_listener(self, listener: Callable[[CompressionEvent], None]) -> None:
        """Add an event listener.

        Args:
            listener: Callback function for events.
        """
        self._event_listeners.append(listener)

    def _emit_event(
        self,
        event_type: CompressionEventType,
        data: Any,
        **metadata,
    ) -> CompressionEvent:
        """Emit a compression event."""
        import time

        event = CompressionEvent(
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            metadata=metadata,
        )

        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")

        return event

    async def process_turn(self, turn: str) -> List[CompressionEvent]:
        """Process a single dialogue turn.

        Args:
            turn: Dialogue turn text (e.g., "User: Hello!").

        Returns:
            List of events generated.
        """
        events = []
        self._raw_history.append(turn)

        # Estimate tokens
        turn_tokens = len(turn.split()) * 1.3

        # Add to buffer
        evicted = self._buffer.add(turn, tokens=turn_tokens)

        # If incremental extraction is enabled, extract from turn
        if self.incremental_extraction:
            turn_events = await self._extract_from_turn(turn)
            events.extend(turn_events)

        # Check if compression is needed
        if self._should_compress():
            compress_events = await self._do_compression()
            events.extend(compress_events)

        return events

    async def _extract_from_turn(self, turn: str) -> List[CompressionEvent]:
        """Extract entities and facts from a single turn."""
        events = []

        try:
            # Quick extraction from single turn
            turn_ir = await asyncio.to_thread(self.compiler.extract, turn)

            # Merge new entities
            for entity in turn_ir.iter_entities():
                existing = self._find_entity(entity.name)

                if existing:
                    # Update existing entity
                    self._merge_entity(existing, entity)
                    events.append(
                        self._emit_event(
                            CompressionEventType.ENTITY_UPDATED,
                            existing,
                            new_data=entity.to_dict(),
                        )
                    )
                else:
                    # Add new entity
                    self._ir.add_entity(entity)
                    events.append(
                        self._emit_event(
                            CompressionEventType.ENTITY_ADDED,
                            entity,
                        )
                    )

            # Merge new facts
            for fact in turn_ir.iter_facts():
                if not self._has_similar_fact(fact):
                    self._ir.add_fact(fact)
                    events.append(
                        self._emit_event(
                            CompressionEventType.FACT_ADDED,
                            fact,
                        )
                    )

        except Exception as e:
            logger.error(f"Extraction error: {e}")

        return events

    def _find_entity(self, name: str) -> Optional[Entity]:
        """Find entity by name or alias."""
        name_lower = name.lower()

        for entity in self._ir.iter_entities():
            if entity.name.lower() == name_lower:
                return entity
            if any(a.lower() == name_lower for a in entity.aliases):
                return entity

        return None

    def _merge_entity(self, existing: Entity, new: Entity) -> None:
        """Merge new entity data into existing."""
        existing.aliases.update(new.aliases)
        existing.attributes.update(new.attributes)
        existing.mentions.extend(new.mentions)
        existing.importance_score = max(existing.importance_score, new.importance_score)

    def _has_similar_fact(self, fact: Fact) -> bool:
        """Check if a similar fact already exists."""
        fact_key = f"{fact.subject}|{fact.predicate}|{fact.object}".lower()

        for existing in self._ir.iter_facts():
            existing_key = f"{existing.subject}|{existing.predicate}|{existing.object}".lower()
            if fact_key == existing_key:
                return True

        return False

    def _should_compress(self) -> bool:
        """Check if full compression should be triggered."""
        return self._buffer.current_tokens > self.token_budget

    async def _do_compression(self) -> List[CompressionEvent]:
        """Perform full compression."""
        events = []
        self._compression_count += 1

        events.append(
            self._emit_event(
                CompressionEventType.COMPRESSION_TRIGGERED,
                {"compression_number": self._compression_count},
            )
        )

        try:
            # Get full history
            full_dialogue = "\n".join(self._raw_history)

            # Compress
            target_tokens = int(self.token_budget * self.compression_ratio_target)

            compressed_ir = await asyncio.to_thread(
                self.compiler.compile,
                full_dialogue,
                token_budget=target_tokens,
            )

            # Track removed facts
            old_facts = set(f.id for f in self._ir.iter_facts())
            new_facts = set(f.id for f in compressed_ir.iter_facts())

            for removed_id in old_facts - new_facts:
                if removed_id in self._ir.facts:
                    events.append(
                        self._emit_event(
                            CompressionEventType.FACT_REMOVED,
                            self._ir.facts[removed_id],
                        )
                    )

            # Update IR
            self._ir = compressed_ir

            # Slide the buffer
            self._buffer.slide_window()

            events.append(
                self._emit_event(
                    CompressionEventType.CONTEXT_UPDATED,
                    self._ir,
                    tokens_before=self._buffer.current_tokens,
                    tokens_after=target_tokens,
                )
            )

        except Exception as e:
            logger.error(f"Compression error: {e}")

        return events

    def get_context(self, style: str = "narrative") -> str:
        """Get current compressed context.

        Args:
            style: Output style.

        Returns:
            Compressed context string.
        """
        return self.compiler.generate(self._ir, style=style)

    async def get_context_async(self, style: str = "narrative") -> str:
        """Get context asynchronously."""
        return await asyncio.to_thread(self.get_context, style)

    def get_stats(self) -> Dict[str, Any]:
        """Get compressor statistics."""
        return {
            "token_budget": self.token_budget,
            "buffer_tokens": self._buffer.current_tokens,
            "buffer_utilization": self._buffer.utilization,
            "history_turns": len(self._raw_history),
            "compression_count": self._compression_count,
            "entities": len(list(self._ir.iter_entities())),
            "facts": len(list(self._ir.iter_facts())),
        }

    def reset(self) -> None:
        """Reset compressor state."""
        self._ir = MemoryIR(session_id="incremental")
        self._buffer.clear()
        self._raw_history.clear()
        self._compression_count = 0


class AdaptiveCompressor(IncrementalCompressor):
    """Adaptive compressor that adjusts parameters based on dialogue.

    Automatically adjusts compression thresholds based on
    dialogue characteristics like information density.
    """

    def __init__(
        self,
        token_budget: int = 2000,
        min_compression_ratio: float = 0.3,
        max_compression_ratio: float = 0.7,
        **kwargs,
    ) -> None:
        """Initialize adaptive compressor.

        Args:
            token_budget: Target token budget.
            min_compression_ratio: Minimum compression ratio.
            max_compression_ratio: Maximum compression ratio.
            **kwargs: Additional arguments for parent.
        """
        super().__init__(token_budget=token_budget, **kwargs)

        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio

        self._info_density_history: List[float] = []

    async def process_turn(self, turn: str) -> List[CompressionEvent]:
        """Process turn with adaptive parameters."""
        # Estimate information density of turn
        density = self._estimate_info_density(turn)
        self._info_density_history.append(density)

        # Adapt compression ratio based on density
        self._adapt_compression_ratio()

        return await super().process_turn(turn)

    def _estimate_info_density(self, turn: str) -> float:
        """Estimate information density of a turn.

        Higher density = more named entities, numbers, etc.
        """
        import re

        words = turn.split()
        if not words:
            return 0.0

        # Count information markers
        markers = 0

        # Named entities (capitalized words)
        markers += len(re.findall(r"\b[A-Z][a-z]+\b", turn))

        # Numbers
        markers += len(re.findall(r"\b\d+\b", turn))

        # Dates/times
        markers += len(re.findall(r"\b\d{1,2}[/\-]\d{1,2}\b", turn))

        return markers / len(words)

    def _adapt_compression_ratio(self) -> None:
        """Adapt compression ratio based on recent density."""
        if len(self._info_density_history) < 5:
            return

        # Average recent density
        recent_density = sum(self._info_density_history[-5:]) / 5

        # Higher density -> lower compression ratio (keep more)
        # Lower density -> higher compression ratio (compress more)
        if recent_density > 0.15:
            # High info density - keep more
            self.compression_ratio_target = self.min_compression_ratio
        elif recent_density < 0.05:
            # Low info density - compress more
            self.compression_ratio_target = self.max_compression_ratio
        else:
            # Interpolate
            t = (recent_density - 0.05) / 0.10
            self.compression_ratio_target = (
                self.max_compression_ratio
                - t * (self.max_compression_ratio - self.min_compression_ratio)
            )
