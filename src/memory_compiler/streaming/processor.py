"""Real-time streaming processor for dialogue compression."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import MemoryCompiler


class ProcessorState(Enum):
    """State of the streaming processor."""

    IDLE = "idle"
    PROCESSING = "processing"
    COMPRESSING = "compressing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for streaming processor.

    Attributes:
        buffer_size: Number of turns to buffer before processing.
        compression_threshold: Token count threshold to trigger compression.
        auto_compress: Whether to automatically compress when threshold hit.
        update_interval: Interval for IR updates in seconds.
        max_context_tokens: Maximum context size before forced compression.
    """

    buffer_size: int = 5
    compression_threshold: int = 4000
    auto_compress: bool = True
    update_interval: float = 1.0
    max_context_tokens: int = 8000


@dataclass
class DialogueTurn:
    """A single dialogue turn.

    Attributes:
        role: Speaker role (user/assistant).
        content: Turn content.
        timestamp: When the turn occurred.
        metadata: Additional turn metadata.
    """

    role: str
    content: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingProcessor:
    """Process dialogue in real-time with incremental compression.

    Buffers incoming dialogue turns and incrementally updates
    the Memory IR, triggering compression as needed.

    Example:
        >>> processor = StreamingProcessor()
        >>> await processor.start()
        >>>
        >>> await processor.add_turn(DialogueTurn(role="user", content="Hello"))
        >>> await processor.add_turn(DialogueTurn(role="assistant", content="Hi!"))
        >>>
        >>> ir = await processor.get_current_ir()
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        compiler: Optional[MemoryCompiler] = None,
    ) -> None:
        """Initialize processor.

        Args:
            config: Stream configuration.
            compiler: MemoryCompiler instance.
        """
        self.config = config or StreamConfig()
        self.compiler = compiler or MemoryCompiler(use_mock=True)

        self._state = ProcessorState.IDLE
        self._buffer: List[DialogueTurn] = []
        self._current_ir: Optional[MemoryIR] = None
        self._full_history: List[DialogueTurn] = []
        self._token_count = 0

        self._callbacks: Dict[str, List[Callable]] = {
            "on_turn": [],
            "on_compress": [],
            "on_update": [],
            "on_error": [],
        }

        self._processing_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> ProcessorState:
        """Get current processor state."""
        return self._state

    @property
    def current_ir(self) -> Optional[MemoryIR]:
        """Get current Memory IR."""
        return self._current_ir

    @property
    def token_count(self) -> int:
        """Get estimated token count."""
        return self._token_count

    async def start(self) -> None:
        """Start the streaming processor."""
        if self._state != ProcessorState.IDLE:
            return

        self._state = ProcessorState.PROCESSING
        self._current_ir = MemoryIR(session_id="streaming")

        logger.info("Streaming processor started")

    async def stop(self) -> None:
        """Stop the streaming processor."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Process remaining buffer
        if self._buffer:
            await self._process_buffer()

        self._state = ProcessorState.IDLE
        logger.info("Streaming processor stopped")

    async def add_turn(self, turn: DialogueTurn) -> None:
        """Add a dialogue turn to the stream.

        Args:
            turn: Dialogue turn to add.
        """
        if self._state not in (ProcessorState.PROCESSING, ProcessorState.COMPRESSING):
            raise RuntimeError(f"Cannot add turn in state: {self._state}")

        async with self._lock:
            self._buffer.append(turn)
            self._full_history.append(turn)

            # Estimate tokens
            self._token_count += len(turn.content.split()) * 1.3

            # Notify callbacks
            await self._notify("on_turn", turn)

            # Check if we should process buffer
            if len(self._buffer) >= self.config.buffer_size:
                await self._process_buffer()

            # Check if we should compress
            if (
                self.config.auto_compress
                and self._token_count >= self.config.compression_threshold
            ):
                await self._trigger_compression()

    async def _process_buffer(self) -> None:
        """Process buffered turns and update IR."""
        if not self._buffer:
            return

        # Convert buffer to dialogue text
        dialogue = "\n".join(
            f"{t.role}: {t.content}" for t in self._buffer
        )

        # Extract from buffer
        try:
            new_ir = await asyncio.to_thread(
                self.compiler.extract, dialogue
            )

            # Merge with current IR
            if self._current_ir:
                from memory_compiler.merging import MemoryMerger
                merger = MemoryMerger()
                result = merger.merge_into_existing(self._current_ir, new_ir)
                self._current_ir = result.merged_ir
            else:
                self._current_ir = new_ir

            await self._notify("on_update", self._current_ir)

        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            await self._notify("on_error", e)

        finally:
            self._buffer.clear()

    async def _trigger_compression(self) -> None:
        """Trigger compression of accumulated memory."""
        if self._state == ProcessorState.COMPRESSING:
            return

        prev_state = self._state
        self._state = ProcessorState.COMPRESSING

        try:
            # Process any remaining buffer first
            await self._process_buffer()

            # Compress the full history
            full_dialogue = "\n".join(
                f"{t.role}: {t.content}" for t in self._full_history
            )

            compressed_ir = await asyncio.to_thread(
                self.compiler.compile,
                full_dialogue,
                token_budget=self.config.compression_threshold // 2,
            )

            self._current_ir = compressed_ir

            # Update token estimate
            compressed_text = await asyncio.to_thread(
                self.compiler.generate, compressed_ir
            )
            self._token_count = len(compressed_text.split()) * 1.3

            await self._notify("on_compress", compressed_ir)

            logger.debug(f"Compressed memory, new token count: {self._token_count}")

        except Exception as e:
            logger.error(f"Compression error: {e}")
            await self._notify("on_error", e)

        finally:
            self._state = prev_state

    async def force_compress(self, token_budget: Optional[int] = None) -> MemoryIR:
        """Force compression regardless of threshold.

        Args:
            token_budget: Optional token budget for compression.

        Returns:
            Compressed Memory IR.
        """
        budget = token_budget or self.config.compression_threshold // 2
        await self._trigger_compression()
        return self._current_ir

    async def get_context(
        self,
        max_tokens: Optional[int] = None,
        style: str = "narrative",
    ) -> str:
        """Get current context for prompt injection.

        Args:
            max_tokens: Maximum tokens for context.
            style: Output style.

        Returns:
            Context string.
        """
        if not self._current_ir:
            return ""

        # Process buffer first
        if self._buffer:
            await self._process_buffer()

        return await asyncio.to_thread(
            self.compiler.generate,
            self._current_ir,
            style=style,
            max_tokens=max_tokens,
        )

    def on(self, event: str, callback: Callable) -> None:
        """Register an event callback.

        Args:
            event: Event name (on_turn, on_compress, on_update, on_error).
            callback: Callback function.
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _notify(self, event: str, data: Any) -> None:
        """Notify registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    async def stream_turns(
        self,
        turns: AsyncIterator[DialogueTurn],
    ) -> AsyncIterator[MemoryIR]:
        """Process a stream of turns and yield IR updates.

        Args:
            turns: Async iterator of dialogue turns.

        Yields:
            Updated Memory IR after processing.
        """
        await self.start()

        try:
            async for turn in turns:
                await self.add_turn(turn)

                # Yield current IR
                if self._current_ir:
                    yield self._current_ir

        finally:
            await self.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "state": self._state.value,
            "buffer_size": len(self._buffer),
            "total_turns": len(self._full_history),
            "token_count": self._token_count,
            "has_ir": self._current_ir is not None,
            "entities": len(list(self._current_ir.iter_entities())) if self._current_ir else 0,
            "facts": len(list(self._current_ir.iter_facts())) if self._current_ir else 0,
        }

    async def reset(self) -> None:
        """Reset the processor state."""
        async with self._lock:
            self._buffer.clear()
            self._full_history.clear()
            self._current_ir = MemoryIR(session_id="streaming")
            self._token_count = 0
            self._state = ProcessorState.PROCESSING
