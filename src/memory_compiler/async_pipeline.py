"""Async and streaming support for MemoryCompiler.

This module provides asynchronous versions of the main compression
pipeline, supporting streaming dialogues and concurrent processing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from loguru import logger

from memory_compiler.extraction.extractor import MemoryExtractor
from memory_compiler.generation.generator import ContextGenerator
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import PassManager
from memory_compiler.passes.dead_memory import DeadMemoryEliminationPass
from memory_compiler.passes.fact_folding import FactFoldingPass
from memory_compiler.passes.importance import ImportancePruningPass
from memory_compiler.passes.temporal import TemporalCompressionPass
from memory_compiler.pipeline import CompressionResult


@dataclass
class StreamingState:
    """State for streaming compression."""

    ir: MemoryIR
    turn_count: int = 0
    last_compressed: str = ""
    compression_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class AsyncMemoryCompiler:
    """Asynchronous memory compiler with streaming support.

    This class provides async methods for compressing dialogues,
    supporting streaming scenarios where turns arrive incrementally.

    Features:
    - Async compression methods
    - Streaming turn processing
    - Configurable compression triggers
    - Concurrent batch processing

    Example:
        >>> compiler = AsyncMemoryCompiler()
        >>> async for result in compiler.stream_compress(dialogue_stream):
        ...     print(result.text)
    """

    def __init__(
        self,
        token_budget: int = 2048,
        compression_interval: int = 5,
        use_mock_extraction: bool = True,
        **kwargs: Any,
    ) -> None:
        self.token_budget = token_budget
        self.compression_interval = compression_interval
        self.config = kwargs

        # Initialize components
        self.extractor = MemoryExtractor(
            use_mock=use_mock_extraction,
        )

        self.pass_manager = PassManager([
            DeadMemoryEliminationPass(),
            FactFoldingPass(use_embeddings=False),
            TemporalCompressionPass(),
            ImportancePruningPass(token_budget=token_budget),
        ])

        self.generator = ContextGenerator(
            output_format=kwargs.get("output_format", "structured"),
        )

    async def compress_async(
        self, dialogue: list[dict[str, str]]
    ) -> CompressionResult:
        """Asynchronously compress a dialogue.

        This method runs compression in a thread pool to avoid
        blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compress_sync, dialogue)

    def _compress_sync(self, dialogue: list[dict[str, str]]) -> CompressionResult:
        """Synchronous compression implementation."""
        # Extract
        ir = self.extractor.extract(dialogue)

        # Optimize
        for pass_ in self.pass_manager.passes:
            pass_.run(ir)

        # Generate
        text = self.generator.generate(ir)

        # Compute metrics
        original_text = "\n".join(f"{t['role']}: {t['content']}" for t in dialogue)
        original_tokens = len(original_text) // 4
        compressed_tokens = len(text) // 4

        return CompressionResult(
            text=text,
            ir=ir,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
        )

    async def stream_compress(
        self,
        turn_stream: AsyncIterator[dict[str, str]],
        yield_interval: int | None = None,
    ) -> AsyncIterator[CompressionResult]:
        """Stream compression as turns arrive.

        Args:
            turn_stream: Async iterator yielding dialogue turns.
            yield_interval: Yield compressed result every N turns.

        Yields:
            CompressionResult after each compression.
        """
        interval = yield_interval or self.compression_interval
        state = StreamingState(ir=MemoryIR())

        async for turn in turn_stream:
            # Add turn to state
            state.ir.add_dialogue_turn(
                state.ir.dialogue_turns.__class__(
                    index=state.turn_count,
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                )
            )
            state.turn_count += 1

            # Compress at intervals
            if state.turn_count % interval == 0:
                result = await self._compress_state(state)
                state.compression_count += 1
                yield result

        # Final compression
        if state.turn_count % interval != 0:
            result = await self._compress_state(state)
            yield result

    async def _compress_state(self, state: StreamingState) -> CompressionResult:
        """Compress current streaming state."""
        # Build dialogue from IR turns
        dialogue = [
            {"role": turn.role, "content": turn.content}
            for turn in state.ir.dialogue_turns
        ]

        return await self.compress_async(dialogue)

    async def compress_batch(
        self,
        dialogues: list[list[dict[str, str]]],
        max_concurrent: int = 5,
    ) -> list[CompressionResult]:
        """Compress multiple dialogues concurrently.

        Args:
            dialogues: List of dialogues to compress.
            max_concurrent: Maximum concurrent compressions.

        Returns:
            List of compression results.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def compress_with_limit(dialogue: list[dict[str, str]]) -> CompressionResult:
            async with semaphore:
                return await self.compress_async(dialogue)

        tasks = [compress_with_limit(d) for d in dialogues]
        return await asyncio.gather(*tasks)

    async def compress_incremental(
        self,
        state: StreamingState,
        new_turns: list[dict[str, str]],
    ) -> tuple[StreamingState, CompressionResult]:
        """Incrementally compress new turns.

        Args:
            state: Current streaming state.
            new_turns: New dialogue turns to add.

        Returns:
            Updated state and compression result.
        """
        # Add new turns
        for turn in new_turns:
            state.ir.add_dialogue_turn(
                state.ir.dialogue_turns.__class__(
                    index=state.turn_count,
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                )
            )
            state.turn_count += 1

        # Re-extract for new turns (incremental)
        state.ir = self.extractor.extract_incremental(state.ir, new_turns)

        # Compress
        result = await self._compress_state(state)
        state.compression_count += 1

        return state, result


class StreamingDialogueProcessor:
    """Processor for handling streaming dialogue with compression.

    This class manages the state of an ongoing dialogue and provides
    compressed context on demand.

    Example:
        >>> processor = StreamingDialogueProcessor()
        >>> await processor.add_turn({"role": "user", "content": "Hello"})
        >>> await processor.add_turn({"role": "assistant", "content": "Hi!"})
        >>> context = await processor.get_compressed_context()
    """

    def __init__(
        self,
        token_budget: int = 2048,
        auto_compress_threshold: int = 10,
        **kwargs: Any,
    ) -> None:
        self.compiler = AsyncMemoryCompiler(
            token_budget=token_budget,
            **kwargs,
        )

        self.auto_compress_threshold = auto_compress_threshold
        self._state = StreamingState(ir=MemoryIR())
        self._last_result: CompressionResult | None = None
        self._dirty = False

    async def add_turn(self, turn: dict[str, str]) -> None:
        """Add a dialogue turn."""
        from memory_compiler.ir.memory_ir import DialogueTurn

        self._state.ir.add_dialogue_turn(
            DialogueTurn(
                index=self._state.turn_count,
                role=turn.get("role", "user"),
                content=turn.get("content", ""),
            )
        )
        self._state.turn_count += 1
        self._dirty = True

        # Auto-compress if threshold reached
        if self._state.turn_count % self.auto_compress_threshold == 0:
            await self._compress()

    async def add_turns(self, turns: list[dict[str, str]]) -> None:
        """Add multiple dialogue turns."""
        for turn in turns:
            await self.add_turn(turn)

    async def get_compressed_context(self, force_refresh: bool = False) -> str:
        """Get the current compressed context.

        Args:
            force_refresh: Force re-compression even if not dirty.

        Returns:
            Compressed context string.
        """
        if force_refresh or self._dirty or self._last_result is None:
            await self._compress()

        return self._last_result.text if self._last_result else ""

    async def get_result(self, force_refresh: bool = False) -> CompressionResult | None:
        """Get the full compression result."""
        if force_refresh or self._dirty or self._last_result is None:
            await self._compress()

        return self._last_result

    async def _compress(self) -> None:
        """Run compression on current state."""
        dialogue = [
            {"role": turn.role, "content": turn.content}
            for turn in self._state.ir.dialogue_turns
        ]

        if dialogue:
            self._last_result = await self.compiler.compress_async(dialogue)
            self._dirty = False

    @property
    def turn_count(self) -> int:
        """Number of turns in the dialogue."""
        return self._state.turn_count

    @property
    def compression_count(self) -> int:
        """Number of times compression has been performed."""
        return self._state.compression_count

    def reset(self) -> None:
        """Reset the processor state."""
        self._state = StreamingState(ir=MemoryIR())
        self._last_result = None
        self._dirty = False


async def demo_streaming():
    """Demo of streaming compression."""
    processor = StreamingDialogueProcessor(
        token_budget=500,
        auto_compress_threshold=3,
        use_mock_extraction=True,
    )

    # Simulate streaming dialogue
    turns = [
        {"role": "user", "content": "Hi, I'm Alice from TechCorp."},
        {"role": "assistant", "content": "Hello Alice! How can I help?"},
        {"role": "user", "content": "I need help with our ML pipeline."},
        {"role": "assistant", "content": "I'd be happy to help. What's the issue?"},
        {"role": "user", "content": "Training is too slow."},
    ]

    for turn in turns:
        await processor.add_turn(turn)
        print(f"Added turn {processor.turn_count}")

        if processor.turn_count % 3 == 0:
            context = await processor.get_compressed_context()
            print(f"Compressed context:\n{context}\n")

    # Final context
    final_context = await processor.get_compressed_context(force_refresh=True)
    print(f"Final compressed context:\n{final_context}")


if __name__ == "__main__":
    asyncio.run(demo_streaming())
