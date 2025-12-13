"""Tests for streaming compression module."""

import pytest
from memory_compiler.streaming import (
    StreamingProcessor,
    TurnBuffer,
    WindowedBuffer,
    PriorityBuffer,
    IncrementalCompressor,
    AdaptiveCompressor,
)


class TestTurnBuffer:
    """Tests for TurnBuffer."""

    def test_add_turn(self):
        """Test adding turns to buffer."""
        buffer = TurnBuffer(max_turns=5)
        buffer.add("Hello")
        buffer.add("How are you?")

        assert buffer.size() == 2
        assert buffer.get_all() == ["Hello", "How are you?"]

    def test_max_turns_limit(self):
        """Test buffer respects max turns limit."""
        buffer = TurnBuffer(max_turns=3)
        for i in range(5):
            buffer.add(f"Turn {i}")

        assert buffer.size() == 3
        # Should keep most recent turns
        turns = buffer.get_all()
        assert len(turns) == 3

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = TurnBuffer(max_turns=10)
        buffer.add("Test")
        buffer.clear()

        assert buffer.size() == 0
        assert buffer.is_empty()

    def test_get_recent(self):
        """Test getting recent turns."""
        buffer = TurnBuffer(max_turns=10)
        for i in range(5):
            buffer.add(f"Turn {i}")

        recent = buffer.get_recent(3)
        assert len(recent) == 3


class TestWindowedBuffer:
    """Tests for WindowedBuffer."""

    def test_token_limit(self):
        """Test buffer respects token limit."""
        buffer = WindowedBuffer(max_tokens=50)
        buffer.add("This is a test message")
        buffer.add("Another test message here")

        # Should maintain token limit
        assert buffer.get_token_count() <= 50

    def test_flush(self):
        """Test flushing buffer."""
        buffer = WindowedBuffer(max_tokens=100)
        buffer.add("Test content")

        content = buffer.flush()
        assert len(content) > 0
        assert buffer.is_empty()


class TestPriorityBuffer:
    """Tests for PriorityBuffer."""

    def test_priority_ordering(self):
        """Test items ordered by priority."""
        buffer = PriorityBuffer(max_items=10)
        buffer.add("Low priority", priority=0.3)
        buffer.add("High priority", priority=0.9)
        buffer.add("Medium priority", priority=0.5)

        items = buffer.get_top(2)
        assert len(items) == 2
        # Higher priority should come first
        assert items[0][0] == "High priority"

    def test_max_items_limit(self):
        """Test buffer respects max items."""
        buffer = PriorityBuffer(max_items=3)
        for i in range(5):
            buffer.add(f"Item {i}", priority=i * 0.2)

        assert buffer.size() == 3


class TestStreamingProcessor:
    """Tests for StreamingProcessor."""

    def test_process_turn(self):
        """Test processing a single turn."""
        processor = StreamingProcessor(
            compression_threshold=5,
            auto_compress=False,
        )

        processor.add_turn("User: Hello!")
        processor.add_turn("Assistant: Hi there!")

        assert processor.turn_count == 2

    def test_auto_compression(self):
        """Test automatic compression triggers."""
        processor = StreamingProcessor(
            compression_threshold=3,
            auto_compress=True,
        )

        for i in range(5):
            processor.add_turn(f"Turn {i}: Some dialogue content here.")

        # Should have triggered compression
        ir = processor.get_current_ir()
        assert ir is not None

    def test_get_context(self):
        """Test getting current context."""
        processor = StreamingProcessor()
        processor.add_turn("User: What's your name?")
        processor.add_turn("Assistant: I'm an AI assistant.")

        context = processor.get_context(max_tokens=100)
        assert "name" in context.lower() or "assistant" in context.lower()


class TestIncrementalCompressor:
    """Tests for IncrementalCompressor."""

    def test_incremental_update(self):
        """Test incremental updates."""
        compressor = IncrementalCompressor()

        compressor.update("User mentioned they like Python programming.")
        compressor.update("User said they work at a tech company.")

        ir = compressor.get_ir()
        assert ir is not None

    def test_entity_tracking(self):
        """Test entity tracking across updates."""
        compressor = IncrementalCompressor()

        compressor.update("John is a software engineer.")
        compressor.update("John works at Google.")

        ir = compressor.get_ir()
        # Should have John as an entity
        entities = list(ir.iter_entities())
        entity_names = [e.name.lower() for e in entities]
        assert any("john" in name for name in entity_names)


class TestAdaptiveCompressor:
    """Tests for AdaptiveCompressor."""

    def test_adaptive_threshold(self):
        """Test adaptive threshold adjustment."""
        compressor = AdaptiveCompressor(
            initial_threshold=5,
            min_threshold=2,
            max_threshold=20,
        )

        # Simulate varying information density
        for i in range(10):
            compressor.update(f"Information dense content {i} with many facts.")

        # Threshold should have adapted
        assert compressor.current_threshold >= 2
        assert compressor.current_threshold <= 20

    def test_compression_quality_feedback(self):
        """Test quality feedback affects compression."""
        compressor = AdaptiveCompressor()

        compressor.update("Test content")
        compressor.provide_feedback(quality_score=0.8)

        # Should accept feedback without error
        assert compressor.get_stats()["feedback_count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
