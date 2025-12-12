"""Streaming and real-time compression support.

Provides capabilities for processing dialogue in real-time,
with incremental updates to the Memory IR.
"""

from memory_compiler.streaming.processor import (
    StreamingProcessor,
    StreamConfig,
    ProcessorState,
)
from memory_compiler.streaming.buffer import (
    TurnBuffer,
    WindowedBuffer,
)
from memory_compiler.streaming.incremental import (
    IncrementalCompressor,
    CompressionEvent,
)

__all__ = [
    "StreamingProcessor",
    "StreamConfig",
    "ProcessorState",
    "TurnBuffer",
    "WindowedBuffer",
    "IncrementalCompressor",
    "CompressionEvent",
]
