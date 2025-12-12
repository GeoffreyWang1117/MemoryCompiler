"""FastAPI web service for MemoryCompiler."""

from memory_compiler.api.app import create_app, app
from memory_compiler.api.models import (
    CompressRequest,
    CompressResponse,
    MemoryCreateRequest,
    MemoryResponse,
    RetrievalRequest,
    RetrievalResponse,
)

__all__ = [
    "create_app",
    "app",
    "CompressRequest",
    "CompressResponse",
    "MemoryCreateRequest",
    "MemoryResponse",
    "RetrievalRequest",
    "RetrievalResponse",
]
