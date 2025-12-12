"""Pydantic models for API request/response."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a dialogue message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DialogueMessage(BaseModel):
    """A single dialogue message.

    Attributes:
        role: Speaker role.
        content: Message content.
        timestamp: Optional message timestamp.
    """

    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None


class CompressRequest(BaseModel):
    """Request to compress a dialogue.

    Attributes:
        messages: List of dialogue messages.
        token_budget: Maximum tokens for compressed output.
        passes: Optimization passes to apply.
        output_format: Format of compressed output.
        session_id: Optional session identifier.
    """

    messages: List[DialogueMessage]
    token_budget: int = Field(default=2048, ge=100, le=16384)
    passes: Optional[List[str]] = None
    output_format: str = Field(default="narrative")
    session_id: Optional[str] = None


class EntityInfo(BaseModel):
    """Information about an extracted entity."""

    id: str
    name: str
    type: str
    importance: float
    aliases: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class FactInfo(BaseModel):
    """Information about an extracted fact."""

    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    importance: float


class CompressResponse(BaseModel):
    """Response from compression request.

    Attributes:
        compressed_text: Compressed output text.
        original_tokens: Estimated original token count.
        compressed_tokens: Estimated compressed token count.
        compression_ratio: Compression ratio achieved.
        entities: Extracted entities.
        facts: Extracted facts.
        ir_json: Optional full IR in JSON format.
    """

    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    entities: List[EntityInfo]
    facts: List[FactInfo]
    ir_json: Optional[Dict[str, Any]] = None


class MemoryCreateRequest(BaseModel):
    """Request to create/update a memory.

    Attributes:
        key: Unique memory key.
        messages: Dialogue messages.
        metadata: Optional metadata.
        compress: Whether to compress before storing.
        token_budget: Token budget if compressing.
    """

    key: str = Field(..., min_length=1, max_length=256)
    messages: List[DialogueMessage]
    metadata: Optional[Dict[str, Any]] = None
    compress: bool = True
    token_budget: int = Field(default=2048, ge=100, le=16384)


class MemoryResponse(BaseModel):
    """Response for memory operations.

    Attributes:
        key: Memory key.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        entity_count: Number of entities.
        fact_count: Number of facts.
        metadata: Associated metadata.
    """

    key: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    entity_count: int
    fact_count: int
    metadata: Optional[Dict[str, Any]] = None


class MemoryLoadResponse(BaseModel):
    """Response when loading a memory.

    Attributes:
        key: Memory key.
        ir: Full IR data.
        metadata: Associated metadata.
        compressed_text: Generated compressed text.
    """

    key: str
    ir: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    compressed_text: str


class RetrievalRequest(BaseModel):
    """Request for memory retrieval.

    Attributes:
        query: Search query.
        top_k: Number of results.
        filter_keys: Optional memory keys to search within.
        include_context: Whether to include formatted context.
        max_context_tokens: Token budget for context.
    """

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filter_keys: Optional[List[str]] = None
    include_context: bool = True
    max_context_tokens: int = Field(default=2048, ge=100, le=8192)


class RetrievalResult(BaseModel):
    """A single retrieval result."""

    text: str
    score: float
    source_key: str
    chunk_id: str
    chunk_type: Optional[str] = None


class RetrievalResponse(BaseModel):
    """Response from retrieval request.

    Attributes:
        results: List of retrieval results.
        context: Formatted context string.
        query: Original query.
    """

    results: List[RetrievalResult]
    context: Optional[str] = None
    query: str


class StoreStatsResponse(BaseModel):
    """Storage statistics response."""

    num_memories: int
    total_entities: int
    total_facts: int
    total_chunks: int
    storage_type: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    storage_available: bool
    retriever_ready: bool


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class BatchCompressRequest(BaseModel):
    """Request for batch compression.

    Attributes:
        sessions: List of sessions to compress.
        token_budget: Token budget per session.
        parallel: Whether to process in parallel.
    """

    sessions: List[CompressRequest]
    token_budget: int = Field(default=2048, ge=100, le=16384)
    parallel: bool = True


class BatchCompressResponse(BaseModel):
    """Response from batch compression."""

    results: List[CompressResponse]
    total_processed: int
    failed: int
