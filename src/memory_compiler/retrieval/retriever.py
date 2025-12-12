"""Memory retriever for RAG-style context injection."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.retrieval.vector_store import (
    InMemoryVectorStore,
    VectorDocument,
    VectorStore,
)


class RetrievalStrategy(Enum):
    """Strategy for retrieving memory chunks."""

    SEMANTIC = "semantic"  # Pure embedding similarity
    RECENCY = "recency"  # Prioritize recent memories
    IMPORTANCE = "importance"  # Prioritize high-importance items
    HYBRID = "hybrid"  # Combine semantic + recency + importance


@dataclass
class RetrievalConfig:
    """Configuration for memory retrieval.

    Attributes:
        strategy: Retrieval strategy to use.
        top_k: Number of results to return.
        similarity_threshold: Minimum similarity score.
        recency_weight: Weight for recency in hybrid mode.
        importance_weight: Weight for importance in hybrid mode.
        semantic_weight: Weight for semantic similarity in hybrid mode.
        chunk_size: Size of text chunks for indexing.
        chunk_overlap: Overlap between chunks.
        rerank: Whether to rerank results.
    """

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    similarity_threshold: float = 0.5
    recency_weight: float = 0.2
    importance_weight: float = 0.3
    semantic_weight: float = 0.5
    chunk_size: int = 512
    chunk_overlap: int = 64
    rerank: bool = False


@dataclass
class RetrievalResult:
    """Result of a memory retrieval query.

    Attributes:
        text: Retrieved text content.
        score: Relevance score (0-1).
        source_key: Key of the source memory.
        chunk_id: ID of the specific chunk.
        metadata: Additional metadata.
    """

    text: str
    score: float
    source_key: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever:
    """RAG-style retriever for Memory IR content.

    Indexes Memory IR content into searchable chunks and provides
    semantic retrieval for context injection into prompts.

    Example:
        >>> retriever = MemoryRetriever(embed_fn=embed_function)
        >>> retriever.index_memory("session_1", memory_ir)
        >>> results = retriever.retrieve("What did the user say about Python?")
        >>> context = retriever.format_context(results)
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        """Initialize retriever.

        Args:
            embed_fn: Function to embed text into vectors.
            vector_store: Vector store backend.
            config: Retrieval configuration.
        """
        self.embed_fn = embed_fn or self._default_embed
        self.vector_store = vector_store or InMemoryVectorStore()
        self.config = config or RetrievalConfig()

        self._memory_timestamps: Dict[str, float] = {}
        self._chunk_importance: Dict[str, float] = {}

    def _default_embed(self, text: str) -> np.ndarray:
        """Default embedding using simple hash-based approach."""
        # Simple character-based embedding for testing
        # In production, use a real embedding model
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).astype(np.float32)

    def index_memory(
        self,
        key: str,
        ir: MemoryIR,
        timestamp: Optional[float] = None,
    ) -> int:
        """Index a Memory IR for retrieval.

        Args:
            key: Unique key for this memory.
            ir: Memory IR to index.
            timestamp: Optional timestamp for recency scoring.

        Returns:
            Number of chunks indexed.
        """
        import time

        self._memory_timestamps[key] = timestamp or time.time()

        chunks = self._chunk_ir(ir, key)
        documents = []

        for chunk in chunks:
            embedding = self.embed_fn(chunk["text"])

            doc = VectorDocument(
                id=chunk["id"],
                text=chunk["text"],
                embedding=embedding,
                metadata={
                    "source_key": key,
                    "chunk_type": chunk["type"],
                    "importance": chunk["importance"],
                },
            )

            self._chunk_importance[chunk["id"]] = chunk["importance"]
            documents.append(doc)

        self.vector_store.add(documents)
        logger.debug(f"Indexed {len(documents)} chunks from memory: {key}")

        return len(documents)

    def _chunk_ir(self, ir: MemoryIR, key: str) -> List[Dict[str, Any]]:
        """Convert Memory IR into searchable chunks."""
        chunks = []

        # Chunk entities
        for entity in ir.iter_entities():
            text = f"{entity.name}"
            if entity.aliases:
                text += f" (also known as: {', '.join(entity.aliases)})"
            if entity.attributes:
                attrs = ", ".join(f"{k}: {v}" for k, v in entity.attributes.items())
                text += f". Attributes: {attrs}"

            chunks.append({
                "id": f"{key}:entity:{entity.id}",
                "text": text,
                "type": "entity",
                "importance": entity.importance_score,
            })

        # Chunk facts
        for fact in ir.iter_facts():
            text = f"{fact.subject} {fact.predicate} {fact.object}"
            if fact.context:
                text += f" (context: {fact.context})"

            chunks.append({
                "id": f"{key}:fact:{fact.id}",
                "text": text,
                "type": "fact",
                "importance": fact.importance_score,
            })

        # Chunk events
        for event in ir.iter_events():
            text = f"Event: {event.description}"
            if event.participants:
                text += f" Participants: {', '.join(event.participants)}"

            chunks.append({
                "id": f"{key}:event:{event.id}",
                "text": text,
                "type": "event",
                "importance": event.importance_score,
            })

        # Chunk summaries
        for summary in ir.summaries:
            chunks.append({
                "id": f"{key}:summary:{uuid.uuid4().hex[:8]}",
                "text": summary,
                "type": "summary",
                "importance": 0.8,  # Summaries are generally important
            })

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_keys: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve relevant memory chunks for a query.

        Args:
            query: Search query.
            top_k: Number of results (overrides config).
            filter_keys: Only search within these memory keys.

        Returns:
            List of retrieval results sorted by relevance.
        """
        k = top_k or self.config.top_k
        query_embedding = self.embed_fn(query)

        # Build metadata filter
        filter_metadata = None
        if filter_keys and len(filter_keys) == 1:
            filter_metadata = {"source_key": filter_keys[0]}

        # Get candidates (fetch more for hybrid scoring)
        candidates = self.vector_store.search(
            query_embedding,
            top_k=k * 3 if self.config.strategy == RetrievalStrategy.HYBRID else k,
            filter_metadata=filter_metadata,
        )

        # Apply key filter if multiple keys
        if filter_keys and len(filter_keys) > 1:
            candidates = [
                (doc, score)
                for doc, score in candidates
                if doc.metadata.get("source_key") in filter_keys
            ]

        # Apply retrieval strategy
        if self.config.strategy == RetrievalStrategy.SEMANTIC:
            results = self._score_semantic(candidates)
        elif self.config.strategy == RetrievalStrategy.RECENCY:
            results = self._score_recency(candidates)
        elif self.config.strategy == RetrievalStrategy.IMPORTANCE:
            results = self._score_importance(candidates)
        else:  # HYBRID
            results = self._score_hybrid(candidates)

        # Filter by threshold
        results = [r for r in results if r.score >= self.config.similarity_threshold]

        # Rerank if enabled
        if self.config.rerank and results:
            results = self._rerank(query, results)

        return results[:k]

    def _score_semantic(
        self, candidates: List[Tuple[VectorDocument, float]]
    ) -> List[RetrievalResult]:
        """Score purely by semantic similarity."""
        results = []
        for doc, score in candidates:
            results.append(
                RetrievalResult(
                    text=doc.text,
                    score=score,
                    source_key=doc.metadata.get("source_key", ""),
                    chunk_id=doc.id,
                    metadata=doc.metadata,
                )
            )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _score_recency(
        self, candidates: List[Tuple[VectorDocument, float]]
    ) -> List[RetrievalResult]:
        """Score by recency with semantic as tiebreaker."""
        import time

        current_time = time.time()
        max_age = 86400 * 30  # 30 days

        results = []
        for doc, semantic_score in candidates:
            source_key = doc.metadata.get("source_key", "")
            timestamp = self._memory_timestamps.get(source_key, current_time)
            age = current_time - timestamp

            # Recency score: 1.0 for new, decays over time
            recency_score = max(0, 1 - (age / max_age))

            # Combine scores
            final_score = 0.7 * recency_score + 0.3 * semantic_score

            results.append(
                RetrievalResult(
                    text=doc.text,
                    score=final_score,
                    source_key=source_key,
                    chunk_id=doc.id,
                    metadata=doc.metadata,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _score_importance(
        self, candidates: List[Tuple[VectorDocument, float]]
    ) -> List[RetrievalResult]:
        """Score by importance with semantic as factor."""
        results = []
        for doc, semantic_score in candidates:
            importance = self._chunk_importance.get(doc.id, 0.5)

            # Combine scores
            final_score = 0.6 * importance + 0.4 * semantic_score

            results.append(
                RetrievalResult(
                    text=doc.text,
                    score=final_score,
                    source_key=doc.metadata.get("source_key", ""),
                    chunk_id=doc.id,
                    metadata=doc.metadata,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _score_hybrid(
        self, candidates: List[Tuple[VectorDocument, float]]
    ) -> List[RetrievalResult]:
        """Hybrid scoring combining semantic, recency, and importance."""
        import time

        current_time = time.time()
        max_age = 86400 * 30  # 30 days

        results = []
        for doc, semantic_score in candidates:
            source_key = doc.metadata.get("source_key", "")

            # Recency score
            timestamp = self._memory_timestamps.get(source_key, current_time)
            age = current_time - timestamp
            recency_score = max(0, 1 - (age / max_age))

            # Importance score
            importance = self._chunk_importance.get(doc.id, 0.5)

            # Weighted combination
            final_score = (
                self.config.semantic_weight * semantic_score
                + self.config.recency_weight * recency_score
                + self.config.importance_weight * importance
            )

            results.append(
                RetrievalResult(
                    text=doc.text,
                    score=final_score,
                    source_key=source_key,
                    chunk_id=doc.id,
                    metadata=doc.metadata,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _rerank(
        self, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder or other method."""
        # Simple reranking based on query term overlap
        query_terms = set(query.lower().split())

        for result in results:
            text_terms = set(result.text.lower().split())
            overlap = len(query_terms & text_terms) / (len(query_terms) + 1)
            result.score = result.score * (1 + 0.2 * overlap)

        return sorted(results, key=lambda r: r.score, reverse=True)

    def format_context(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 2048,
        include_sources: bool = True,
    ) -> str:
        """Format retrieval results as context for prompt injection.

        Args:
            results: Retrieval results to format.
            max_tokens: Maximum token budget.
            include_sources: Whether to include source references.

        Returns:
            Formatted context string.
        """
        if not results:
            return ""

        lines = ["Relevant Context:"]

        # Estimate ~4 chars per token
        char_budget = max_tokens * 4
        current_chars = len(lines[0])

        for i, result in enumerate(results, 1):
            if include_sources:
                line = f"{i}. [{result.source_key}] {result.text}"
            else:
                line = f"{i}. {result.text}"

            if current_chars + len(line) > char_budget:
                break

            lines.append(line)
            current_chars += len(line)

        return "\n".join(lines)

    def remove_memory(self, key: str) -> int:
        """Remove all chunks from a memory.

        Args:
            key: Memory key to remove.

        Returns:
            Number of chunks removed.
        """
        # Find all chunks for this key
        chunks_to_remove = [
            chunk_id
            for chunk_id in self._chunk_importance.keys()
            if chunk_id.startswith(f"{key}:")
        ]

        if not chunks_to_remove:
            return 0

        deleted = self.vector_store.delete(chunks_to_remove)

        # Clean up metadata
        for chunk_id in chunks_to_remove:
            self._chunk_importance.pop(chunk_id, None)

        self._memory_timestamps.pop(key, None)

        logger.debug(f"Removed {deleted} chunks for memory: {key}")
        return deleted

    def clear(self) -> None:
        """Clear all indexed memories."""
        self.vector_store.clear()
        self._memory_timestamps.clear()
        self._chunk_importance.clear()
        logger.info("Cleared all indexed memories")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_chunks": self.vector_store.count,
            "total_memories": len(self._memory_timestamps),
            "strategy": self.config.strategy.value,
            "config": {
                "top_k": self.config.top_k,
                "threshold": self.config.similarity_threshold,
            },
        }
