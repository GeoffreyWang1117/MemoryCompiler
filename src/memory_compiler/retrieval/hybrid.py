"""Hybrid retrieval combining keyword and semantic search."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.retrieval.retriever import RetrievalResult


@dataclass
class KeywordDocument:
    """Document for keyword-based retrieval.

    Attributes:
        id: Document identifier.
        text: Original text.
        tokens: Tokenized and normalized text.
        term_freq: Term frequency map.
        metadata: Additional metadata.
    """

    id: str
    text: str
    tokens: List[str]
    term_freq: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeywordRetriever:
    """BM25-based keyword retriever.

    Implements BM25 scoring for keyword-based retrieval.
    Good for exact term matching and rare terms.

    Example:
        >>> retriever = KeywordRetriever()
        >>> retriever.index_text("doc1", "Hello world from Python")
        >>> results = retriever.search("Python programming")
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """Initialize keyword retriever.

        Args:
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            stopwords: Set of words to ignore.
        """
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords or self._default_stopwords()

        self._documents: Dict[str, KeywordDocument] = {}
        self._doc_freq: Dict[str, int] = Counter()
        self._avg_doc_len: float = 0.0
        self._total_docs: int = 0

    def _default_stopwords(self) -> Set[str]:
        """Get default English stopwords."""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "were", "will", "with", "the",
            "this", "but", "they", "have", "had", "what", "when", "where",
            "who", "which", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "can", "just", "should", "now", "i", "you", "your", "we",
            "our", "my", "me", "him", "her", "them", "their",
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        return tokens

    def index_text(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a text document.

        Args:
            doc_id: Document identifier.
            text: Text content.
            metadata: Optional metadata.
        """
        tokens = self._tokenize(text)
        term_freq = Counter(tokens)

        doc = KeywordDocument(
            id=doc_id,
            text=text,
            tokens=tokens,
            term_freq=dict(term_freq),
            metadata=metadata or {},
        )

        # Update document frequency
        for term in set(tokens):
            self._doc_freq[term] += 1

        self._documents[doc_id] = doc
        self._total_docs = len(self._documents)

        # Update average document length
        total_tokens = sum(len(d.tokens) for d in self._documents.values())
        self._avg_doc_len = total_tokens / self._total_docs if self._total_docs > 0 else 0

    def index_memory(self, key: str, ir: MemoryIR) -> int:
        """Index a Memory IR for keyword search.

        Args:
            key: Memory key.
            ir: Memory IR to index.

        Returns:
            Number of documents indexed.
        """
        count = 0

        # Index entities
        for entity in ir.iter_entities():
            text = f"{entity.name}"
            if entity.aliases:
                text += f" {' '.join(entity.aliases)}"
            if entity.attributes:
                text += " " + " ".join(f"{k} {v}" for k, v in entity.attributes.items())

            self.index_text(
                f"{key}:entity:{entity.id}",
                text,
                {"source_key": key, "type": "entity", "importance": entity.importance_score},
            )
            count += 1

        # Index facts
        for fact in ir.iter_facts():
            text = f"{fact.subject} {fact.predicate} {fact.object}"
            if fact.context:
                text += f" {fact.context}"

            self.index_text(
                f"{key}:fact:{fact.id}",
                text,
                {"source_key": key, "type": "fact", "importance": fact.importance_score},
            )
            count += 1

        # Index events
        for event in ir.iter_events():
            text = f"{event.description}"
            if event.participants:
                text += f" {' '.join(event.participants)}"

            self.index_text(
                f"{key}:event:{event.id}",
                text,
                {"source_key": key, "type": "event", "importance": event.importance_score},
            )
            count += 1

        logger.debug(f"Indexed {count} documents for keyword search: {key}")
        return count

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_keys: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Search for documents using BM25 scoring.

        Args:
            query: Search query.
            top_k: Number of results.
            filter_keys: Only search within these memory keys.

        Returns:
            List of results sorted by BM25 score.
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = []

        for doc_id, doc in self._documents.items():
            # Apply key filter
            if filter_keys:
                source_key = doc.metadata.get("source_key", "")
                if source_key not in filter_keys:
                    continue

            score = self._bm25_score(query_tokens, doc)
            if score > 0:
                scores.append((doc, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Convert to results
        results = []
        for doc, score in scores[:top_k]:
            results.append(
                RetrievalResult(
                    text=doc.text,
                    score=score,
                    source_key=doc.metadata.get("source_key", ""),
                    chunk_id=doc.id,
                    metadata=doc.metadata,
                )
            )

        return results

    def _bm25_score(self, query_tokens: List[str], doc: KeywordDocument) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_len = len(doc.tokens)

        for term in query_tokens:
            if term not in doc.term_freq:
                continue

            tf = doc.term_freq[term]
            df = self._doc_freq.get(term, 0)

            if df == 0:
                continue

            # IDF component
            idf = np.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)

            # TF component with length normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_doc_len + 1e-10))
            )

            score += idf * tf_norm

        return score

    def remove(self, doc_ids: List[str]) -> int:
        """Remove documents from index."""
        removed = 0

        for doc_id in doc_ids:
            if doc_id in self._documents:
                doc = self._documents[doc_id]

                # Update document frequencies
                for term in set(doc.tokens):
                    self._doc_freq[term] = max(0, self._doc_freq[term] - 1)

                del self._documents[doc_id]
                removed += 1

        # Update stats
        self._total_docs = len(self._documents)
        if self._total_docs > 0:
            total_tokens = sum(len(d.tokens) for d in self._documents.values())
            self._avg_doc_len = total_tokens / self._total_docs
        else:
            self._avg_doc_len = 0

        return removed

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._doc_freq.clear()
        self._avg_doc_len = 0.0
        self._total_docs = 0


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    multiple retrieval methods.

    Example:
        >>> retriever = HybridRetriever(embed_fn=embed_function)
        >>> retriever.index_memory("session_1", memory_ir)
        >>> results = retriever.retrieve("What is the user's name?")
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            embed_fn: Function to embed text into vectors.
            semantic_weight: Weight for semantic results (0-1).
            keyword_weight: Weight for keyword results (0-1).
            rrf_k: RRF smoothing parameter.
        """
        from memory_compiler.retrieval.retriever import MemoryRetriever

        self.semantic_retriever = MemoryRetriever(embed_fn=embed_fn)
        self.keyword_retriever = KeywordRetriever()

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

    def index_memory(
        self,
        key: str,
        ir: MemoryIR,
        timestamp: Optional[float] = None,
    ) -> int:
        """Index a Memory IR for both retrieval methods.

        Args:
            key: Memory key.
            ir: Memory IR to index.
            timestamp: Optional timestamp.

        Returns:
            Number of chunks indexed.
        """
        semantic_count = self.semantic_retriever.index_memory(key, ir, timestamp)
        keyword_count = self.keyword_retriever.index_memory(key, ir)

        return max(semantic_count, keyword_count)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_keys: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid search.

        Args:
            query: Search query.
            top_k: Number of results.
            filter_keys: Only search within these memory keys.

        Returns:
            Fused results from both retrievers.
        """
        # Get results from both retrievers
        semantic_results = self.semantic_retriever.retrieve(
            query, top_k=top_k * 2, filter_keys=filter_keys
        )
        keyword_results = self.keyword_retriever.search(
            query, top_k=top_k * 2, filter_keys=filter_keys
        )

        # Apply RRF fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results, keyword_results
        )

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) for each ranking
        """
        scores: Dict[str, float] = {}
        results_map: Dict[str, RetrievalResult] = {}

        # Score semantic results
        for rank, result in enumerate(semantic_results, 1):
            rrf_score = self.semantic_weight / (self.rrf_k + rank)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + rrf_score
            results_map[result.chunk_id] = result

        # Score keyword results
        for rank, result in enumerate(keyword_results, 1):
            rrf_score = self.keyword_weight / (self.rrf_k + rank)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + rrf_score

            if result.chunk_id not in results_map:
                results_map[result.chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results with fused scores
        fused_results = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id]
            fused_results.append(
                RetrievalResult(
                    text=result.text,
                    score=scores[chunk_id],
                    source_key=result.source_key,
                    chunk_id=result.chunk_id,
                    metadata=result.metadata,
                )
            )

        return fused_results

    def format_context(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 2048,
        include_sources: bool = True,
    ) -> str:
        """Format results as context string."""
        return self.semantic_retriever.format_context(
            results, max_tokens, include_sources
        )

    def remove_memory(self, key: str) -> int:
        """Remove all chunks for a memory."""
        semantic_removed = self.semantic_retriever.remove_memory(key)

        # Remove from keyword retriever
        to_remove = [
            doc_id
            for doc_id in self.keyword_retriever._documents.keys()
            if doc_id.startswith(f"{key}:")
        ]
        keyword_removed = self.keyword_retriever.remove(to_remove)

        return max(semantic_removed, keyword_removed)

    def clear(self) -> None:
        """Clear all indexed memories."""
        self.semantic_retriever.clear()
        self.keyword_retriever.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "semantic": self.semantic_retriever.get_stats(),
            "keyword": {
                "total_documents": len(self.keyword_retriever._documents),
                "vocabulary_size": len(self.keyword_retriever._doc_freq),
            },
            "weights": {
                "semantic": self.semantic_weight,
                "keyword": self.keyword_weight,
            },
        }
