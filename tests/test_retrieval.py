"""Tests for RAG-style retrieval."""

import numpy as np
import pytest

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.retrieval import (
    HybridRetriever,
    InMemoryVectorStore,
    MemoryRetriever,
    RetrievalConfig,
    RetrievalStrategy,
    VectorDocument,
)
from memory_compiler.retrieval.hybrid import KeywordRetriever


def create_test_ir(name: str = "Alice", job: str = "engineer") -> MemoryIR:
    """Create a test Memory IR."""
    ir = MemoryIR(session_id=f"test_{name}")

    ir.add_entity(Entity(id=f"e_{name}", name=name, type=EntityType.PERSON))
    ir.add_fact(Fact(
        id=f"f_{name}",
        subject=name,
        predicate="works as",
        object=job,
    ))

    return ir


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    def test_add_and_search(self):
        """Test adding documents and searching."""
        store = InMemoryVectorStore()

        # Add documents
        docs = [
            VectorDocument(
                id="doc1",
                text="Hello world",
                embedding=np.array([1.0, 0.0, 0.0]),
            ),
            VectorDocument(
                id="doc2",
                text="Python programming",
                embedding=np.array([0.0, 1.0, 0.0]),
            ),
        ]
        store.add(docs)

        assert store.count == 2

        # Search
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, top_k=1)

        assert len(results) == 1
        assert results[0][0].id == "doc1"
        assert results[0][1] > 0.9

    def test_delete(self):
        """Test deleting documents."""
        store = InMemoryVectorStore()

        docs = [
            VectorDocument(id="d1", text="A", embedding=np.array([1, 0])),
            VectorDocument(id="d2", text="B", embedding=np.array([0, 1])),
        ]
        store.add(docs)

        deleted = store.delete(["d1"])
        assert deleted == 1
        assert store.count == 1

    def test_metadata_filter(self):
        """Test filtering by metadata."""
        store = InMemoryVectorStore()

        docs = [
            VectorDocument(
                id="d1",
                text="A",
                embedding=np.array([1, 0, 0]),
                metadata={"type": "entity"},
            ),
            VectorDocument(
                id="d2",
                text="B",
                embedding=np.array([1, 0.1, 0]),
                metadata={"type": "fact"},
            ),
        ]
        store.add(docs)

        # Search with filter
        results = store.search(
            np.array([1, 0, 0]),
            top_k=2,
            filter_metadata={"type": "entity"},
        )

        assert len(results) == 1
        assert results[0][0].id == "d1"

    def test_clear(self):
        """Test clearing store."""
        store = InMemoryVectorStore()
        store.add([VectorDocument(id="d1", text="A", embedding=np.array([1, 0]))])
        assert store.count == 1

        store.clear()
        assert store.count == 0


class TestKeywordRetriever:
    """Tests for KeywordRetriever."""

    def test_index_and_search(self):
        """Test indexing and BM25 search."""
        retriever = KeywordRetriever()

        retriever.index_text("doc1", "Python is a great programming language")
        retriever.index_text("doc2", "Java is also a popular language")

        results = retriever.search("Python programming", top_k=2)

        assert len(results) >= 1
        assert results[0].chunk_id == "doc1"

    def test_search_with_filter(self):
        """Test search with key filter."""
        retriever = KeywordRetriever()

        retriever.index_text("key1:doc1", "Python programming", {"source_key": "key1"})
        retriever.index_text("key2:doc2", "Python development", {"source_key": "key2"})

        results = retriever.search("Python", top_k=2, filter_keys=["key1"])

        assert len(results) == 1
        assert "key1" in results[0].chunk_id

    def test_index_memory(self):
        """Test indexing Memory IR."""
        retriever = KeywordRetriever()
        ir = create_test_ir("Alice", "engineer")

        count = retriever.index_memory("test", ir)
        assert count >= 1

        results = retriever.search("Alice")
        assert len(results) >= 1


class TestMemoryRetriever:
    """Tests for MemoryRetriever."""

    def test_index_and_retrieve(self):
        """Test indexing and retrieving."""
        retriever = MemoryRetriever()
        ir = create_test_ir("Alice", "data scientist")

        chunks = retriever.index_memory("alice_session", ir)
        assert chunks >= 1

        results = retriever.retrieve("Alice", top_k=5)
        assert len(results) >= 1

    def test_retrieval_strategies(self):
        """Test different retrieval strategies."""
        configs = [
            RetrievalConfig(strategy=RetrievalStrategy.SEMANTIC),
            RetrievalConfig(strategy=RetrievalStrategy.RECENCY),
            RetrievalConfig(strategy=RetrievalStrategy.IMPORTANCE),
            RetrievalConfig(strategy=RetrievalStrategy.HYBRID),
        ]

        for config in configs:
            retriever = MemoryRetriever(config=config)
            ir = create_test_ir()
            retriever.index_memory("test", ir)

            results = retriever.retrieve("test query")
            # Should not raise and should return results
            assert isinstance(results, list)

    def test_format_context(self):
        """Test formatting results as context."""
        retriever = MemoryRetriever()
        ir = create_test_ir("Alice", "engineer")
        retriever.index_memory("session", ir)

        results = retriever.retrieve("Alice", top_k=3)
        context = retriever.format_context(results, max_tokens=500)

        assert "Relevant Context:" in context
        assert len(context) > 0

    def test_remove_memory(self):
        """Test removing indexed memory."""
        retriever = MemoryRetriever()
        ir = create_test_ir()

        retriever.index_memory("to_remove", ir)
        assert retriever.vector_store.count > 0

        removed = retriever.remove_memory("to_remove")
        assert removed > 0

    def test_filter_by_keys(self):
        """Test filtering retrieval by keys."""
        retriever = MemoryRetriever()

        ir1 = create_test_ir("Alice", "engineer")
        ir2 = create_test_ir("Bob", "manager")

        retriever.index_memory("alice_session", ir1)
        retriever.index_memory("bob_session", ir2)

        # Filter to only Alice's session
        results = retriever.retrieve("person", filter_keys=["alice_session"])

        for result in results:
            assert result.source_key == "alice_session"


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_index_and_retrieve(self):
        """Test hybrid indexing and retrieval."""
        retriever = HybridRetriever(
            semantic_weight=0.6,
            keyword_weight=0.4,
        )

        ir = create_test_ir("Alice", "software engineer")
        chunks = retriever.index_memory("test_session", ir)
        assert chunks >= 1

        results = retriever.retrieve("Alice software", top_k=5)
        assert len(results) >= 1

    def test_rrf_fusion(self):
        """Test that RRF fusion combines results."""
        retriever = HybridRetriever()

        # Create two memories with different content
        ir1 = MemoryIR(session_id="semantic_match")
        ir1.add_entity(Entity(id="e1", name="Semantic content here", type=EntityType.CONCEPT))

        ir2 = MemoryIR(session_id="keyword_match")
        ir2.add_fact(Fact(id="f1", subject="Exact", predicate="keyword", object="match"))

        retriever.index_memory("sem", ir1)
        retriever.index_memory("kw", ir2)

        # Retrieve - should get results from both retrievers
        results = retriever.retrieve("content keyword", top_k=5)

        # Results should be fused
        assert isinstance(results, list)

    def test_format_context(self):
        """Test context formatting."""
        retriever = HybridRetriever()
        ir = create_test_ir()
        retriever.index_memory("test", ir)

        results = retriever.retrieve("test", top_k=3)
        context = retriever.format_context(results)

        assert isinstance(context, str)

    def test_get_stats(self):
        """Test getting statistics."""
        retriever = HybridRetriever()
        ir = create_test_ir()
        retriever.index_memory("test", ir)

        stats = retriever.get_stats()

        assert "semantic" in stats
        assert "keyword" in stats
        assert "weights" in stats

    def test_remove_memory(self):
        """Test removing from both retrievers."""
        retriever = HybridRetriever()
        ir = create_test_ir()
        retriever.index_memory("to_remove", ir)

        removed = retriever.remove_memory("to_remove")
        assert removed > 0

    def test_clear(self):
        """Test clearing both retrievers."""
        retriever = HybridRetriever()
        ir = create_test_ir()
        retriever.index_memory("test", ir)

        retriever.clear()

        stats = retriever.get_stats()
        assert stats["semantic"]["total_chunks"] == 0
