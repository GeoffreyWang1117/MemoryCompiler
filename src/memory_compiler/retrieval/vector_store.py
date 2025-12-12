"""Vector store implementations for embedding-based retrieval."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class VectorDocument:
    """A document with its embedding vector.

    Attributes:
        id: Unique document identifier.
        text: Original text content.
        embedding: Vector embedding.
        metadata: Additional document metadata.
    """

    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector storage backends.

    Provides semantic search over embedded documents using
    cosine similarity or other distance metrics.
    """

    @abstractmethod
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to the store.

        Args:
            documents: List of documents with embeddings.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of (document, score) tuples sorted by similarity.
        """
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents by ID.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass

    def save(self, path: str | Path) -> None:
        """Save the store to disk."""
        pass

    def load(self, path: str | Path) -> None:
        """Load the store from disk."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Get number of documents in store."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using numpy.

    Good for development and small-scale use.
    Uses brute-force cosine similarity search.

    Example:
        >>> store = InMemoryVectorStore()
        >>> store.add([VectorDocument(id="1", text="hello", embedding=np.array([0.1, 0.2]))])
        >>> results = store.search(np.array([0.1, 0.2]), top_k=5)
    """

    def __init__(self) -> None:
        self._documents: Dict[str, VectorDocument] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._needs_rebuild = True

    def _rebuild_index(self) -> None:
        """Rebuild the embedding matrix for efficient search."""
        if not self._documents:
            self._embeddings = None
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._needs_rebuild = False
            return

        self._id_to_idx = {}
        self._idx_to_id = {}

        embeddings = []
        for idx, (doc_id, doc) in enumerate(self._documents.items()):
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id
            embeddings.append(doc.embedding)

        self._embeddings = np.vstack(embeddings)
        self._needs_rebuild = False

    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to the store."""
        for doc in documents:
            self._documents[doc.id] = doc

        self._needs_rebuild = True
        logger.debug(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents using cosine similarity."""
        if self._needs_rebuild:
            self._rebuild_index()

        if self._embeddings is None or len(self._documents) == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Normalize embeddings
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normalized = self._embeddings / norms

        # Compute cosine similarities
        similarities = np.dot(normalized, query_norm)

        # Apply metadata filter
        if filter_metadata:
            for idx in range(len(similarities)):
                doc_id = self._idx_to_id[idx]
                doc = self._documents[doc_id]

                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        similarities[idx] = -1
                        break

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0:
                continue

            doc_id = self._idx_to_id[idx]
            doc = self._documents[doc_id]
            results.append((doc, score))

        return results

    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents by ID."""
        deleted = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                deleted += 1

        if deleted > 0:
            self._needs_rebuild = True

        return deleted

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
        self._embeddings = None
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._needs_rebuild = True

    @property
    def count(self) -> int:
        """Get number of documents."""
        return len(self._documents)

    def save(self, path: str | Path) -> None:
        """Save store to disk."""
        path = Path(path)

        data = {
            "documents": {
                doc_id: {
                    "id": doc.id,
                    "text": doc.text,
                    "embedding": doc.embedding.tolist(),
                    "metadata": doc.metadata,
                }
                for doc_id, doc in self._documents.items()
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str | Path) -> None:
        """Load store from disk."""
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._documents.clear()

        for doc_data in data["documents"].values():
            doc = VectorDocument(
                id=doc_data["id"],
                text=doc_data["text"],
                embedding=np.array(doc_data["embedding"]),
                metadata=doc_data["metadata"],
            )
            self._documents[doc.id] = doc

        self._needs_rebuild = True
        logger.info(f"Loaded {len(self._documents)} documents from {path}")


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search.

    Uses FAISS library for scalable approximate nearest neighbor search.
    Supports multiple index types for different accuracy/speed tradeoffs.

    Example:
        >>> store = FAISSVectorStore(dimension=384)
        >>> store.add([VectorDocument(...)])
        >>> results = store.search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        nlist: int = 100,
    ) -> None:
        """Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension.
            index_type: Index type - 'flat' for exact, 'ivf' for approximate.
            nlist: Number of clusters for IVF index.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist

        self._documents: Dict[str, VectorDocument] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._index = None
        self._faiss = None

        self._init_faiss()

    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss

            self._faiss = faiss

            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist
                )
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

            logger.debug(f"Initialized FAISS {self.index_type} index")

        except ImportError:
            logger.warning("FAISS not installed, falling back to numpy search")
            self._index = None

    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to the store."""
        if not documents:
            return

        embeddings = []
        for doc in documents:
            idx = len(self._documents)
            self._documents[doc.id] = doc
            self._id_to_idx[doc.id] = idx
            self._idx_to_id[idx] = doc.id

            # Normalize for inner product search
            norm = np.linalg.norm(doc.embedding) + 1e-10
            embeddings.append(doc.embedding / norm)

        embeddings_array = np.vstack(embeddings).astype(np.float32)

        if self._index is not None and self._faiss is not None:
            if self.index_type == "ivf" and not self._index.is_trained:
                if embeddings_array.shape[0] >= self.nlist:
                    self._index.train(embeddings_array)

            if self.index_type != "ivf" or self._index.is_trained:
                self._index.add(embeddings_array)

        logger.debug(f"Added {len(documents)} documents to FAISS store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents."""
        if len(self._documents) == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        query_array = query_norm.reshape(1, -1).astype(np.float32)

        if self._index is not None and self._index.ntotal > 0:
            # Use FAISS search
            scores, indices = self._index.search(query_array, min(top_k * 2, self._index.ntotal))
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to numpy
            embeddings = []
            for idx in range(len(self._idx_to_id)):
                doc_id = self._idx_to_id[idx]
                doc = self._documents[doc_id]
                norm = np.linalg.norm(doc.embedding) + 1e-10
                embeddings.append(doc.embedding / norm)

            embeddings_array = np.vstack(embeddings)
            scores = np.dot(embeddings_array, query_norm)
            indices = np.argsort(scores)[::-1][: top_k * 2]
            scores = scores[indices]

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self._idx_to_id):
                continue

            doc_id = self._idx_to_id[int(idx)]
            doc = self._documents[doc_id]

            # Apply metadata filter
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append((doc, float(score)))

            if len(results) >= top_k:
                break

        return results

    def delete(self, doc_ids: List[str]) -> int:
        """Delete documents (requires index rebuild)."""
        deleted = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                deleted += 1

        if deleted > 0:
            # Rebuild index
            self._rebuild_index()

        return deleted

    def _rebuild_index(self) -> None:
        """Rebuild the entire index."""
        old_docs = list(self._documents.values())

        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._init_faiss()

        if old_docs:
            self.add(old_docs)

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._init_faiss()

    @property
    def count(self) -> int:
        """Get number of documents."""
        return len(self._documents)

    def save(self, path: str | Path) -> None:
        """Save store to disk."""
        path = Path(path)

        # Save documents
        docs_data = {
            doc_id: {
                "id": doc.id,
                "text": doc.text,
                "embedding": doc.embedding.tolist(),
                "metadata": doc.metadata,
            }
            for doc_id, doc in self._documents.items()
        }

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(docs_data, f)

        # Save FAISS index
        if self._index is not None and self._faiss is not None:
            self._faiss.write_index(self._index, str(path.with_suffix(".faiss")))

        logger.info(f"Saved FAISS store to {path}")

    def load(self, path: str | Path) -> None:
        """Load store from disk."""
        path = Path(path)

        # Load documents
        with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()

        for idx, (doc_id, doc_data) in enumerate(docs_data.items()):
            doc = VectorDocument(
                id=doc_data["id"],
                text=doc_data["text"],
                embedding=np.array(doc_data["embedding"]),
                metadata=doc_data["metadata"],
            )
            self._documents[doc.id] = doc
            self._id_to_idx[doc.id] = idx
            self._idx_to_id[idx] = doc.id

        # Load FAISS index
        if self._faiss is not None and path.with_suffix(".faiss").exists():
            self._index = self._faiss.read_index(str(path.with_suffix(".faiss")))

        logger.info(f"Loaded {len(self._documents)} documents from {path}")
