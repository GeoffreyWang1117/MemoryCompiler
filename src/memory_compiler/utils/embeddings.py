"""Embedding management utilities for MemoryCompiler.

This module provides efficient embedding computation with caching,
batching, and support for multiple embedding models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from memory_compiler.utils.cache import EmbeddingCache


class EmbeddingManager:
    """Manages embedding computation with caching and batching.

    This class provides a unified interface for computing embeddings
    using various models (sentence-transformers, OpenAI, etc.) with
    automatic caching and efficient batching.

    Attributes:
        model_name: Name of the embedding model.
        cache: Embedding cache instance.
        batch_size: Batch size for embedding computation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
        cache_enabled: bool = True,
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self._model = None
        self._model_type = self._detect_model_type(model_name)

        # Initialize cache
        self.cache = EmbeddingCache(
            cache_dir=cache_dir,
            enabled=cache_enabled,
        )

        logger.info(f"EmbeddingManager initialized with model: {model_name}")

    def _detect_model_type(self, model_name: str) -> str:
        """Detect the type of embedding model."""
        if "sentence-transformers" in model_name or model_name.startswith("all-"):
            return "sentence_transformers"
        elif "openai" in model_name.lower() or model_name.startswith("text-embedding"):
            return "openai"
        elif "cohere" in model_name.lower():
            return "cohere"
        else:
            # Default to sentence-transformers
            return "sentence_transformers"

    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return

        if self._model_type == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)

                # Move to device if specified
                if self.device != "auto":
                    self._model = self._model.to(self.device)

                logger.info(f"Loaded sentence-transformers model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise

        elif self._model_type == "openai":
            try:
                import openai
                self._model = openai
                logger.info("Using OpenAI embeddings API")
            except ImportError:
                logger.error("openai not installed. Run: pip install openai")
                raise

        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

    def embed(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        self._load_model()

        if self._model_type == "sentence_transformers":
            embedding = self._model.encode(text, convert_to_numpy=True)
        elif self._model_type == "openai":
            response = self._model.embeddings.create(
                model=self.model_name,
                input=text,
            )
            embedding = np.array(response.data[0].embedding)
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

        # Cache and return
        self.cache.set(text, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        # Check cache for existing embeddings
        cached_results, missing_indices = self.cache.get_batch(texts)

        # If all cached, return immediately
        if not missing_indices:
            return np.array([e for e in cached_results if e is not None])

        # Compute missing embeddings
        self._load_model()
        missing_texts = [texts[i] for i in missing_indices]

        if self._model_type == "sentence_transformers":
            # Process in batches
            missing_embeddings = []
            for i in range(0, len(missing_texts), self.batch_size):
                batch = missing_texts[i:i + self.batch_size]
                batch_embeddings = self._model.encode(batch, convert_to_numpy=True)
                missing_embeddings.extend(batch_embeddings)

        elif self._model_type == "openai":
            # OpenAI handles batching internally
            response = self._model.embeddings.create(
                model=self.model_name,
                input=missing_texts,
            )
            missing_embeddings = [np.array(d.embedding) for d in response.data]

        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

        # Cache computed embeddings
        self.cache.set_batch(missing_texts, missing_embeddings)

        # Combine cached and computed results
        result = []
        missing_iter = iter(missing_embeddings)

        for i, cached in enumerate(cached_results):
            if cached is not None:
                result.append(cached)
            else:
                result.append(next(missing_iter))

        return np.array(result)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        return self._cosine_similarity(emb1, emb2)

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Compute pairwise similarity matrix for texts.

        Args:
            texts: List of texts.

        Returns:
            Similarity matrix of shape (len(texts), len(texts)).
        """
        embeddings = self.embed_batch(texts)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        # Compute similarity matrix
        return np.dot(normalized, normalized.T)

    def most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find most similar texts to a query.

        Args:
            query: Query text.
            candidates: List of candidate texts.
            top_k: Number of top results to return.

        Returns:
            List of (text, similarity) tuples, sorted by similarity descending.
        """
        if not candidates:
            return []

        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        # Compute similarities
        similarities = [
            self._cosine_similarity(query_emb, cand_emb)
            for cand_emb in candidate_embs
        ]

        # Sort and return top-k
        results = list(zip(candidates, similarities))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def cluster_texts(
        self,
        texts: list[str],
        n_clusters: int | None = None,
        min_cluster_size: int = 2,
    ) -> dict[int, list[str]]:
        """Cluster texts by semantic similarity.

        Args:
            texts: List of texts to cluster.
            n_clusters: Number of clusters (auto-detected if None).
            min_cluster_size: Minimum texts per cluster.

        Returns:
            Dictionary mapping cluster ID to list of texts.
        """
        if len(texts) < 2:
            return {0: texts}

        embeddings = self.embed_batch(texts)

        try:
            from sklearn.cluster import AgglomerativeClustering

            if n_clusters is None:
                n_clusters = max(1, len(texts) // 5)

            n_clusters = min(n_clusters, len(texts))

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)

            # Group texts by cluster
            clusters: dict[int, list[str]] = {}
            for text, label in zip(texts, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(text)

            return clusters

        except ImportError:
            logger.warning("sklearn not installed, returning single cluster")
            return {0: texts}

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        self._load_model()

        if self._model_type == "sentence_transformers":
            return self._model.get_sentence_embedding_dimension()
        elif self._model_type == "openai":
            # Common OpenAI embedding dimensions
            dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            return dims.get(self.model_name, 1536)

        return 384  # Default

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()
