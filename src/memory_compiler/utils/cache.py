"""Caching utilities for MemoryCompiler.

This module provides caching mechanisms for embeddings, extraction results,
and other computed values to improve performance.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from loguru import logger


T = TypeVar("T")


class EmbeddingCache:
    """Cache for text embeddings.

    Stores computed embeddings to avoid redundant computation.
    Supports both in-memory and disk-based caching.

    Attributes:
        cache_dir: Directory for disk cache.
        max_memory_items: Maximum items in memory cache.
        ttl: Time-to-live for cache entries in seconds.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_memory_items: int = 10000,
        ttl: int = 3600,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.ttl = ttl

        self._memory_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._access_order: list[str] = []

        if self.cache_dir and self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Get embedding from cache."""
        if not self.enabled:
            return None

        key = self._hash_text(text)

        # Check memory cache
        if key in self._memory_cache:
            embedding, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end of access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return embedding
            else:
                # Expired
                del self._memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    mtime = cache_file.stat().st_mtime
                    if time.time() - mtime < self.ttl:
                        embedding = np.load(cache_file)
                        self._memory_cache[key] = (embedding, time.time())
                        self._access_order.append(key)
                        return embedding
                    else:
                        cache_file.unlink()  # Remove expired
                except Exception as e:
                    logger.debug(f"Failed to load cached embedding: {e}")

        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        if not self.enabled:
            return

        key = self._hash_text(text)

        # Evict if necessary
        while len(self._memory_cache) >= self.max_memory_items:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._memory_cache:
                    del self._memory_cache[oldest_key]

        # Store in memory
        self._memory_cache[key] = (embedding, time.time())
        self._access_order.append(key)

        # Store on disk
        if self.cache_dir:
            try:
                cache_file = self.cache_dir / f"{key}.npy"
                np.save(cache_file, embedding)
            except Exception as e:
                logger.debug(f"Failed to save embedding to disk: {e}")

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Get embeddings for multiple texts.

        Returns:
            Tuple of (cached_embeddings, missing_indices).
            cached_embeddings has None for texts not in cache.
            missing_indices lists indices of texts needing computation.
        """
        results = []
        missing = []

        for i, text in enumerate(texts):
            embedding = self.get(text)
            results.append(embedding)
            if embedding is None:
                missing.append(i)

        return results, missing

    def set_batch(self, texts: list[str], embeddings: list[np.ndarray]) -> None:
        """Store multiple embeddings in cache."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        self._access_order.clear()

        if self.cache_dir and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.npy"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        disk_items = 0
        disk_size = 0

        if self.cache_dir and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.npy"):
                disk_items += 1
                disk_size += f.stat().st_size

        return {
            "memory_items": len(self._memory_cache),
            "disk_items": disk_items,
            "disk_size_mb": disk_size / (1024 * 1024),
            "max_memory_items": self.max_memory_items,
            "ttl": self.ttl,
        }


class ResultCache:
    """General-purpose cache for computed results.

    Supports caching of extraction results, compression results,
    and other serializable objects.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_items: int = 1000,
        ttl: int = 3600,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_items = max_items
        self.ttl = ttl

        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_order: list[str] = []

        if self.cache_dir and self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get cached result."""
        if not self.enabled:
            return None

        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self._cache[key]

        # Check disk
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    mtime = cache_file.stat().st_mtime
                    if time.time() - mtime < self.ttl:
                        with open(cache_file, "rb") as f:
                            return pickle.load(f)
                    else:
                        cache_file.unlink()
                except Exception:
                    pass

        return None

    def set(self, key: str, value: Any) -> None:
        """Store result in cache."""
        if not self.enabled:
            return

        # Evict if necessary
        while len(self._cache) >= self.max_items:
            if self._access_order:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    del self._cache[oldest]

        self._cache[key] = (value, time.time())
        self._access_order.append(key)

        # Store on disk
        if self.cache_dir:
            try:
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, "wb") as f:
                    pickle.dump(value, f)
            except Exception:
                pass

    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to cache function results."""
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = self._make_key(func.__name__, *args, **kwargs)

            result = self.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            self.set(key, result)
            return result

        return wrapper

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._access_order.clear()

        if self.cache_dir and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.pkl"):
                try:
                    f.unlink()
                except Exception:
                    pass


class DialogueCache:
    """Cache specifically for dialogue processing results.

    Caches Memory IR extractions keyed by dialogue content hash.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir and self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_dialogue(self, dialogue: list[dict[str, str]]) -> str:
        """Generate hash for dialogue content."""
        content = json.dumps(dialogue, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_ir(self, dialogue: list[dict[str, str]]) -> dict[str, Any] | None:
        """Get cached Memory IR for dialogue."""
        if not self.enabled or not self.cache_dir:
            return None

        key = self._hash_dialogue(dialogue)
        cache_file = self.cache_dir / f"ir_{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    def set_ir(self, dialogue: list[dict[str, str]], ir_dict: dict[str, Any]) -> None:
        """Cache Memory IR for dialogue."""
        if not self.enabled or not self.cache_dir:
            return

        key = self._hash_dialogue(dialogue)
        cache_file = self.cache_dir / f"ir_{key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(ir_dict, f)
        except Exception:
            pass
