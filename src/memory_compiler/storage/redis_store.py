"""Redis storage backend for Memory IR with caching support."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger

from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.storage.base import MemoryStore


class RedisMemoryStore(MemoryStore):
    """Redis-based storage for Memory IR with caching capabilities.

    Provides fast caching with optional TTL, secondary indices for
    entity/fact search, and connection pooling.

    Example:
        >>> store = RedisMemoryStore(host="localhost", port=6379)
        >>> store.save("session_1", memory_ir, ttl=3600)  # 1 hour TTL
        >>> ir = store.load("session_1")
        >>> keys = store.search_by_entity("Alice")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "memorycompiler:",
        default_ttl: Optional[int] = None,
        use_json: bool = True,
        connection_pool_size: int = 10,
        ssl: bool = False,
        decode_responses: bool = True,
    ) -> None:
        """Initialize Redis store.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Redis password.
            prefix: Key prefix for namespacing.
            default_ttl: Default TTL in seconds (None for no expiry).
            use_json: Use JSON serialization (vs pickle).
            connection_pool_size: Connection pool size.
            ssl: Enable SSL/TLS.
            decode_responses: Decode byte responses to strings.
        """
        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.use_json = use_json

        self._redis: Optional[Any] = None
        self._pool: Optional[Any] = None

        # Initialize connection
        self._init_connection(
            host, port, db, password,
            connection_pool_size, ssl, decode_responses
        )

    def _init_connection(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str],
        pool_size: int,
        ssl: bool,
        decode_responses: bool,
    ) -> None:
        """Initialize Redis connection with pool."""
        try:
            import redis

            self._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=pool_size,
                ssl=ssl,
                decode_responses=decode_responses,
            )

            self._redis = redis.Redis(connection_pool=self._pool)

            # Test connection
            self._redis.ping()
            logger.debug(f"Connected to Redis at {host}:{port}")

        except ImportError:
            raise ImportError(
                "Redis support requires 'redis' package. "
                "Install with: pip install redis"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}memory:{key}"

    def _make_metadata_key(self, key: str) -> str:
        """Create metadata key."""
        return f"{self.prefix}meta:{key}"

    def _make_entity_index_key(self, entity_name: str) -> str:
        """Create entity index key."""
        return f"{self.prefix}idx:entity:{entity_name.lower()}"

    def _make_fact_index_key(self, subject: str, predicate: Optional[str] = None) -> str:
        """Create fact index key."""
        if predicate:
            return f"{self.prefix}idx:fact:{subject.lower()}:{predicate.lower()}"
        return f"{self.prefix}idx:fact:{subject.lower()}"

    def _serialize(self, ir: MemoryIR) -> Union[str, bytes]:
        """Serialize Memory IR."""
        if self.use_json:
            return json.dumps(ir.to_dict())
        else:
            import pickle
            return pickle.dumps(ir.to_dict())

    def _deserialize(self, data: Union[str, bytes]) -> MemoryIR:
        """Deserialize Memory IR."""
        if self.use_json:
            ir_dict = json.loads(data) if isinstance(data, str) else json.loads(data.decode())
        else:
            import pickle
            ir_dict = pickle.loads(data if isinstance(data, bytes) else data.encode())

        return MemoryIR.from_dict(ir_dict)

    def save(
        self,
        key: str,
        ir: MemoryIR,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Save Memory IR to Redis.

        Args:
            key: Unique identifier.
            ir: Memory IR to save.
            metadata: Optional metadata.
            ttl: TTL in seconds (overrides default).
        """
        redis_key = self._make_key(key)
        data = self._serialize(ir)

        # Use pipeline for atomic operations
        pipe = self._redis.pipeline()

        # Set main data
        effective_ttl = ttl if ttl is not None else self.default_ttl
        if effective_ttl:
            pipe.setex(redis_key, effective_ttl, data)
        else:
            pipe.set(redis_key, data)

        # Save metadata
        if metadata:
            meta_key = self._make_metadata_key(key)
            if effective_ttl:
                pipe.setex(meta_key, effective_ttl, json.dumps(metadata))
            else:
                pipe.set(meta_key, json.dumps(metadata))

        # Update indices
        self._update_indices(pipe, key, ir, effective_ttl)

        pipe.execute()
        logger.debug(f"Saved memory to Redis: {key}")

    def _update_indices(
        self,
        pipe: Any,
        key: str,
        ir: MemoryIR,
        ttl: Optional[int],
    ) -> None:
        """Update secondary indices."""
        # Index by entity names
        for entity in ir.iter_entities():
            index_key = self._make_entity_index_key(entity.name)
            pipe.sadd(index_key, key)
            if ttl:
                pipe.expire(index_key, ttl)

            # Also index aliases
            for alias in entity.aliases:
                alias_key = self._make_entity_index_key(alias)
                pipe.sadd(alias_key, key)
                if ttl:
                    pipe.expire(alias_key, ttl)

        # Index by facts (subject and subject+predicate)
        for fact in ir.iter_facts():
            # Index by subject
            subj_key = self._make_fact_index_key(fact.subject)
            pipe.sadd(subj_key, key)
            if ttl:
                pipe.expire(subj_key, ttl)

            # Index by subject + predicate
            subj_pred_key = self._make_fact_index_key(fact.subject, fact.predicate)
            pipe.sadd(subj_pred_key, key)
            if ttl:
                pipe.expire(subj_pred_key, ttl)

    def load(self, key: str) -> Optional[MemoryIR]:
        """Load Memory IR from Redis.

        Args:
            key: Identifier to load.

        Returns:
            Memory IR or None if not found.
        """
        redis_key = self._make_key(key)
        data = self._redis.get(redis_key)

        if data is None:
            return None

        try:
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Failed to deserialize memory {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete Memory IR from Redis.

        Args:
            key: Identifier to delete.

        Returns:
            True if deleted.
        """
        redis_key = self._make_key(key)
        meta_key = self._make_metadata_key(key)

        # First load to remove from indices
        ir = self.load(key)
        if ir:
            self._remove_from_indices(key, ir)

        # Delete main key and metadata
        result = self._redis.delete(redis_key, meta_key)
        return result > 0

    def _remove_from_indices(self, key: str, ir: MemoryIR) -> None:
        """Remove key from indices."""
        pipe = self._redis.pipeline()

        for entity in ir.iter_entities():
            index_key = self._make_entity_index_key(entity.name)
            pipe.srem(index_key, key)

            for alias in entity.aliases:
                alias_key = self._make_entity_index_key(alias)
                pipe.srem(alias_key, key)

        for fact in ir.iter_facts():
            subj_key = self._make_fact_index_key(fact.subject)
            pipe.srem(subj_key, key)

            subj_pred_key = self._make_fact_index_key(fact.subject, fact.predicate)
            pipe.srem(subj_pred_key, key)

        pipe.execute()

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all memory keys.

        Args:
            prefix: Optional prefix filter.

        Returns:
            List of memory keys.
        """
        pattern = self._make_key(prefix + "*" if prefix else "*")
        keys = self._redis.keys(pattern)

        # Remove prefix to return clean keys
        base_prefix = self._make_key("")
        return [k.replace(base_prefix, "") if isinstance(k, str)
                else k.decode().replace(base_prefix, "") for k in keys]

    def exists(self, key: str) -> bool:
        """Check if memory exists.

        Args:
            key: Identifier to check.

        Returns:
            True if exists.
        """
        redis_key = self._make_key(key)
        return bool(self._redis.exists(redis_key))

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a memory.

        Args:
            key: Memory identifier.

        Returns:
            Metadata dictionary or None.
        """
        meta_key = self._make_metadata_key(key)
        data = self._redis.get(meta_key)

        if data is None:
            return None

        try:
            return json.loads(data) if isinstance(data, str) else json.loads(data.decode())
        except Exception:
            return None

    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a memory.

        Args:
            key: Memory identifier.
            metadata: New metadata to merge.

        Returns:
            True if updated.
        """
        if not self.exists(key):
            return False

        existing = self.get_metadata(key) or {}
        existing.update(metadata)

        meta_key = self._make_metadata_key(key)
        ttl = self._redis.ttl(self._make_key(key))

        if ttl and ttl > 0:
            self._redis.setex(meta_key, ttl, json.dumps(existing))
        else:
            self._redis.set(meta_key, json.dumps(existing))

        return True

    def search_by_entity(self, entity_name: str) -> List[str]:
        """Search for memories containing an entity.

        Args:
            entity_name: Entity name to search.

        Returns:
            List of memory keys.
        """
        index_key = self._make_entity_index_key(entity_name)
        members = self._redis.smembers(index_key)

        return [m if isinstance(m, str) else m.decode() for m in members]

    def search_by_fact(
        self,
        subject: str,
        predicate: Optional[str] = None,
    ) -> List[str]:
        """Search for memories by fact.

        Args:
            subject: Fact subject.
            predicate: Optional predicate.

        Returns:
            List of memory keys.
        """
        index_key = self._make_fact_index_key(subject, predicate)
        members = self._redis.smembers(index_key)

        return [m if isinstance(m, str) else m.decode() for m in members]

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a memory.

        Args:
            key: Memory identifier.

        Returns:
            TTL in seconds, -1 if no expiry, None if not found.
        """
        redis_key = self._make_key(key)
        ttl = self._redis.ttl(redis_key)

        if ttl == -2:  # Key doesn't exist
            return None
        return ttl

    def set_ttl(self, key: str, ttl: int) -> bool:
        """Set TTL for a memory.

        Args:
            key: Memory identifier.
            ttl: New TTL in seconds.

        Returns:
            True if set successfully.
        """
        redis_key = self._make_key(key)
        meta_key = self._make_metadata_key(key)

        pipe = self._redis.pipeline()
        pipe.expire(redis_key, ttl)
        pipe.expire(meta_key, ttl)

        results = pipe.execute()
        return all(results)

    def persist(self, key: str) -> bool:
        """Remove TTL from a memory (make permanent).

        Args:
            key: Memory identifier.

        Returns:
            True if persisted.
        """
        redis_key = self._make_key(key)
        meta_key = self._make_metadata_key(key)

        pipe = self._redis.pipeline()
        pipe.persist(redis_key)
        pipe.persist(meta_key)

        results = pipe.execute()
        return all(results)

    def close(self) -> None:
        """Close Redis connection."""
        if self._pool:
            self._pool.disconnect()
            logger.debug("Closed Redis connection")

    def clear_all(self, confirm: bool = False) -> int:
        """Clear all memories (use with caution).

        Args:
            confirm: Must be True to proceed.

        Returns:
            Number of keys deleted.
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear all memories")

        pattern = f"{self.prefix}*"
        keys = self._redis.keys(pattern)

        if keys:
            return self._redis.delete(*keys)
        return 0

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Statistics dictionary.
        """
        memory_keys = self.list_keys()
        total_size = 0

        for key in memory_keys:
            redis_key = self._make_key(key)
            size = self._redis.strlen(redis_key)
            total_size += size

        info = self._redis.info("memory")

        return {
            "memory_count": len(memory_keys),
            "total_size_bytes": total_size,
            "redis_used_memory": info.get("used_memory", 0),
            "redis_used_memory_human": info.get("used_memory_human", "N/A"),
            "redis_connected_clients": self._redis.info("clients").get("connected_clients", 0),
        }


class RedisCacheLayer:
    """Caching layer that wraps another MemoryStore with Redis caching.

    Provides read-through and write-through caching for any MemoryStore.

    Example:
        >>> backend = SQLiteMemoryStore("memories.db")
        >>> cached = RedisCacheLayer(backend, host="localhost")
        >>> cached.save("key", ir)  # Saves to both Redis and SQLite
        >>> ir = cached.load("key")  # Loads from Redis (fast)
    """

    def __init__(
        self,
        backend: MemoryStore,
        host: str = "localhost",
        port: int = 6379,
        cache_ttl: int = 3600,
        **redis_kwargs,
    ) -> None:
        """Initialize cache layer.

        Args:
            backend: Backend storage to cache.
            host: Redis host.
            port: Redis port.
            cache_ttl: Cache TTL in seconds.
            **redis_kwargs: Additional Redis arguments.
        """
        self.backend = backend
        self.cache = RedisMemoryStore(
            host=host,
            port=port,
            default_ttl=cache_ttl,
            **redis_kwargs,
        )
        self.cache_ttl = cache_ttl

    def save(
        self,
        key: str,
        ir: MemoryIR,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save to both cache and backend.

        Args:
            key: Memory identifier.
            ir: Memory IR to save.
            metadata: Optional metadata.
        """
        # Write to backend first (source of truth)
        self.backend.save(key, ir, metadata)

        # Then cache
        try:
            self.cache.save(key, ir, metadata, ttl=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def load(self, key: str) -> Optional[MemoryIR]:
        """Load from cache, falling back to backend.

        Args:
            key: Memory identifier.

        Returns:
            Memory IR or None.
        """
        # Try cache first
        try:
            ir = self.cache.load(key)
            if ir is not None:
                logger.debug(f"Cache hit: {key}")
                return ir
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

        # Fall back to backend
        logger.debug(f"Cache miss: {key}")
        ir = self.backend.load(key)

        # Populate cache if found
        if ir is not None:
            try:
                self.cache.save(key, ir, ttl=self.cache_ttl)
            except Exception as e:
                logger.warning(f"Cache population failed: {e}")

        return ir

    def delete(self, key: str) -> bool:
        """Delete from both cache and backend.

        Args:
            key: Memory identifier.

        Returns:
            True if deleted from backend.
        """
        # Delete from cache
        try:
            self.cache.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")

        # Delete from backend
        return self.backend.delete(key)

    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List keys from backend.

        Args:
            prefix: Optional prefix filter.

        Returns:
            List of keys.
        """
        return self.backend.list_keys(prefix)

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Memory identifier.

        Returns:
            True if exists.
        """
        # Check cache first
        try:
            if self.cache.exists(key):
                return True
        except Exception:
            pass

        return self.backend.exists(key)

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata from cache or backend."""
        try:
            meta = self.cache.get_metadata(key)
            if meta is not None:
                return meta
        except Exception:
            pass

        return self.backend.get_metadata(key)

    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry.

        Args:
            key: Memory identifier.

        Returns:
            True if invalidated.
        """
        try:
            return self.cache.delete(key)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return False

    def invalidate_all(self) -> int:
        """Invalidate all cached entries.

        Returns:
            Number of entries invalidated.
        """
        try:
            return self.cache.clear_all(confirm=True)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return 0

    def close(self) -> None:
        """Close both cache and backend connections."""
        self.cache.close()
        self.backend.close()

    def __enter__(self) -> "RedisCacheLayer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class RedisSessionStore:
    """Redis-based session storage for real-time dialogue.

    Optimized for high-frequency reads/writes during active sessions.

    Example:
        >>> session = RedisSessionStore("user_123", host="localhost")
        >>> session.append_turn("User: Hello!")
        >>> session.append_turn("Assistant: Hi there!")
        >>> turns = session.get_turns()
    """

    def __init__(
        self,
        session_id: str,
        host: str = "localhost",
        port: int = 6379,
        prefix: str = "memorycompiler:session:",
        ttl: int = 86400,  # 24 hours
        **redis_kwargs,
    ) -> None:
        """Initialize session store.

        Args:
            session_id: Unique session identifier.
            host: Redis host.
            port: Redis port.
            prefix: Key prefix.
            ttl: Session TTL in seconds.
            **redis_kwargs: Additional Redis arguments.
        """
        self.session_id = session_id
        self.prefix = prefix
        self.ttl = ttl

        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                **redis_kwargs,
            )
        except ImportError:
            raise ImportError("Redis support requires 'redis' package")

    def _turns_key(self) -> str:
        """Get key for turns list."""
        return f"{self.prefix}{self.session_id}:turns"

    def _metadata_key(self) -> str:
        """Get key for session metadata."""
        return f"{self.prefix}{self.session_id}:meta"

    def _ir_key(self) -> str:
        """Get key for compressed IR."""
        return f"{self.prefix}{self.session_id}:ir"

    def append_turn(self, turn: str) -> int:
        """Append a turn to the session.

        Args:
            turn: Dialogue turn text.

        Returns:
            Total turn count after append.
        """
        key = self._turns_key()
        pipe = self._redis.pipeline()
        pipe.rpush(key, turn)
        pipe.expire(key, self.ttl)
        results = pipe.execute()
        return results[0]

    def get_turns(
        self,
        start: int = 0,
        end: int = -1,
    ) -> List[str]:
        """Get turns from session.

        Args:
            start: Start index (0-based).
            end: End index (-1 for all).

        Returns:
            List of turns.
        """
        return self._redis.lrange(self._turns_key(), start, end)

    def get_turn_count(self) -> int:
        """Get total turn count."""
        return self._redis.llen(self._turns_key())

    def get_recent_turns(self, count: int = 10) -> List[str]:
        """Get most recent turns.

        Args:
            count: Number of turns to get.

        Returns:
            List of recent turns.
        """
        return self._redis.lrange(self._turns_key(), -count, -1)

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set session metadata.

        Args:
            metadata: Metadata dictionary.
        """
        key = self._metadata_key()
        self._redis.setex(key, self.ttl, json.dumps(metadata))

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get session metadata.

        Returns:
            Metadata dictionary or None.
        """
        data = self._redis.get(self._metadata_key())
        if data:
            return json.loads(data)
        return None

    def update_metadata(self, **kwargs) -> None:
        """Update session metadata.

        Args:
            **kwargs: Key-value pairs to update.
        """
        meta = self.get_metadata() or {}
        meta.update(kwargs)
        self.set_metadata(meta)

    def save_ir(self, ir: MemoryIR) -> None:
        """Save compressed IR for session.

        Args:
            ir: Memory IR to save.
        """
        key = self._ir_key()
        self._redis.setex(key, self.ttl, json.dumps(ir.to_dict()))

    def load_ir(self) -> Optional[MemoryIR]:
        """Load compressed IR for session.

        Returns:
            Memory IR or None.
        """
        data = self._redis.get(self._ir_key())
        if data:
            return MemoryIR.from_dict(json.loads(data))
        return None

    def clear(self) -> None:
        """Clear all session data."""
        self._redis.delete(
            self._turns_key(),
            self._metadata_key(),
            self._ir_key(),
        )

    def extend_ttl(self, ttl: Optional[int] = None) -> None:
        """Extend session TTL.

        Args:
            ttl: New TTL (uses default if None).
        """
        effective_ttl = ttl or self.ttl

        pipe = self._redis.pipeline()
        pipe.expire(self._turns_key(), effective_ttl)
        pipe.expire(self._metadata_key(), effective_ttl)
        pipe.expire(self._ir_key(), effective_ttl)
        pipe.execute()

    def is_active(self) -> bool:
        """Check if session is still active."""
        return bool(self._redis.exists(self._turns_key()))
