"""Storage backends for persistent Memory IR."""

from memory_compiler.storage.base import MemoryStore
from memory_compiler.storage.sqlite_store import SQLiteMemoryStore
from memory_compiler.storage.file_store import FileMemoryStore

# Redis is optional - import only if available
try:
    from memory_compiler.storage.redis_store import (
        RedisMemoryStore,
        RedisCacheLayer,
        RedisSessionStore,
    )
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    RedisMemoryStore = None
    RedisCacheLayer = None
    RedisSessionStore = None

__all__ = [
    "MemoryStore",
    "SQLiteMemoryStore",
    "FileMemoryStore",
    "RedisMemoryStore",
    "RedisCacheLayer",
    "RedisSessionStore",
]
