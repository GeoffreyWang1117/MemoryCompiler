"""FastAPI application for MemoryCompiler service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from memory_compiler import __version__
from memory_compiler.api.models import (
    BatchCompressRequest,
    BatchCompressResponse,
    CompressRequest,
    CompressResponse,
    EntityInfo,
    ErrorResponse,
    FactInfo,
    HealthResponse,
    MemoryCreateRequest,
    MemoryLoadResponse,
    MemoryResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    StoreStatsResponse,
)
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.pipeline import MemoryCompiler
from memory_compiler.retrieval import HybridRetriever, MemoryRetriever
from memory_compiler.storage import FileMemoryStore, MemoryStore, SQLiteMemoryStore


# Global instances
_compiler: Optional[MemoryCompiler] = None
_store: Optional[MemoryStore] = None
_retriever: Optional[HybridRetriever] = None


def get_compiler() -> MemoryCompiler:
    """Get or create the compiler instance."""
    global _compiler
    if _compiler is None:
        _compiler = MemoryCompiler(use_mock=True)
    return _compiler


def get_store() -> MemoryStore:
    """Get or create the storage instance."""
    global _store
    if _store is None:
        _store = SQLiteMemoryStore("./memory_api_store.db")
    return _store


def get_retriever() -> HybridRetriever:
    """Get or create the retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting MemoryCompiler API service")

    # Initialize components
    get_compiler()
    get_store()
    get_retriever()

    # Index existing memories for retrieval
    store = get_store()
    retriever = get_retriever()

    for key in store.list_keys():
        ir = store.load(key)
        if ir:
            retriever.index_memory(key, ir)

    logger.info(f"Indexed {len(store.list_keys())} existing memories")

    yield

    # Cleanup
    logger.info("Shutting down MemoryCompiler API service")
    store.close()


def create_app(
    title: str = "MemoryCompiler API",
    storage_backend: str = "sqlite",
    storage_path: str = "./memory_api_store.db",
    use_mock_llm: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title.
        storage_backend: Storage backend type ('sqlite' or 'file').
        storage_path: Path for storage.
        use_mock_llm: Whether to use mock LLM extraction.

    Returns:
        Configured FastAPI application.
    """
    global _store, _compiler

    # Initialize storage
    if storage_backend == "sqlite":
        _store = SQLiteMemoryStore(storage_path)
    else:
        _store = FileMemoryStore(storage_path)

    # Initialize compiler
    _compiler = MemoryCompiler(use_mock=use_mock_llm)

    app = FastAPI(
        title=title,
        description="Compiler-inspired dialogue memory compression service",
        version=__version__,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check service health."""
        store = get_store()
        retriever = get_retriever()

        return HealthResponse(
            status="healthy",
            version=__version__,
            storage_available=store is not None,
            retriever_ready=retriever is not None,
        )

    @app.post(
        "/compress",
        response_model=CompressResponse,
        responses={400: {"model": ErrorResponse}},
        tags=["Compression"],
    )
    async def compress_dialogue(request: CompressRequest):
        """Compress a dialogue conversation.

        Takes a list of dialogue messages and returns compressed output
        with extracted entities and facts.
        """
        try:
            compiler = get_compiler()

            # Convert messages to dialogue format
            dialogue = "\n".join(
                f"{msg.role.value.capitalize()}: {msg.content}"
                for msg in request.messages
            )

            # Compress
            ir = await asyncio.to_thread(
                compiler.compile,
                dialogue,
                token_budget=request.token_budget,
            )

            # Generate compressed text
            compressed_text = await asyncio.to_thread(
                compiler.generate,
                ir,
                style=request.output_format,
            )

            # Estimate tokens
            original_tokens = len(dialogue.split()) * 1.3
            compressed_tokens = len(compressed_text.split()) * 1.3

            # Extract entity/fact info
            entities = [
                EntityInfo(
                    id=e.id,
                    name=e.name,
                    type=e.type.value,
                    importance=e.importance_score,
                    aliases=list(e.aliases),
                    attributes=e.attributes,
                )
                for e in ir.iter_entities()
            ]

            facts = [
                FactInfo(
                    id=f.id,
                    subject=f.subject,
                    predicate=f.predicate,
                    object=f.object,
                    confidence=f.confidence,
                    importance=f.importance_score,
                )
                for f in ir.iter_facts()
            ]

            return CompressResponse(
                compressed_text=compressed_text,
                original_tokens=int(original_tokens),
                compressed_tokens=int(compressed_tokens),
                compression_ratio=original_tokens / max(compressed_tokens, 1),
                entities=entities,
                facts=facts,
                ir_json=ir.to_dict() if request.session_id else None,
            )

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post(
        "/compress/batch",
        response_model=BatchCompressResponse,
        tags=["Compression"],
    )
    async def batch_compress(request: BatchCompressRequest):
        """Compress multiple dialogues in batch."""
        results = []
        failed = 0

        async def process_one(req: CompressRequest) -> Optional[CompressResponse]:
            try:
                return await compress_dialogue(req)
            except Exception:
                return None

        if request.parallel:
            tasks = [process_one(session) for session in request.sessions]
            responses = await asyncio.gather(*tasks)

            for resp in responses:
                if resp:
                    results.append(resp)
                else:
                    failed += 1
        else:
            for session in request.sessions:
                resp = await process_one(session)
                if resp:
                    results.append(resp)
                else:
                    failed += 1

        return BatchCompressResponse(
            results=results,
            total_processed=len(results),
            failed=failed,
        )

    @app.post(
        "/memories",
        response_model=MemoryResponse,
        tags=["Storage"],
    )
    async def create_memory(
        request: MemoryCreateRequest,
        background_tasks: BackgroundTasks,
    ):
        """Create or update a stored memory."""
        try:
            compiler = get_compiler()
            store = get_store()
            retriever = get_retriever()

            # Convert messages to dialogue
            dialogue = "\n".join(
                f"{msg.role.value.capitalize()}: {msg.content}"
                for msg in request.messages
            )

            # Optionally compress
            if request.compress:
                ir = await asyncio.to_thread(
                    compiler.compile,
                    dialogue,
                    token_budget=request.token_budget,
                )
            else:
                ir = await asyncio.to_thread(compiler.extract, dialogue)

            # Save to store
            metadata = request.metadata or {}
            metadata["created_at"] = datetime.now().isoformat()

            await asyncio.to_thread(store.save, request.key, ir, metadata)

            # Index for retrieval in background
            background_tasks.add_task(retriever.index_memory, request.key, ir)

            return MemoryResponse(
                key=request.key,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                entity_count=len(list(ir.iter_entities())),
                fact_count=len(list(ir.iter_facts())),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.get(
        "/memories/{key}",
        response_model=MemoryLoadResponse,
        responses={404: {"model": ErrorResponse}},
        tags=["Storage"],
    )
    async def get_memory(key: str, format: str = Query(default="narrative")):
        """Retrieve a stored memory by key."""
        store = get_store()
        compiler = get_compiler()

        ir = await asyncio.to_thread(store.load, key)
        if ir is None:
            raise HTTPException(status_code=404, detail=f"Memory not found: {key}")

        metadata = await asyncio.to_thread(store.get_metadata, key)

        # Generate compressed text
        compressed_text = await asyncio.to_thread(
            compiler.generate,
            ir,
            style=format,
        )

        return MemoryLoadResponse(
            key=key,
            ir=ir.to_dict(),
            metadata=metadata,
            compressed_text=compressed_text,
        )

    @app.delete(
        "/memories/{key}",
        responses={404: {"model": ErrorResponse}},
        tags=["Storage"],
    )
    async def delete_memory(key: str):
        """Delete a stored memory."""
        store = get_store()
        retriever = get_retriever()

        deleted = await asyncio.to_thread(store.delete, key)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory not found: {key}")

        # Remove from retriever
        retriever.remove_memory(key)

        return {"deleted": key}

    @app.get(
        "/memories",
        response_model=List[str],
        tags=["Storage"],
    )
    async def list_memories(prefix: Optional[str] = Query(default=None)):
        """List all stored memory keys."""
        store = get_store()
        keys = await asyncio.to_thread(store.list_keys, prefix)
        return keys

    @app.get("/memories/stats", response_model=StoreStatsResponse, tags=["Storage"])
    async def get_storage_stats():
        """Get storage statistics."""
        store = get_store()
        retriever = get_retriever()

        stats = await asyncio.to_thread(store.get_stats)

        return StoreStatsResponse(
            num_memories=stats.get("num_memories", 0),
            total_entities=stats.get("total_entities", 0),
            total_facts=stats.get("total_facts", 0),
            total_chunks=retriever.semantic_retriever.vector_store.count,
            storage_type=type(store).__name__,
        )

    @app.post(
        "/retrieve",
        response_model=RetrievalResponse,
        tags=["Retrieval"],
    )
    async def retrieve_memories(request: RetrievalRequest):
        """Retrieve relevant memories for a query."""
        retriever = get_retriever()

        results = await asyncio.to_thread(
            retriever.retrieve,
            request.query,
            top_k=request.top_k,
            filter_keys=request.filter_keys,
        )

        # Format context if requested
        context = None
        if request.include_context and results:
            context = await asyncio.to_thread(
                retriever.format_context,
                results,
                max_tokens=request.max_context_tokens,
            )

        return RetrievalResponse(
            results=[
                RetrievalResult(
                    text=r.text,
                    score=r.score,
                    source_key=r.source_key,
                    chunk_id=r.chunk_id,
                    chunk_type=r.metadata.get("type"),
                )
                for r in results
            ],
            context=context,
            query=request.query,
        )

    @app.get("/search/entity", response_model=List[str], tags=["Search"])
    async def search_by_entity(name: str = Query(..., min_length=1)):
        """Search memories containing an entity."""
        store = get_store()
        keys = await asyncio.to_thread(store.search_by_entity, name)
        return keys

    @app.get("/search/fact", response_model=List[str], tags=["Search"])
    async def search_by_fact(
        subject: str = Query(..., min_length=1),
        predicate: Optional[str] = Query(default=None),
    ):
        """Search memories containing a fact."""
        store = get_store()
        keys = await asyncio.to_thread(store.search_by_fact, subject, predicate)
        return keys


# Create default app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    uvicorn.run(
        "memory_compiler.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
