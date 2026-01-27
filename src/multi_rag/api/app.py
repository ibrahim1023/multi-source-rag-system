# FastAPI application for ingestion, search, and chat.

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os

from fastapi import FastAPI, HTTPException

from multi_rag.answering.grounded import GroundedAnswerer
from multi_rag.answering.pipeline import AnsweringPipeline, AnsweringPipelineConfig
from multi_rag.api.models import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from multi_rag.indexing import Indexer, IndexerStores
from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import HashEmbeddingProvider, SentenceTransformerProvider
from multi_rag.indexing.metadata_store import (
    InMemoryMetadataStore,
    PostgresConfig,
    PostgresMetadataStore,
)
from multi_rag.indexing.vector_store import InMemoryVectorStore
from multi_rag.models import RawDocument
from multi_rag.observability.tracing import NullTracer, Tracer
from multi_rag.pipeline.normalize import normalize_and_chunk
from multi_rag.retrieval.hybrid import HybridRetriever


@dataclass
class APISettings:
    metadata_backend: str = "auto"
    database_url: str | None = None
    embedding_provider: str = "hash"
    embedding_dim: int = 8
    top_k: int = 5
    context_max_chunks: int = 6
    neighbor_window: int = 1


@dataclass
class AppDependencies:
    indexer: Indexer
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore
    retriever: HybridRetriever
    answerer: GroundedAnswerer
    pipeline: AnsweringPipeline
    tracer: Tracer


def _select_metadata_store(settings: APISettings) -> InMemoryMetadataStore | PostgresMetadataStore:
    backend = settings.metadata_backend
    if backend == "auto":
        backend = "postgres" if settings.database_url else "memory"
    if backend == "postgres":
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is required for postgres metadata backend.")
        config = PostgresConfig(dsn=settings.database_url)
        return PostgresMetadataStore(config)
    return InMemoryMetadataStore()


def _select_embedder(settings: APISettings):
    if settings.embedding_provider == "sentence-transformer":
        return SentenceTransformerProvider()
    return HashEmbeddingProvider(dim=settings.embedding_dim)


def build_dependencies(settings: APISettings, *, tracer: Tracer | None = None) -> AppDependencies:
    active_tracer = tracer or NullTracer()
    metadata_store = _select_metadata_store(settings)
    embedder = _select_embedder(settings)
    vector_store = InMemoryVectorStore()
    keyword_index = BM25Index()
    stores = IndexerStores(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
    )
    indexer = Indexer(stores)
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
    )
    answerer = GroundedAnswerer(metadata_store=metadata_store)
    pipeline = AnsweringPipeline(
        retriever=retriever,
        answerer=answerer,
        config=AnsweringPipelineConfig(
            top_k=settings.top_k,
            context_max_chunks=settings.context_max_chunks,
            neighbor_window=settings.neighbor_window,
        ),
        tracer=active_tracer,
    )
    return AppDependencies(
        indexer=indexer,
        metadata_store=metadata_store,
        retriever=retriever,
        answerer=answerer,
        pipeline=pipeline,
        tracer=active_tracer,
    )


def create_app(settings: APISettings | None = None, dependencies: AppDependencies | None = None) -> FastAPI:
    settings = settings or _settings_from_env()
    dependencies = dependencies or build_dependencies(settings)
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if hasattr(dependencies.metadata_store, "create_tables"):
            dependencies.metadata_store.create_tables()
        yield

    app = FastAPI(title="Multi-RAG API", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    def ingest(source_type: str, payload: IngestRequest) -> IngestResponse:
        raw = RawDocument(
            source_type=source_type,
            title=payload.title,
            origin=payload.origin,
            text=payload.text,
            metadata=payload.metadata,
        )
        document, chunks = normalize_and_chunk(raw)
        dependencies.indexer.index_document(document, chunks)
        dependencies.tracer.record_event(
            "ingest.completed",
            {"doc_id": document.doc_id, "chunk_count": len(
                chunks), "source_type": source_type},
        )
        return IngestResponse(doc_id=document.doc_id, chunks_indexed=len(chunks))

    @app.post("/ingest/pdf", response_model=IngestResponse)
    def ingest_pdf(payload: IngestRequest) -> IngestResponse:
        return ingest("pdf", payload)

    @app.post("/ingest/web", response_model=IngestResponse)
    def ingest_web(payload: IngestRequest) -> IngestResponse:
        return ingest("web", payload)

    @app.post("/ingest/markdown", response_model=IngestResponse)
    def ingest_markdown(payload: IngestRequest) -> IngestResponse:
        return ingest("markdown", payload)

    @app.post("/ingest/code", response_model=IngestResponse)
    def ingest_code(payload: IngestRequest) -> IngestResponse:
        return ingest("code", payload)

    @app.post("/search", response_model=SearchResponse)
    def search(payload: SearchRequest) -> SearchResponse:
        results = dependencies.retriever.retrieve(
            payload.query,
            top_k=payload.top_k,
            metadata_filter=payload.metadata_filter,
        )
        hits: list[SearchResult] = []
        for result in results:
            chunk = dependencies.metadata_store.get_chunk(result.chunk_id)
            if not chunk:
                continue
            hits.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=result.score,
                    chunk_text=chunk.chunk_text,
                    metadata=chunk.metadata,
                    section_path=chunk.section_path,
                )
            )
        dependencies.tracer.record_event(
            "search.completed",
            {"query": payload.query, "result_count": len(hits)},
        )
        return SearchResponse(results=hits)

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        response = dependencies.pipeline.answer(
            payload.query,
            metadata_filter=payload.metadata_filter,
        )
        if not response.answer and not response.refused:
            raise HTTPException(status_code=500, detail="Answering failed.")
        return ChatResponse.from_response(response)

    return app


def _settings_from_env() -> APISettings:
    return APISettings(
        metadata_backend=os.getenv("METADATA_BACKEND", "auto"),
        database_url=os.getenv("DATABASE_URL"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "8")),
        top_k=int(os.getenv("TOP_K", "5")),
        context_max_chunks=int(os.getenv("CONTEXT_MAX_CHUNKS", "6")),
        neighbor_window=int(os.getenv("NEIGHBOR_WINDOW", "1")),
    )
