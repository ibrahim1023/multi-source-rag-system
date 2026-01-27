# FastAPI application for ingestion, search, and chat.

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from multi_rag.answering.grounded import GroundedAnswerer
from multi_rag.answering.pipeline import AnsweringPipeline, AnsweringPipelineConfig
from multi_rag.api.ingest_log import InMemoryIngestLog
from multi_rag.api.models import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    IngestJobModel,
    IngestStatusResponse,
    DocumentListResponse,
    DocumentModel,
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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    context_max_chunks: int = 6
    neighbor_window: int = 1
    allowed_origins: list[str] = field(default_factory=list)


@dataclass
class AppDependencies:
    indexer: Indexer
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore
    retriever: HybridRetriever
    answerer: GroundedAnswerer
    pipeline: AnsweringPipeline
    tracer: Tracer
    ingest_log: InMemoryIngestLog


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
        return SentenceTransformerProvider(model_name=settings.embedding_model)
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
        ingest_log=InMemoryIngestLog(),
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
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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
        try:
            document, chunks = normalize_and_chunk(raw)
            dependencies.indexer.index_document(document, chunks)
            dependencies.tracer.record_event(
                "ingest.completed",
                {
                    "doc_id": document.doc_id,
                    "chunk_count": len(chunks),
                    "source_type": source_type,
                },
            )
            dependencies.ingest_log.record_success(
                source_type=source_type,
                title=payload.title,
                origin=payload.origin,
            )
            return IngestResponse(doc_id=document.doc_id, chunks_indexed=len(chunks))
        except Exception as exc:
            dependencies.ingest_log.record_failure(
                source_type=source_type,
                title=payload.title,
                origin=payload.origin,
                error=str(exc),
            )
            raise

    def ingest_file(
        source_type: str,
        *,
        title: str | None,
        origin: str | None,
        file: UploadFile,
        metadata_json: str | None,
    ) -> IngestResponse:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid metadata JSON.") from exc

        content_bytes = file.file.read()
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("utf-8", errors="ignore")
        text = text.replace("\x00", "")

        file_name = file.filename or "upload"
        resolved_title = title or file_name
        resolved_origin = origin or file_name

        raw = RawDocument(
            source_type=source_type,
            title=resolved_title,
            origin=resolved_origin,
            text=text,
            metadata=metadata,
        )
        try:
            document, chunks = normalize_and_chunk(raw)
            dependencies.indexer.index_document(document, chunks)
            dependencies.tracer.record_event(
                "ingest.completed",
                {
                    "doc_id": document.doc_id,
                    "chunk_count": len(chunks),
                    "source_type": source_type,
                    "file_name": file.filename,
                },
            )
            dependencies.ingest_log.record_success(
                source_type=source_type,
                title=resolved_title,
                origin=resolved_origin,
            )
            return IngestResponse(doc_id=document.doc_id, chunks_indexed=len(chunks))
        except Exception as exc:
            dependencies.ingest_log.record_failure(
                source_type=source_type,
                title=resolved_title,
                origin=resolved_origin,
                error=str(exc),
            )
            raise

    @app.post("/ingest/pdf", response_model=IngestResponse)
    def ingest_pdf(payload: IngestRequest) -> IngestResponse:
        return ingest("pdf", payload)

    @app.post("/ingest/pdf/file", response_model=IngestResponse)
    def ingest_pdf_file(
        title: str | None = Form(None),
        origin: str | None = Form(None),
        file: UploadFile = File(...),
        metadata: str | None = Form(None),
    ) -> IngestResponse:
        return ingest_file("pdf", title=title, origin=origin, file=file, metadata_json=metadata)

    @app.post("/ingest/web", response_model=IngestResponse)
    def ingest_web(payload: IngestRequest) -> IngestResponse:
        return ingest("web", payload)

    @app.post("/ingest/markdown", response_model=IngestResponse)
    def ingest_markdown(payload: IngestRequest) -> IngestResponse:
        return ingest("markdown", payload)

    @app.post("/ingest/code", response_model=IngestResponse)
    def ingest_code(payload: IngestRequest) -> IngestResponse:
        return ingest("code", payload)

    @app.get("/ingest/status", response_model=IngestStatusResponse)
    def ingest_status(limit: int = 50) -> IngestStatusResponse:
        jobs = dependencies.ingest_log.list_jobs(limit=limit)
        return IngestStatusResponse(
            jobs=[
                IngestJobModel(
                    job_id=job.job_id,
                    source_type=job.source_type,
                    title=job.title,
                    origin=job.origin,
                    status=job.status,
                    error=job.error,
                    created_at=job.created_at.isoformat(),
                )
                for job in jobs
            ]
        )

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents() -> DocumentListResponse:
        documents = dependencies.metadata_store.list_documents()
        return DocumentListResponse(
            documents=[
                DocumentModel(
                    doc_id=document.doc_id,
                    source_type=document.source_type,
                    title=document.title,
                    origin=document.origin,
                    owner=document.owner,
                    created_at=document.created_at.isoformat() if document.created_at else None,
                    updated_at=document.updated_at.isoformat() if document.updated_at else None,
                    tags=document.tags,
                    access_scope=document.access_scope,
                )
                for document in documents
            ]
        )

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
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
    return APISettings(
        metadata_backend=os.getenv("METADATA_BACKEND", "auto"),
        database_url=os.getenv("DATABASE_URL"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "8")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        top_k=int(os.getenv("TOP_K", "5")),
        context_max_chunks=int(os.getenv("CONTEXT_MAX_CHUNKS", "6")),
        neighbor_window=int(os.getenv("NEIGHBOR_WINDOW", "1")),
        allowed_origins=origins,
    )
