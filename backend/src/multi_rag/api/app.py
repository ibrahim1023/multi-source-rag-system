# FastAPI application for ingestion, search, and chat.

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
import os

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from dotenv import load_dotenv
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
from multi_rag.pipeline.quality import analyze_text_quality, is_low_quality_ingest
from multi_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger("uvicorn.error")


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
    bm25_path: str | None = None
    ocr_enabled: bool = False
    ocr_min_word_count: int = 10
    ocr_lang: str = "eng"
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 8
    query_expansion_enabled: bool = True
    query_expansion_max_terms: int = 24


@dataclass
class AppDependencies:
    indexer: Indexer
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore
    retriever: HybridRetriever
    answerer: GroundedAnswerer
    pipeline: AnsweringPipeline
    tracer: Tracer
    ingest_log: InMemoryIngestLog
    bm25_loaded: bool = False
    bm25_path: str | None = None
    ocr_enabled: bool = False
    ocr_min_word_count: int = 10
    ocr_lang: str = "eng"


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
    keyword_index = BM25Index(persist_path=settings.bm25_path)
    bm25_loaded = False
    if settings.bm25_path and os.path.exists(settings.bm25_path):
        keyword_index.load(settings.bm25_path)
        bm25_loaded = True
    stores = IndexerStores(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
    )
    indexer = Indexer(stores)
    reranker = None
    if settings.reranker_enabled:
        from multi_rag.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    from multi_rag.retrieval.query_expansion import QueryExpansionConfig

    query_config = QueryExpansionConfig(
        enabled=settings.query_expansion_enabled,
        max_total_terms=settings.query_expansion_max_terms,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
        query_config=query_config,
        reranker=reranker,
        rerank_top_k=settings.reranker_top_k,
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
        bm25_loaded=bm25_loaded,
        bm25_path=settings.bm25_path,
        ocr_enabled=settings.ocr_enabled,
        ocr_min_word_count=settings.ocr_min_word_count,
        ocr_lang=settings.ocr_lang,
    )


def _rehydrate_indexes(dependencies: AppDependencies) -> None:
    documents = dependencies.metadata_store.list_documents()
    if not documents:
        return
    total = len(documents)
    indexed = 0
    logger.info("Rehydrating %s document(s) into in-memory indexes...", total)
    for document in documents:
        chunks = dependencies.metadata_store.list_chunks(document.doc_id)
        if not chunks:
            continue
        dependencies.indexer.index_document(document, chunks, include_bm25=not dependencies.bm25_loaded)
        indexed += 1
        if indexed % 25 == 0 or indexed == total:
            logger.info("Rehydration progress: %s/%s documents indexed.", indexed, total)
    if indexed:
        logger.info("Rehydrated %s document(s) into in-memory indexes.", indexed)


@dataclass(frozen=True)
class PDFExtractionResult:
    text: str
    used_ocr: bool


def _run_pdf_ocr(content_bytes: bytes, *, lang: str) -> str:
    try:
        from pdf2image import convert_from_bytes
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("pdf2image is required for OCR fallback.") from exc
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("pytesseract is required for OCR fallback.") from exc

    images = convert_from_bytes(content_bytes)
    text = "\n".join(pytesseract.image_to_string(image, lang=lang) for image in images)
    return text.replace("\x00", "")


def _extract_pdf_text(
    content_bytes: bytes,
    *,
    ocr_enabled: bool,
    ocr_min_word_count: int,
    ocr_lang: str,
) -> PDFExtractionResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - runtime only
        raise RuntimeError("pypdf is required for PDF ingestion.") from exc

    from io import BytesIO

    reader = PdfReader(BytesIO(content_bytes))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = text.replace("\x00", "")
    quality = analyze_text_quality(text)
    if quality.word_count >= ocr_min_word_count:
        return PDFExtractionResult(text=text, used_ocr=False)
    if not ocr_enabled:
        return PDFExtractionResult(text=text, used_ocr=False)
    try:
        ocr_text = _run_pdf_ocr(content_bytes, lang=ocr_lang)
    except RuntimeError as exc:
        logger.warning("OCR fallback unavailable: %s", exc)
        return PDFExtractionResult(text=text, used_ocr=False)
    if ocr_text.strip():
        return PDFExtractionResult(text=ocr_text, used_ocr=True)
    return PDFExtractionResult(text=text, used_ocr=False)


def _apply_ocr_metadata(metadata: dict, *, used_ocr: bool) -> dict:
    if not used_ocr:
        return metadata
    metadata.setdefault("extraction_method", "ocr")
    metadata["ocr_used"] = True
    tags = metadata.get("tags")
    if tags is None:
        metadata["tags"] = ["ocr"]
    elif isinstance(tags, list):
        if "ocr" not in tags:
            tags.append("ocr")
    return metadata


def _validate_ingest_text(text: str, *, source_type: str) -> None:
    if is_low_quality_ingest(text, source_type=source_type):
        raise HTTPException(
            status_code=422,
            detail="Extracted text is empty or too low-quality to ingest.",
        )


def create_app(settings: APISettings | None = None, dependencies: AppDependencies | None = None) -> FastAPI:
    settings = settings or _settings_from_env()
    dependencies = dependencies or build_dependencies(settings)
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if hasattr(dependencies.metadata_store, "create_tables"):
            dependencies.metadata_store.create_tables()
        _rehydrate_indexes(dependencies)
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
            _validate_ingest_text(payload.text, source_type=source_type)
            document, chunks = normalize_and_chunk(raw)
            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail="No usable chunks were produced after cleaning.",
                )
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
        if source_type == "pdf":
            extracted = _extract_pdf_text(
                content_bytes,
                ocr_enabled=dependencies.ocr_enabled,
                ocr_min_word_count=dependencies.ocr_min_word_count,
                ocr_lang=dependencies.ocr_lang,
            )
            text = extracted.text
            metadata = _apply_ocr_metadata(metadata, used_ocr=extracted.used_ocr)
            if not text.strip():
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No text could be extracted from the PDF. "
                        "Ensure it contains selectable text or enable OCR."
                    ),
                )
        else:
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
            _validate_ingest_text(text, source_type=source_type)
            document, chunks = normalize_and_chunk(raw)
            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail="No usable chunks were produced after cleaning.",
                )
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
    load_dotenv()
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
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
        bm25_path=os.getenv("BM25_PATH"),
        ocr_enabled=_env_flag("OCR_ENABLED", False),
        ocr_min_word_count=int(os.getenv("OCR_MIN_WORD_COUNT", "10")),
        ocr_lang=os.getenv("OCR_LANG", "eng"),
        reranker_enabled=_env_flag("RERANKER_ENABLED", False),
        reranker_model=os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        reranker_top_k=int(os.getenv("RERANKER_TOP_K", "8")),
        query_expansion_enabled=_env_flag("QUERY_EXPANSION_ENABLED", True),
        query_expansion_max_terms=int(os.getenv("QUERY_EXPANSION_MAX_TERMS", "24")),
    )
