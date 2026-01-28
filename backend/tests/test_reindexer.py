# Unit tests for background reindexer.

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time

from multi_rag.api.reindexer import BackgroundReindexer, ReindexSettings
from multi_rag.indexing import Indexer, IndexerStores
from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import HashEmbeddingProvider
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore
from multi_rag.models import RawDocument
from multi_rag.pipeline.normalize import normalize_and_chunk


@dataclass(frozen=True)
class DummyPdfResult:
    text: str


def test_reindexer_updates_changed_files(tmp_path) -> None:
    file_path = tmp_path / "doc.md"
    file_path.write_text("Retention policy v1.")

    metadata_store = InMemoryMetadataStore()
    stores = IndexerStores(
        embedder=HashEmbeddingProvider(dim=8),
        vector_store=InMemoryVectorStore(),
        keyword_index=BM25Index(),
        metadata_store=metadata_store,
    )
    indexer = Indexer(stores)

    metadata = {
        "doc_id": "doc1",
        "created_at": "2000-01-01T00:00:00",
        "updated_at": "2000-01-01T00:00:00",
    }
    raw = RawDocument(
        source_type="markdown",
        title="Doc",
        origin=str(file_path),
        text="Retention policy v1.",
        metadata=metadata,
    )
    document, chunks = normalize_and_chunk(raw)
    indexer.index_document(document, chunks)

    file_path.write_text("Retention policy v2.")
    os.utime(file_path, (time.time(), time.time()))

    reindexer = BackgroundReindexer(
        metadata_store=metadata_store,
        indexer=indexer,
        settings=ReindexSettings(enabled=True, interval_seconds=1, max_documents=10),
        logger=logging.getLogger("test"),
        pdf_extractor=lambda *_args, **_kwargs: DummyPdfResult(text=""),
        ocr_enabled=False,
        ocr_min_word_count=10,
        ocr_lang="eng",
    )

    reindexed = reindexer.scan_once()
    updated_chunks = metadata_store.list_chunks("doc1")

    assert reindexed == 1
    assert any("v2" in chunk.chunk_text for chunk in updated_chunks)
