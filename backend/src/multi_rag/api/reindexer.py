# Background reindex job for changed sources.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import threading
import time

from multi_rag.indexing import Indexer
from multi_rag.indexing.metadata_store import InMemoryMetadataStore, PostgresMetadataStore
from multi_rag.models import RawDocument
from multi_rag.pipeline.normalize import normalize_and_chunk
from multi_rag.pipeline.quality import is_low_quality_ingest


@dataclass(frozen=True)
class ReindexSettings:
    enabled: bool = False
    interval_seconds: int = 300
    max_documents: int = 50


class BackgroundReindexer:
    def __init__(
        self,
        *,
        metadata_store: InMemoryMetadataStore | PostgresMetadataStore,
        indexer: Indexer,
        settings: ReindexSettings,
        logger: logging.Logger,
        pdf_extractor,
        ocr_enabled: bool,
        ocr_min_word_count: int,
        ocr_lang: str,
    ) -> None:
        self._metadata_store = metadata_store
        self._indexer = indexer
        self._settings = settings
        self._logger = logger
        self._pdf_extractor = pdf_extractor
        self._ocr_enabled = ocr_enabled
        self._ocr_min_word_count = ocr_min_word_count
        self._ocr_lang = ocr_lang
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._settings.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._logger.info("Reindexer started (interval=%ss).", self._settings.interval_seconds)

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None
        self._logger.info("Reindexer stopped.")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception as exc:  # pragma: no cover - safety
                self._logger.warning("Reindexer scan failed: %s", exc)
            self._stop.wait(self._settings.interval_seconds)

    def scan_once(self) -> int:
        if not self._settings.enabled:
            return 0
        reindexed = 0
        documents = self._metadata_store.list_documents()
        for document in documents[: self._settings.max_documents]:
            origin = document.origin
            if not origin or not os.path.exists(origin):
                continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(origin), tz=timezone.utc)
            except OSError:
                continue
            if not _is_newer_than(document.updated_at, mtime):
                continue
            if not _is_newer_than(document.created_at, mtime) and document.updated_at is None:
                continue
            if not self._reindex_document(document.doc_id, origin, mtime):
                continue
            reindexed += 1
        if reindexed:
            self._logger.info("Reindexed %s document(s) from disk.", reindexed)
        return reindexed

    def _reindex_document(self, doc_id: str, origin: str, mtime: datetime) -> bool:
        document = self._metadata_store.get_document(doc_id)
        if not document:
            return False
        chunks = self._metadata_store.list_chunks(doc_id)
        base_metadata = dict(chunks[0].metadata) if chunks else {}
        metadata = dict(base_metadata)
        metadata["doc_id"] = doc_id
        metadata["owner"] = document.owner
        metadata["tags"] = document.tags
        metadata["access_scope"] = document.access_scope
        if document.created_at:
            metadata["created_at"] = document.created_at.isoformat()
        metadata["updated_at"] = mtime.isoformat()

        text = self._read_origin_text(document.source_type, origin)
        if not text or is_low_quality_ingest(text, source_type=document.source_type):
            self._logger.warning("Reindex skipped for %s due to low-quality text.", origin)
            return False

        raw = RawDocument(
            source_type=document.source_type,
            title=document.title,
            origin=origin,
            text=text,
            metadata=metadata,
        )
        updated_document, updated_chunks = normalize_and_chunk(raw)
        if not updated_chunks:
            self._logger.warning("Reindex produced no chunks for %s.", origin)
            return False
        self._indexer.remove_document(doc_id)
        self._indexer.index_document(updated_document, updated_chunks)
        return True

    def _read_origin_text(self, source_type: str, origin: str) -> str:
        content = _read_bytes(origin)
        if source_type == "pdf":
            result = self._pdf_extractor(
                content,
                ocr_enabled=self._ocr_enabled,
                ocr_min_word_count=self._ocr_min_word_count,
                ocr_lang=self._ocr_lang,
            )
            return result.text
        return _decode_text(content)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="ignore")


def _is_newer_than(previous: datetime | None, current: datetime) -> bool:
    if not previous:
        return True
    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=timezone.utc)
    return current > previous
