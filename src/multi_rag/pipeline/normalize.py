# Normalize raw inputs into shared Document and Chunk records.

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from multi_rag.models import Chunk, Document, RawDocument
from multi_rag.pipeline.chunking import ChunkConfig, chunk_text
from multi_rag.pipeline.cleaning import strip_boilerplate
from multi_rag.pipeline.metadata import merge_metadata


def normalize_document(raw: RawDocument) -> Document:
    metadata = raw.metadata
    return Document(
        doc_id=metadata.get("doc_id", raw.title),
        source_type=raw.source_type,
        title=raw.title,
        origin=raw.origin,
        owner=metadata.get("owner"),
        created_at=_parse_dt(metadata.get("created_at")),
        updated_at=_parse_dt(metadata.get("updated_at")),
        tags=metadata.get("tags", []),
        access_scope=metadata.get("access_scope"),
    )


def build_chunks(
    document: Document,
    raw_text: str,
    *,
    base_metadata: dict | None = None,
    chunk_config: ChunkConfig | None = None,
) -> list[Chunk]:
    cleaned = strip_boilerplate(raw_text)
    config = chunk_config or ChunkConfig()
    chunks_text = chunk_text(cleaned, config=config, source_type=document.source_type)
    base = base_metadata or {}
    chunks: list[Chunk] = []
    for idx, text in enumerate(chunks_text):
        chunk_id = f"{document.doc_id}#{idx:04d}"
        metadata = merge_metadata(
            base,
            {"source_type": document.source_type, "origin": document.origin},
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                chunk_text=text,
                chunk_index=idx,
                metadata=metadata,
            )
        )
    return chunks


def normalize_and_chunk(
    raw: RawDocument,
    *,
    chunk_config: ChunkConfig | None = None,
) -> tuple[Document, list[Chunk]]:
    document = normalize_document(raw)
    chunks = build_chunks(
        document,
        raw.text,
        base_metadata=raw.metadata,
        chunk_config=chunk_config,
    )
    return document, chunks


def normalize_batch(
    raws: Iterable[RawDocument],
    *,
    chunk_config: ChunkConfig | None = None,
) -> list[tuple[Document, list[Chunk]]]:
    return [normalize_and_chunk(raw, chunk_config=chunk_config) for raw in raws]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)
