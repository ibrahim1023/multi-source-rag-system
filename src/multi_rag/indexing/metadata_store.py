# Metadata storage backends.

from __future__ import annotations

from dataclasses import dataclass, field

from multi_rag.models import Chunk, Document


class InMemoryMetadataStore:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}

    def upsert_document(self, document: Document) -> None:
        self._documents[document.doc_id] = document

    def upsert_chunk(self, chunk: Chunk) -> None:
        self._chunks[chunk.chunk_id] = chunk

    def get_document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def list_chunks(self, doc_id: str) -> list[Chunk]:
        chunks = [chunk for chunk in self._chunks.values() if chunk.doc_id == doc_id]
        return sorted(chunks, key=lambda item: item.chunk_index)

@dataclass
class PostgresConfig:
    dsn: str
    documents_table: str = "documents"
    chunks_table: str = "chunks"


class PostgresMetadataStore:
    def __init__(self, config: PostgresConfig) -> None:
        self._config = config

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "psycopg is required for PostgresMetadataStore.") from exc
        return psycopg.connect(self._config.dsn)

    def create_tables(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._config.documents_table} (
                    doc_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    owner TEXT,
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ,
                    tags TEXT[],
                    access_scope TEXT
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._config.chunks_table} (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    section_path TEXT,
                    start_offset INT,
                    end_offset INT,
                    metadata JSONB
                )
                """
            )
            conn.commit()

    def upsert_document(self, document: Document) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._config.documents_table}
                (doc_id, source_type, title, origin, owner, created_at, updated_at, tags, access_scope)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    source_type = EXCLUDED.source_type,
                    title = EXCLUDED.title,
                    origin = EXCLUDED.origin,
                    owner = EXCLUDED.owner,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    tags = EXCLUDED.tags,
                    access_scope = EXCLUDED.access_scope
                """,
                (
                    document.doc_id,
                    document.source_type,
                    document.title,
                    document.origin,
                    document.owner,
                    document.created_at,
                    document.updated_at,
                    document.tags,
                    document.access_scope,
                ),
            )
            conn.commit()

    def upsert_chunk(self, chunk: Chunk) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._config.chunks_table}
                (chunk_id, doc_id, chunk_text, chunk_index, section_path,
                 start_offset, end_offset, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    chunk_text = EXCLUDED.chunk_text,
                    chunk_index = EXCLUDED.chunk_index,
                    section_path = EXCLUDED.section_path,
                    start_offset = EXCLUDED.start_offset,
                    end_offset = EXCLUDED.end_offset,
                    metadata = EXCLUDED.metadata
                """,
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_text,
                    chunk.chunk_index,
                    chunk.section_path,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.metadata,
                ),
            )
            conn.commit()

    def list_chunks(self, doc_id: str) -> list[Chunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, doc_id, chunk_text, chunk_index, section_path,
                       start_offset, end_offset, metadata
                FROM {self._config.chunks_table}
                WHERE doc_id = %s
                ORDER BY chunk_index
                """,
                (doc_id,),
            )
            rows = cur.fetchall()
        return [
            Chunk(
                chunk_id=row[0],
                doc_id=row[1],
                chunk_text=row[2],
                chunk_index=row[3],
                section_path=row[4],
                start_offset=row[5],
                end_offset=row[6],
                metadata=row[7] or {},
            )
            for row in rows
        ]
