# Optional Postgres metadata store tests.

from __future__ import annotations

import os
import uuid

import pytest

from multi_rag.indexing.metadata_store import PostgresConfig, PostgresMetadataStore
from multi_rag.models import Chunk, Document


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL is required for Postgres metadata store tests.",
)
def test_postgres_metadata_store_round_trip() -> None:
    psycopg = pytest.importorskip("psycopg")
    suffix = uuid.uuid4().hex[:8]
    config = PostgresConfig(
        dsn=os.environ["DATABASE_URL"],
        documents_table=f"documents_test_{suffix}",
        chunks_table=f"chunks_test_{suffix}",
    )
    store = PostgresMetadataStore(config)
    store.create_tables()

    document = Document(
        doc_id="doc_pg_1",
        source_type="markdown",
        title="Doc",
        origin="/tmp/doc.md",
    )
    chunk = Chunk(
        chunk_id="doc_pg_1#0001",
        doc_id="doc_pg_1",
        chunk_text="Sample",
        chunk_index=1,
    )
    store.upsert_document(document)
    store.upsert_chunk(chunk)

    assert store.get_document("doc_pg_1") == document
    assert store.get_chunk("doc_pg_1#0001") == chunk

    with psycopg.connect(config.dsn) as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {config.chunks_table}")
        cur.execute(f"DROP TABLE IF EXISTS {config.documents_table}")
        conn.commit()
