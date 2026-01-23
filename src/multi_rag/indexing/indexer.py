# Indexer service that wires embeddings, vector, keyword, and metadata stores.

from __future__ import annotations

from dataclasses import dataclass

from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import EmbeddingProvider
from multi_rag.indexing.vector_store import InMemoryVectorStore, VectorRecord
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.models import Chunk, Document


@dataclass
class IndexerStores:
    embedder: EmbeddingProvider
    vector_store: InMemoryVectorStore
    keyword_index: BM25Index
    metadata_store: InMemoryMetadataStore


class Indexer:
    def __init__(self, stores: IndexerStores) -> None:
        self._stores = stores

    def index_document(self, document: Document, chunks: list[Chunk]) -> None:
        self._stores.metadata_store.upsert_document(document)
        for chunk in chunks:
            self._stores.metadata_store.upsert_chunk(chunk)

        vectors = self._stores.embedder.embed_texts([chunk.chunk_text for chunk in chunks])
        records = [
            VectorRecord(record_id=chunk.chunk_id, vector=vector, payload=chunk.metadata)
            for chunk, vector in zip(chunks, vectors)
        ]
        self._stores.vector_store.upsert(records)
        self._stores.keyword_index.add_documents(
            [chunk.chunk_id for chunk in chunks],
            [chunk.chunk_text for chunk in chunks],
        )
