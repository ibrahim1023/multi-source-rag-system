# Multi-Source RAG System (Docs Chat)

Production-style multi-source Retrieval-Augmented Generation (RAG) system for
chatting with internal company docs across PDFs, web pages, markdown notes, and
code documentation.

## What This Repo Is

- A scoped plan and build checklist for a demo-ready RAG product.
- A Python implementation in progress focused on the data pipeline.

## Implemented So Far

- Processing pipeline: normalization, cleaning, chunking, metadata merge.
- Indexing components: embeddings provider, vector store, BM25 keyword index,
  metadata store, and an indexer service for wiring them together.
- Unit tests for pipeline, BM25, vector store, and metadata store.

## Usage Example

```python
from multi_rag.indexing import Indexer, IndexerStores
from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import HashEmbeddingProvider
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore
from multi_rag.models import RawDocument
from multi_rag.pipeline.normalize import normalize_and_chunk

raw = RawDocument(
    source_type="markdown",
    title="sample",
    origin="/tmp/sample.md",
    text="# Title\nBody",
    metadata={"tags": ["demo"]},
)
document, chunks = normalize_and_chunk(raw)

stores = IndexerStores(
    embedder=HashEmbeddingProvider(dim=8),
    vector_store=InMemoryVectorStore(),
    keyword_index=BM25Index(),
    metadata_store=InMemoryMetadataStore(),
)
indexer = Indexer(stores)
indexer.index_document(document, chunks)
```

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in required variables for local services.

## Run Tests

```bash
python -m pytest
```

## Repo Map

- `scope.md`: project scope, architecture, milestones, and tooling.
- `tasks.md`: step-by-step checklist.
- `AGENTS.md`: contributor guidance and standards.
- `docs/`: architecture notes and decisions.
- `src/`: Python implementation.
- `tests/`: unit tests.
