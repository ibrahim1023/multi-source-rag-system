# Multi-Source RAG System (Docs Chat)

Production-style multi-source Retrieval-Augmented Generation (RAG) system for
chatting with internal company docs across PDFs, web pages, markdown notes, and
code documentation.

## What This Repo Is

- A scoped plan and build checklist for a demo-ready RAG product.
- A split backend/frontend implementation (FastAPI + Next.js).

## Implemented So Far

- Processing pipeline: normalization, cleaning, chunking, metadata merge.
- Indexing components: embeddings provider, vector store, BM25 keyword index,
  metadata store, and an indexer service for wiring them together.
- Answering: grounded answerer with citations, confidence, and refusal policy.
- API: FastAPI app with ingest, search, and chat endpoints.
- UI: Next.js chat console with filters, citations panel, and ingestion console.
- Unit tests for pipeline, indexing, retrieval, answering, and API.

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

Backend:

1. Copy `backend/.env.example` to `backend/.env`.
2. Fill in required variables for local services.
3. Install dependencies: `python -m pip install -e backend`

Dependency tracking:

- `backend/requirements.txt` mirrors `backend/pyproject.toml` runtime deps.
- Update both files whenever a new package is added.

Environment loading:

- The API auto-loads `backend/.env` on startup using `python-dotenv`.

If you want higher-quality embeddings, set:

```
EMBEDDING_PROVIDER=sentence-transformer
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

Frontend:

1. Copy `frontend/.env.local.example` to `frontend/.env.local`.
2. Install dependencies: `cd frontend && npm install`

## Run Tests

```bash
cd backend
python -m pytest
```

## Run API

```bash
cd backend
PYTHONPATH=src uvicorn multi_rag.api.app:create_app --factory
```

If you want Postgres-backed metadata, set `DATABASE_URL` (and optionally
`METADATA_BACKEND=postgres`) in your `backend/.env`.

When using Postgres, the API will rehydrate in-memory indexes from stored
chunks on startup so retrieval works after a restart.

Shorter option:

```bash
./backend/scripts/run_api.sh
```

## Run Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` for chat and `http://localhost:3000/ingest` for ingestion.

## Uploading Files

Use the file upload endpoint for PDFs:

- `POST /ingest/pdf/file`

It expects `multipart/form-data` with `title`, `origin`, optional `metadata`
JSON string, and `file`.

PDF upload extracts text with `pypdf`. Image-only PDFs will return a 422 error
because no selectable text can be extracted.

## Web Ingestion

The web ingest endpoint accepts either raw page text or a URL. If a URL is
provided, the backend fetches the page and extracts readable text before
indexing.

## Repo Map

- `backend/`: Python implementation.
- `frontend/`: Next.js chat console + ingestion console.
