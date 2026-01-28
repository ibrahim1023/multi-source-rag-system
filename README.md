# Multi-Source RAG System (Docs Chat)

Production-style Retrieval-Augmented Generation (RAG) system for chatting with
internal company docs across PDFs, Markdown notes, and code documentation.

## Overview

- Purpose: demo-ready RAG system with grounded answers and citations.
- Stack: FastAPI backend + Next.js frontend.
- Scope: file-based ingestion only (PDF, Markdown, code docs).

## Highlights

- Hybrid retrieval (vector + BM25) with reranking and query expansion.
- Grounded answering with per-claim citations and confidence policy.
- Ingestion quality checks, OCR fallback, and background reindexing.
- Offline eval harness and structured observability logs.

## Architecture (at a glance)

- Ingest: PDF, Markdown, code docs -> normalize -> chunk -> metadata.
- Index: embeddings + vector store + BM25 + metadata store.
- Retrieve: hybrid search + rerank + context assembly.
- Answer: grounded claims + citations + refusal behavior.
- Serve: ingest, search, chat APIs + web UI.

## Quickstart

Backend:

1. Copy `backend/.env.example` to `backend/.env`.
2. Fill in required variables.
3. Install dependencies:

```bash
python -m pip install -e backend
```

Run API:

```bash
cd backend
PYTHONPATH=src uvicorn multi_rag.api.app:create_app --factory
```

Frontend:

1. Copy `frontend/.env.local.example` to `frontend/.env.local`.
2. Install dependencies:

```bash
cd frontend
npm install
```

Run UI:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` for chat and `http://localhost:3000/ingest` for ingestion.

## Configuration Notes

Embedding quality:

```
EMBEDDING_PROVIDER=sentence-transformer
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

Optional Gemini answering:

- Set `LLM_MODEL=gemini-flash-latest` and `GOOGLE_API_KEY` in `backend/.env`.
- Uses `google-genai` (already listed in backend dependencies).
- Synthesizes answers from grounded claims while preserving citations.

BM25 persistence:

- Set `BM25_PATH` to a writable file path.
- The API loads the BM25 index from this path on startup if it exists.

Postgres metadata:

- Set `DATABASE_URL` (and optionally `METADATA_BACKEND=postgres`).
- On startup, the API rehydrates in-memory indexes from stored chunks.

OCR fallback (PDF):

- Set `OCR_ENABLED=true`.
- Requires `pytesseract`, `pdf2image`, and system Tesseract + Poppler.
- OCR ingests are tagged with `ocr` and `extraction_method=ocr`.
- Set `NEXT_PUBLIC_OCR_ENABLED=true` in the frontend env to show a UI badge.

Observability:

- Set `OBSERVABILITY_MODE=structured` for JSON logs.
- Use `OBSERVABILITY_SERVICE_NAME` and `OBSERVABILITY_STATIC_FIELDS` for tags.

## Tests

```bash
cd backend
python -m pytest
```

## Offline Evaluation

```bash
python backend/scripts/run_eval.py --gold eval/gold_set.jsonl --config eval/config.json
```

Populate `eval/gold_set.jsonl` with 30-80 questions and expected sources.

## Demo Sources

```bash
./backend/scripts/ingest_demo.sh
```

This posts files in `demo_sources/` to the running API for quick eval runs.

## Repo Map

- `backend/`: FastAPI implementation.
- `frontend/`: Next.js chat console + ingestion console.
- `eval/`: offline eval harness and gold set.
- `demo_sources/`: sample files for ingestion.
