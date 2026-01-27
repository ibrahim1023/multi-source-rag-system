# Unit tests for the processing pipeline (Task 4).

from __future__ import annotations

from multi_rag.models import RawDocument
from multi_rag.pipeline.chunking import ChunkConfig, chunk_text
from multi_rag.pipeline.cleaning import strip_boilerplate
from multi_rag.pipeline.normalize import normalize_and_chunk


def test_strip_boilerplate_collapses_whitespace() -> None:
    text = "Hello   world\n\nThis  is\tfine."
    cleaned = strip_boilerplate(text)
    assert cleaned == "Hello world This is fine."


def test_chunk_text_markdown_heading_split() -> None:
    text = "# Title\nAlpha\n## Section\nBeta\n"
    chunks = chunk_text(text, config=ChunkConfig(max_chars=100, overlap_chars=0), source_type="markdown")
    assert len(chunks) == 2
    assert chunks[0].startswith("# Title")
    assert chunks[1].startswith("## Section")


def test_normalize_and_chunk_builds_document_and_chunks() -> None:
    raw = RawDocument(
        source_type="markdown",
        title="sample",
        origin="/tmp/sample.md",
        text="# Title\nBody",
        metadata={"tags": ["demo"], "owner": "team"},
    )
    document, chunks = normalize_and_chunk(raw, chunk_config=ChunkConfig(max_chars=100, overlap_chars=0))
    assert document.doc_id == "sample"
    assert document.source_type == "markdown"
    assert document.tags == ["demo"]
    assert len(chunks) == 1
    assert chunks[0].doc_id == document.doc_id
    assert chunks[0].metadata["source_type"] == "markdown"
