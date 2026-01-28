# Tests for PDF ingestion text extraction.

from __future__ import annotations

from io import BytesIO

import pytest

from multi_rag.api import app as api_app


def test_extract_pdf_text_blank_page_returns_empty() -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)

    result = api_app._extract_pdf_text(
        buffer.getvalue(),
        ocr_enabled=False,
        ocr_min_word_count=1,
        ocr_lang="eng",
    )

    assert result.text == ""
    assert result.used_ocr is False


def test_extract_pdf_text_uses_ocr_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)

    monkeypatch.setattr(api_app, "_run_pdf_ocr", lambda _content, *, lang: "OCR text")

    result = api_app._extract_pdf_text(
        buffer.getvalue(),
        ocr_enabled=True,
        ocr_min_word_count=1,
        ocr_lang="eng",
    )

    assert result.text == "OCR text"
    assert result.used_ocr is True
