# Tests for PDF ingestion text extraction.

from __future__ import annotations

from io import BytesIO

import pytest

from multi_rag.api.app import _extract_pdf_text


def test_extract_pdf_text_blank_page_returns_empty() -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)

    text = _extract_pdf_text(buffer.getvalue())

    assert text == ""
