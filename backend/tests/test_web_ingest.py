# Unit tests for web ingestion helpers.

from __future__ import annotations

from multi_rag.api.app import _extract_html_text, _looks_like_url


def test_extract_html_text_strips_scripts() -> None:
    html = """
    <html>
      <head><title>LangGraph</title><style>.x { color: red; }</style></head>
      <body>
        <script>console.log('ignore');</script>
        <h1>LangGraph Persistence</h1>
        <p>Checkpoints store state across runs.</p>
      </body>
    </html>
    """
    text = _extract_html_text(html)

    assert "LangGraph Persistence" in text
    assert "Checkpoints store state" in text
    assert "console.log" not in text


def test_looks_like_url_detection() -> None:
    assert _looks_like_url("https://docs.langchain.com/oss/python/langgraph/overview")
    assert _looks_like_url("http://example.com")
    assert _looks_like_url("https://example.com/path")
    assert _looks_like_url("https://example.com?x=1")
    assert _looks_like_url("https://example.com#frag")
    assert not _looks_like_url("example.com")
    assert not _looks_like_url("not a url")
