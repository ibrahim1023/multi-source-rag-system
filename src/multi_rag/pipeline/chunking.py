# Source-aware chunking with simple heading support.

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ChunkConfig:
    max_chars: int = 2000
    overlap_chars: int = 200


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_text(text: str, *, config: ChunkConfig, source_type: str) -> list[str]:
    if not text:
        return []

    if source_type == "markdown":
        chunks = _chunk_by_markdown_headings(text, config)
        if chunks:
            return chunks

    return _chunk_by_length(text, config)


def _chunk_by_markdown_headings(text: str, config: ChunkConfig) -> list[str]:
    lines = text.splitlines()
    sections: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if _MD_HEADING_RE.match(line) and buffer:
            sections.append("\n".join(buffer).strip())
            buffer = []
        buffer.append(line)
    if buffer:
        sections.append("\n".join(buffer).strip())

    chunks: list[str] = []
    for section in sections:
        if len(section) <= config.max_chars:
            chunks.append(section)
        else:
            chunks.extend(_chunk_by_length(section, config))
    return [chunk for chunk in chunks if chunk]


def _chunk_by_length(text: str, config: ChunkConfig) -> list[str]:
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + config.max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - config.overlap_chars, end)
    return chunks
