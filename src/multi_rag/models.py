# Core data models for normalization and chunking.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    doc_id: str
    source_type: str
    title: str
    origin: str
    owner: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    access_scope: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_text: str
    chunk_index: int
    section_path: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RawDocument:
    source_type: str
    title: str
    origin: str
    text: str
    metadata: dict = field(default_factory=dict)
