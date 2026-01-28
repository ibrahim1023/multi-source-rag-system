# Quality checks for ingested text and chunks.

from __future__ import annotations

from dataclasses import dataclass
import re


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class TextQuality:
    stripped_length: int
    word_count: int
    alpha_ratio: float


def analyze_text_quality(text: str) -> TextQuality:
    stripped = text.strip()
    stripped_length = len(stripped)
    if not stripped_length:
        return TextQuality(stripped_length=0, word_count=0, alpha_ratio=0.0)
    word_count = len(_WORD_RE.findall(stripped))
    alnum_count = sum(ch.isalnum() for ch in stripped)
    alpha_ratio = alnum_count / stripped_length if stripped_length else 0.0
    return TextQuality(
        stripped_length=stripped_length,
        word_count=word_count,
        alpha_ratio=alpha_ratio,
    )


def is_low_quality_chunk(text: str) -> bool:
    quality = analyze_text_quality(text)
    if quality.stripped_length == 0:
        return True
    if quality.word_count == 0:
        return True
    if quality.alpha_ratio < 0.05:
        return True
    return False


def is_low_quality_ingest(text: str, *, source_type: str) -> bool:
    quality = analyze_text_quality(text)
    if quality.stripped_length == 0:
        return True
    if source_type == "code":
        return False
    if quality.alpha_ratio < 0.1:
        return True
    return False
