# Cleaning utilities for ingested text.

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def strip_boilerplate(text: str) -> str:
    # Basic whitespace normalization; replace with stronger rules per source.
    cleaned = _WHITESPACE_RE.sub(" ", text)
    return cleaned.strip()
