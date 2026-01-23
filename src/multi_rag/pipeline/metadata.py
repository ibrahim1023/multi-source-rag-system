# Metadata helpers for chunk records.

from __future__ import annotations


def merge_metadata(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    merged.update(overrides)
    return merged
