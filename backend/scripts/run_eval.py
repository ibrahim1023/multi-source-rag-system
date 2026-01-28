#!/usr/bin/env python3
"""Run offline evaluation over the gold set."""

from __future__ import annotations
from multi_rag.eval.offline import evaluate_gold_set, load_gold_set
from multi_rag.api.app import _settings_from_env, build_dependencies

import argparse
import json
from pathlib import Path
import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_thresholds(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _rehydrate_indexes(dependencies) -> int:
    documents = dependencies.metadata_store.list_documents()
    if not documents:
        return 0
    indexed = 0
    for document in documents:
        chunks = dependencies.metadata_store.list_chunks(document.doc_id)
        if not chunks:
            continue
        dependencies.indexer.index_document(
            document, chunks, include_bm25=not dependencies.bm25_loaded
        )
        indexed += 1
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation runner.")
    parser.add_argument(
        "--gold",
        default="eval/gold_set.jsonl",
        help="Path to the gold set JSONL.",
    )
    parser.add_argument(
        "--config",
        default="eval/config.json",
        help="Path to the eval config JSON.",
    )
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override retrieval top-k.")
    args = parser.parse_args()

    gold_path = Path(args.gold)
    config_path = Path(args.config)
    cases = load_gold_set(gold_path)
    thresholds = _load_thresholds(config_path)

    settings = _settings_from_env()
    dependencies = build_dependencies(settings)
    indexed_docs = _rehydrate_indexes(dependencies)
    if indexed_docs == 0:
        print("Warning: no documents found in metadata store. Did you ingest data?")
    if args.top_k is not None:
        top_k = args.top_k
    else:
        top_k = settings.top_k

    report = evaluate_gold_set(
        cases,
        retriever=dependencies.retriever,
        pipeline=dependencies.pipeline,
        metadata_store=dependencies.metadata_store,
        top_k=top_k,
        thresholds=thresholds,
    )

    print("Offline evaluation results")
    print(f"Cases: {report.metrics.total_cases}")
    print(f"Retrieval hit rate: {report.metrics.retrieval_hit_rate:.2f}")
    print(f"Citation coverage: {report.metrics.citation_coverage:.2f}")
    print(f"Refusal accuracy: {report.metrics.refusal_accuracy:.2f}")
    print(f"Answered rate: {report.metrics.answered_rate:.2f}")
    print(f"Avg claims/answer: {report.metrics.avg_claims_per_answer:.2f}")
    print(
        f"Avg citations/answer: {report.metrics.avg_citations_per_answer:.2f}")
    if report.failures:
        print("Regression failures:")
        for failure in report.failures:
            print(f"- {failure}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
