# Offline Evaluation

This folder stores the gold set and regression thresholds for Task 11.

## Files

- `gold_set.jsonl`: One JSON object per line. Required fields: `id`, `query`.
  Optional: `expected_sources` (list of origin/title/doc_id strings), `expected_refusal` (bool), `notes`.
- `config.json`: Thresholds for regression checks.

## How It Works

The evaluator checks three signals:

- Retrieval hit rate: any retrieved chunk matches an expected source.
- Citation coverage: share of claims with citations.
- Refusal accuracy: whether the system refused when expected.

Populate `expected_sources` with origins or titles that exist in your metadata store.
Expand the gold set to 30-80 questions before demo readiness.
