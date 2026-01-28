#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

ingest_markdown() {
  local title="$1"
  local origin="$2"
  local file_path="$3"
  local text
  text=$(python - <<PY
import json
from pathlib import Path
print(json.dumps(Path("$file_path").read_text(encoding="utf-8")))
PY
)
  curl -s "${API_URL}/ingest/markdown" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"${title}\",\"origin\":\"${origin}\",\"text\":${text}}" >/dev/null
}

ingest_code() {
  local title="$1"
  local origin="$2"
  local file_path="$3"
  local text
  text=$(python - <<PY
import json
from pathlib import Path
print(json.dumps(Path("$file_path").read_text(encoding="utf-8")))
PY
)
  curl -s "${API_URL}/ingest/code" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"${title}\",\"origin\":\"${origin}\",\"text\":${text}}" >/dev/null
}

ingest_pdf() {
  local title="$1"
  local origin="$2"
  local file_path="$3"
  curl -s "${API_URL}/ingest/pdf/file" \
    -F "file=@${file_path}" \
    -F "title=${title}" \
    -F "origin=${origin}" >/dev/null
}

ingest_markdown "Log Retention Policy" "demo_sources/log-retention-policy.md" \
  "${ROOT_DIR}/demo_sources/log-retention-policy.md"
ingest_code "Rate Limiting" "demo_sources/rate-limiting.md" \
  "${ROOT_DIR}/demo_sources/rate-limiting.md"
ingest_markdown "Engineering Onboarding" "demo_sources/onboarding.md" \
  "${ROOT_DIR}/demo_sources/onboarding.md"
ingest_pdf "Log Retention Policy PDF" "demo_sources/log-retention-policy.pdf" \
  "${ROOT_DIR}/demo_sources/log-retention-policy.pdf"

echo "Demo sources ingested."
