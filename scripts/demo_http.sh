#!/usr/bin/env bash
# Runs the mock bot as a real HTTP server, evaluates it over HTTP, then shuts it down.
set -euo pipefail
PY=${PY:-python}
$PY -m uvicorn mock_bot.app:app --port 8000 --log-level warning &
BOT=$!
trap 'kill $BOT 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 0.5
done
$PY -m chateval run --input data/sample_questions.csv --target http://127.0.0.1:8000/chat --mock-llm --out results/demo_http
