PY ?= python

.PHONY: install demo demo-http test benchmark benchmark-mock screenshot bot

install:
	$(PY) -m pip install -r requirements.txt

# Fully offline: bundled bot served in-process + offline mock "LLM". No API key, no server.
demo:
	$(PY) -m chateval run --input data/sample_questions.csv --target mock --mock-llm --out results/demo
	@echo "Open results/demo/report.html"

# Same, but the bot runs as a real HTTP server on :8000.
demo-http:
	bash scripts/demo_http.sh

bot:
	$(PY) -m uvicorn mock_bot.app:app --port 8000

test:
	$(PY) -m pytest -q

# Real evidence run (needs OPENAI_API_KEY in .env, or `--llm ollama`). Writes results/benchmark_results.md
benchmark:
	$(PY) benchmark.py --repeats 3

# Code smoke test only. NOT evidence.
benchmark-mock:
	$(PY) benchmark.py --mock-llm --out results/benchmark_results_mock.md

screenshot:
	$(PY) scripts/screenshot.py
