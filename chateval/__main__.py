"""CLI:  python -m chateval run --input data/sample_questions.csv --target mock --mock-llm"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_settings
from .io import load_cases, write_audit, write_csv
from .llm import make_llm
from .pipeline import check_gate, run_cases, summarize
from .prompts import PROMPT_VERSION
from .report import render_report
from .target import make_target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chateval", description="LLM-as-judge regression testing for chatbots")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate a chatbot against a question/groundtruth file")
    r.add_argument("--input", required=True, help="CSV/XLSX with question, groundtruth [, id, source, topic, risk]")
    r.add_argument("--target", default="mock", help="'mock' (bundled bot, in-process) or an http(s) chat endpoint URL")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--llm", choices=["openai", "gemini", "huggingface", "ollama", "mock"],
                   help="override provider from config.yaml; openai falls back to gemini then huggingface if keys are missing")
    r.add_argument("--mock-llm", action="store_true", help="fully offline heuristic 'LLM' (no key needed; demo only)")
    r.add_argument("--out", default="results/run", help="output directory")
    r.add_argument("--limit", type=int, help="only the first N questions")
    r.add_argument("--gate", action="store_true", help="exit 1 if the CI gate rules in config.yaml are violated")
    return p


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, "mock" if args.mock_llm else args.llm)
    cases = load_cases(args.input, settings.high_risk_keywords)[: args.limit]
    llm, target = make_llm(settings), make_target(args.target, settings.target["request_field"],
                                                  settings.target["response_field"], settings.timeout_s)
    started = time.perf_counter()
    try:
        results = await run_cases(cases, target, llm, settings)
    finally:
        await target.aclose()
        await llm.aclose()
    runtime = time.perf_counter() - started

    meta = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "target": args.target,
            "provider": settings.provider, "chat_model": settings.chat_model, "embedding_model": settings.embedding_model,
            "prompt_version": PROMPT_VERSION, "judge_runs": settings.judge_runs, "runtime_s": round(runtime, 2),
            "token_usage": llm.usage, "thresholds": vars(settings.thresholds)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(results, out / "results.csv", settings.chat_model)
    write_audit(results, out / "audit.jsonl", meta)
    (out / "report.html").write_text(render_report(results, meta), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    s = summarize(results)
    print(f"ChatEval: {s['total']} questions in {runtime:.1f}s  |  PASS {s['pass']}  FAIL {s['fail']}  "
          f"NEEDS_HUMAN_REVIEW {s['review']}  (pass rate {s['pass_rate']:.0%})")
    if settings.provider == "mock":
        print("NOTE: --mock-llm uses an offline heuristic, not a real model. Good for a demo, not for evidence.")
    print(f"Report: {out / 'report.html'}\nCSV:    {out / 'results.csv'}\nAudit:  {out / 'audit.jsonl'}")

    if args.gate:
        problems = check_gate(results, settings.gate)
        for p in problems:
            print(f"GATE FAILED: {p}")
        return 1 if problems else 0
    return 0


def main() -> None:
    args = build_parser().parse_args()
    try:
        sys.exit(asyncio.run(_run(args)))
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
