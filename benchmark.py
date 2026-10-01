"""Benchmark: keyword baseline (A) vs similarity-only (B) vs ChatEval, against HUMAN labels.

    python benchmark.py                 # uses the provider in config.yaml (real model, needs a key / Ollama)
    python benchmark.py --mock-llm      # offline smoke test of the code; NOT evidence
    python benchmark.py --repeats 3     # repeat ChatEval to measure judge non-determinism

Nothing here is invented: every figure is computed from benchmark/benchmark_set.csv labels and from
real runs. Anything that cannot be measured is printed as a placeholder or "not measured".
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from chateval.config import load_settings
from chateval.llm import make_llm
from chateval.metrics import compute_metrics, pct
from chateval.models import BotReply, TestCase
from chateval.pipeline import evaluate_replies
from chateval.prompts import PROMPT_VERSION

BENCH = Path("benchmark/benchmark_set.csv")
TIMING = Path("benchmark/manual_timing.csv")
TODO = "⟦TODO⟧"


def load_benchmark(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["human_label"] not in ("PASS", "FAIL"):
            raise SystemExit(f"{r['id']}: human_label must be PASS or FAIL, got {r['human_label']!r}")
    return rows


def baseline_keyword(answer: str, keywords: str) -> str:
    """Baseline A: the classic 'expected text contains X' assertion (all keywords must appear)."""
    wanted = [k.strip().lower() for k in keywords.split("|") if k.strip()]
    return "PASS" if wanted and all(k in answer.lower() for k in wanted) else "FAIL"


async def run_chateval(rows: list[dict], settings) -> tuple[list, float, dict]:
    cases = [TestCase(r["id"], r["question"], r["groundtruth"], r.get("source", "")) for r in rows]
    replies = [BotReply(r["chatbot_answer"], 0.0) for r in rows]
    llm = make_llm(settings)
    start = time.perf_counter()
    try:
        results = await evaluate_replies(cases, replies, llm, settings)
    finally:
        await llm.aclose()
    return results, time.perf_counter() - start, dict(llm.usage)


def read_manual_timing() -> list[float]:
    if not TIMING.exists():
        return []
    values = []
    with TIMING.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                values.append(float(r["seconds_to_review"]))
            except (ValueError, TypeError, KeyError):
                pass
    return values


def metric_table(cols: dict[str, dict]) -> str:
    rows = [
        ("Accuracy (auto-decided items only)", "accuracy_auto"),
        ("Precision (of auto FAIL, truly defective)", "precision"),
        ("Recall - defects auto-FAILed", "recall_auto"),
        ("Defects caught incl. those sent to review", "defect_catch_incl_review"),
        ("**False-pass rate** (defects auto-PASSed)", "false_pass_rate"),
        ("% of ALL answers routed to human review", "review_rate"),
        ("% of CLEAN answers routed to human review", "clean_sent_to_review_rate"),
    ]
    out = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for label, key in rows:
        out.append(f"| {label} | " + " | ".join(pct(m[key]) for m in cols.values()) + " |")
    out.append("| Items auto-decided / total | " + " | ".join(f"{m['auto_decided']}/{m['n']}" for m in cols.values()) + " |")
    out.append("| Defects falsely auto-passed (count) | " + " | ".join(f"{m['false_pass_count']}/{m['defects']}" for m in cols.values()) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--llm", choices=["openai", "ollama", "mock"])
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="run ChatEval N times to measure judge variance")
    ap.add_argument("--out", default="results/benchmark_results.md")
    args = ap.parse_args()

    settings = load_settings(args.config, "mock" if args.mock_llm else args.llm)
    rows = load_benchmark(BENCH)
    truth = [r["human_label"] for r in rows]
    verified = sum(r["label_status"].strip().upper() == "VERIFIED" for r in rows)

    runs = []
    for i in range(max(1, args.repeats)):
        results, wall, usage = asyncio.run(run_chateval(rows, settings))
        runs.append((results, wall, usage))
        print(f"run {i + 1}/{args.repeats}: {wall:.1f}s")
    results, wall, usage = runs[0]

    pred_a = [baseline_keyword(r["chatbot_answer"], r["expected_keywords"]) for r in rows]
    pred_b = ["PASS" if res.evaluation.similarity is not None and res.evaluation.similarity >= settings.thresholds.similarity_pass
              else "FAIL" for res in results]
    pred_c = [res.decision.verdict.value.replace("NEEDS_HUMAN_REVIEW", "REVIEW") for res in results]
    cols = {"A: keyword": compute_metrics(zip(truth, pred_a)),
            "B: similarity only": compute_metrics(zip(truth, pred_b)),
            "ChatEval (3 evaluators)": compute_metrics(zip(truth, pred_c))}

    # per-variant breakdown
    by_var = defaultdict(lambda: [0, Counter(), Counter(), Counter()])
    for r, a, b, c in zip(rows, pred_a, pred_b, pred_c):
        v = by_var[r["variant"]]
        v[0] += 1; v[1][a] += 1; v[2][b] += 1; v[3][c] += 1
    fmt = lambda c, keys: " / ".join(str(c[k]) for k in keys)  # noqa: E731
    label_of = {r["variant"]: r["human_label"] for r in rows}
    breakdown = ["| Answer type | Human label | n | A: PASS/FAIL | B: PASS/FAIL | ChatEval: PASS/FAIL/REVIEW |", "|---|---|---|---|---|---|"]
    for variant, (n, a, b, c) in sorted(by_var.items()):
        breakdown.append(f"| {variant} | {label_of[variant]} | {n} | {fmt(a, ('PASS', 'FAIL'))} | {fmt(b, ('PASS', 'FAIL'))} | {fmt(c, ('PASS', 'FAIL', 'REVIEW'))} |")

    md = [f"# ChatEval benchmark results", "",
          f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · judge: `{settings.provider}/{settings.chat_model}` · "
          f"embeddings: `{settings.embedding_model}` · prompts {PROMPT_VERSION} · "
          f"thresholds {vars(settings.thresholds)} · {len(rows)} labelled answers ({sum(t == 'FAIL' for t in truth)} defective)_", ""]
    if settings.provider == "mock":
        md += ["> ⚠️ **MOCK MODE.** The offline heuristic stands in for the LLM. This run only proves the code works. "
               "It is **not evidence** of ChatEval's accuracy. Re-run without `--mock-llm`.", ""]
    if verified < len(rows):
        md += [f"> ⚠️ **Labels not fully verified:** {verified} of {len(rows)} rows are marked `VERIFIED` in "
               "`benchmark/benchmark_set.csv`; the rest are still AI-drafted proposals. "
               "Treat every number below as provisional until a human has checked each label.", ""]
    md += ["## Accuracy and defect detection", "",
           "Positive class = defect (human label FAIL). `REVIEW` = tool declined to decide, so those items are excluded from "
           "accuracy/precision/recall and reported separately.", "", metric_table(cols), "",
           "## Where each approach gets it wrong (by answer type)", "", *breakdown, "",
           "## Judge variance", ""]
    if args.repeats > 1:
        per = [compute_metrics(zip(truth, [r.decision.verdict.value.replace("NEEDS_HUMAN_REVIEW", "REVIEW") for r in run[0]])) for run in runs]
        span = lambda k: f"{pct(min(m[k] for m in per if m[k] is not None))} – {pct(max(m[k] for m in per if m[k] is not None))}"  # noqa: E731
        md += [f"{args.repeats} repeated ChatEval runs (temperature {settings.temperature}): accuracy {span('accuracy_auto')}; "
               f"false-pass rate {span('false_pass_rate')}; review rate {span('review_rate')}."]
    else:
        md += [f"Not measured in this run (`--repeats 1`). Re-run with `--repeats 3` or more. {TODO}"]
    md += ["", "## Time and cost", ""]
    manual = read_manual_timing()
    n = len(rows)
    md += [f"- ChatEval wall-clock for {n} answers: **{wall:.1f}s** ({wall / n:.2f}s per answer, concurrency {settings.concurrency})"
           + (" (offline mock; real model latency will be far higher)" if settings.provider == "mock" else ""),
           f"- Token usage this run: {usage}"]
    p = settings.pricing
    if all(p.get(k) is not None for k in ("chat_input_per_1m_usd", "chat_output_per_1m_usd", "embedding_per_1m_usd")):
        cost = (usage["prompt_tokens"] * p["chat_input_per_1m_usd"] + usage["completion_tokens"] * p["chat_output_per_1m_usd"]
                + usage["embedding_tokens"] * p["embedding_per_1m_usd"]) / 1e6
        md.append(f"- Cost of this run at the prices in config.yaml: **${cost:.4f}** (${cost / n:.5f} per answer)")
    else:
        md.append(f"- Cost: not computed. Fill `pricing:` in config.yaml from your provider's current price list. {TODO}")
    if manual:
        mean = statistics.mean(manual)
        review_n = cols["ChatEval (3 evaluators)"]["n"] - cols["ChatEval (3 evaluators)"]["auto_decided"]
        md += [f"- Manual review, measured by you: **{mean:.0f}s per answer** (mean of {len(manual)} timed answers, "
               f"range {min(manual):.0f}–{max(manual):.0f}s)",
               f"- Reviewing all {n} answers manually ≈ **{mean * n / 60:.0f} min**; "
               f"reviewing only the {review_n} routed to humans ≈ **{mean * review_n / 60:.0f} min** "
               "(assumes a routed item takes the same time as a manually reviewed one; PASS/FAIL spot-checks not included)"]
        if len(manual) < 20:
            md.append(f"- Only {len(manual)} of 20 planned timing samples are filled in.")
    else:
        md += [f"- Manual review time per answer: {TODO} fill `benchmark/manual_timing.csv` (time yourself on the 20 listed answers)",
               f"- Time saved: {TODO} computed automatically once timing is filled in"]
    md += ["", "## Not measured / caveats", "",
           "- Baseline A's strength depends on which keywords were chosen per question (`expected_keywords`); those were chosen to be what a tester would plausibly assert (the key fact), but another tester might choose differently.",
           "- The benchmark is small (see n above); percentages move several points per single item. Report counts, not just rates.",
           "- Answers are frozen, so this measures evaluation quality only, not the bot itself.",
           "- Thresholds were not tuned on this set unless you did so yourself; if you did, say so, and hold out a fresh set.",
           "- Non-English answers, long answers and multi-turn conversations are not covered."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")

    rows_csv = Path(args.out).with_suffix(".rows.csv")
    with rows_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "variant", "human_label", "A_keyword", "B_similarity", "chateval", "reason", "judge_score", "similarity"])
        for r, a, b, c, res in zip(rows, pred_a, pred_b, pred_c, results):
            ev = res.evaluation
            w.writerow([r["id"], r["variant"], r["human_label"], a, b, c, res.decision.reason,
                        "" if ev.judge is None else ev.judge.score, "" if ev.similarity is None else round(ev.similarity, 3)])
    print("\n".join(md))
    print(f"\nWrote {args.out} and {rows_csv}")


if __name__ == "__main__":
    main()
