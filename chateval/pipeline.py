"""Orchestration: ask the bot, evaluate concurrently, decide."""
from __future__ import annotations

import asyncio

from .config import Settings
from .evaluators import evaluate
from .llm.base import LLMClient
from .models import BotReply, CaseResult, Decision, Evaluation, TestCase, Verdict
from .verdict import decide


async def _judge_case(case: TestCase, reply: BotReply, llm: LLMClient, settings: Settings) -> CaseResult:
    if reply.error or not reply.answer.strip():
        detail = reply.error or "empty answer"
        return CaseResult(case, reply, Evaluation(), Decision(Verdict.REVIEW, "target_error", detail))
    ev = await evaluate(llm, case, reply.answer, settings.judge_runs)
    decision = decide(ev, settings.thresholds)
    if (settings.route_high_risk_passes_to_review and case.high_risk and decision.verdict is Verdict.PASS):
        decision = Decision(Verdict.REVIEW, "high_risk_topic", "policy: high-risk topics need human sign-off")
    return CaseResult(case, reply, ev, decision)


async def evaluate_replies(cases: list[TestCase], replies: list[BotReply], llm: LLMClient,
                           settings: Settings) -> list[CaseResult]:
    """Evaluate already-collected answers (used by benchmark.py with frozen answers)."""
    sem = asyncio.Semaphore(settings.concurrency)

    async def one(case: TestCase, reply: BotReply) -> CaseResult:
        async with sem:
            return await _judge_case(case, reply, llm, settings)

    return list(await asyncio.gather(*[one(c, r) for c, r in zip(cases, replies)]))


async def run_cases(cases: list[TestCase], target, llm: LLMClient, settings: Settings) -> list[CaseResult]:
    sem = asyncio.Semaphore(settings.concurrency)

    async def one(case: TestCase) -> CaseResult:
        async with sem:
            reply = await target.ask(case.question)
            return await _judge_case(case, reply, llm, settings)

    return list(await asyncio.gather(*[one(c) for c in cases]))


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    count = lambda v: sum(r.decision.verdict is v for r in results)  # noqa: E731
    return {"total": n, "pass": count(Verdict.PASS), "fail": count(Verdict.FAIL), "review": count(Verdict.REVIEW),
            "pass_rate": count(Verdict.PASS) / n if n else 0.0, "review_rate": count(Verdict.REVIEW) / n if n else 0.0}


def check_gate(results: list[CaseResult], gate: dict) -> list[str]:
    """CI gating rules from config.yaml. Returns a list of violations (empty = gate passes)."""
    s, problems = summarize(results), []
    if s["pass_rate"] < gate.get("min_pass_rate", 0.0):
        problems.append(f"pass rate {s['pass_rate']:.0%} < required {gate['min_pass_rate']:.0%}")
    if s["review_rate"] > gate.get("max_review_rate", 1.0):
        problems.append(f"review rate {s['review_rate']:.0%} > allowed {gate['max_review_rate']:.0%}")
    if gate.get("fail_on_high_risk_fail", False):
        bad = [r.case.id for r in results if r.case.high_risk and r.decision.verdict is Verdict.FAIL]
        if bad:
            problems.append(f"high-risk topics failed: {', '.join(bad)}")
    return problems
