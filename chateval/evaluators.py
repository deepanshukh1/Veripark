"""The three evaluators. Each is independent and runs concurrently via asyncio.gather."""
from __future__ import annotations

import asyncio
import math
import statistics

from .llm.base import LLMClient, LLMError
from .models import Claim, Evaluation, HallucinationResult, JudgeResult, TestCase
from .prompts import CLAIM_STATUSES, HALLUCINATION_SYSTEM, ISSUES, JUDGE_SYSTEM, build_user_prompt


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def semantic_similarity(llm: LLMClient, case: TestCase, answer: str) -> float:
    """Cosine similarity between the answer and groundtruth embeddings, clamped to 0..1."""
    answer_vec, truth_vec = await llm.embed([answer, case.groundtruth])
    return max(0.0, min(1.0, cosine(answer_vec, truth_vec)))


async def _judge_once(llm: LLMClient, case: TestCase, answer: str) -> tuple[float, str, str]:
    data = await llm.complete_json(JUDGE_SYSTEM, build_user_prompt(case.question, case.groundtruth, case.source, answer))
    try:
        score = float(data["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"judge returned no numeric score: {data!r}") from exc
    if not 0.0 <= score <= 1.0:
        raise LLMError(f"judge score out of range: {score}")
    issue = str(data.get("issue", "other")).lower()
    return score, issue if issue in ISSUES else "other", str(data.get("reasoning", ""))[:500]


async def llm_judge(llm: LLMClient, case: TestCase, answer: str, runs: int = 1) -> JudgeResult:
    """0-1 score + reasoning. With runs>1 the judge is repeated; we keep the median and the spread."""
    results = await asyncio.gather(*[_judge_once(llm, case, answer) for _ in range(max(1, runs))])
    ordered = sorted(results, key=lambda r: r[0])
    scores = [r[0] for r in ordered]
    mid = ordered[len(ordered) // 2]
    return JudgeResult(score=statistics.median(scores), issue=mid[1], reasoning=mid[2],
                       spread=max(scores) - min(scores), runs=len(scores))


async def hallucination_check(llm: LLMClient, case: TestCase, answer: str) -> HallucinationResult:
    """Extract claims from the answer and label each supported / unsupported / contradicted."""
    data = await llm.complete_json(HALLUCINATION_SYSTEM,
                                   build_user_prompt(case.question, case.groundtruth, case.source, answer))
    raw = data.get("claims")
    if not isinstance(raw, list):
        raise LLMError(f"hallucination check returned no claims list: {data!r}")
    claims = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).lower()
        # An unknown label is treated as unsupported: failing safe is better than silently passing.
        claims.append(Claim(text=str(item.get("claim", ""))[:300], status=status if status in CLAIM_STATUSES else "unsupported"))
    return HallucinationResult(claims)


async def evaluate(llm: LLMClient, case: TestCase, answer: str, judge_runs: int = 1) -> Evaluation:
    """Run all three evaluators concurrently. A failing evaluator is recorded, not raised,
    so one bad API call never aborts a regression run (the verdict logic sends it to a human)."""
    outcomes = await asyncio.gather(
        semantic_similarity(llm, case, answer),
        llm_judge(llm, case, answer, judge_runs),
        hallucination_check(llm, case, answer),
        return_exceptions=True,
    )
    ev = Evaluation()
    for name, outcome in zip(("similarity", "judge", "hallucination"), outcomes):
        if isinstance(outcome, Exception):
            ev.errors.append(f"{name}: {outcome}")
        else:
            setattr(ev, name, outcome)
    return ev
