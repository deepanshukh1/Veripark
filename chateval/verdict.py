"""Combine the three evaluator signals into PASS / FAIL / NEEDS_HUMAN_REVIEW.

Design principle: the LLM judge is the primary signal; similarity and the hallucination check are
independent cross-checks. When signals disagree or the judge is near the threshold we do NOT guess,
we route to a human. Pure function, no I/O, easy to unit-test.

  judge_clear_pass  = judge >= judge_pass + margin
  judge_clear_fail  = judge <= judge_pass - margin
  otherwise         = borderline

  evaluator error / unstable judge ............................ REVIEW
  judge_clear_pass + no unsupported claims .................... PASS
  judge_clear_pass + flagged claims + low similarity .......... FAIL    (2 of 3 checks say bad)
  judge_clear_pass + flagged claims + similar text ............ REVIEW  (fluent-but-wrong suspicion)
  judge_clear_fail + (flagged claims or low similarity) ....... FAIL
  judge_clear_fail + both cross-checks fine ................... REVIEW  (judge may be too harsh)
  borderline judge ............................................ REVIEW
"""
from __future__ import annotations

from .config import Thresholds
from .models import Decision, Evaluation, Verdict

REASON_PASS = "all_checks_agree"


def decide(ev: Evaluation, th: Thresholds) -> Decision:
    if ev.errors or ev.judge is None or ev.similarity is None or ev.hallucination is None:
        return Decision(Verdict.REVIEW, "evaluator_error", "; ".join(ev.errors) or "missing evaluator output")

    judge = ev.judge.score
    if ev.judge.spread > th.max_judge_spread:
        return Decision(Verdict.REVIEW, "unstable_judge",
                        f"judge scores varied by {ev.judge.spread:.2f} across {ev.judge.runs} runs")

    eps = 1e-9
    clear_pass = judge + eps >= th.judge_pass + th.review_margin
    clear_fail = judge - eps <= th.judge_pass - th.review_margin
    low_similarity = ev.similarity < th.similarity_pass
    flagged = ev.hallucination.n_flagged > 0

    if clear_pass:
        if not flagged:
            return Decision(Verdict.PASS, REASON_PASS, "judge pass, no unsupported claims")
        if low_similarity:
            return Decision(Verdict.FAIL, _fail_reason(ev), "judge passed but fact-check and similarity both objected")
        return Decision(Verdict.REVIEW, "judge_pass_but_hallucination_flagged",
                        f"{ev.hallucination.n_flagged} claim(s) not supported by the reference")
    if clear_fail:
        if flagged or low_similarity:
            return Decision(Verdict.FAIL, _fail_reason(ev), ev.judge.reasoning)
        return Decision(Verdict.REVIEW, "judge_fail_but_other_checks_pass",
                        "judge scored low but similarity and fact-check found no problem")
    return Decision(Verdict.REVIEW, "borderline_judge_score", f"judge score {judge:.2f} is near the pass line")


def _fail_reason(ev: Evaluation) -> str:
    """Category used to group failures in the report (judge's issue label wins)."""
    if ev.judge and ev.judge.issue not in ("none", "other"):
        return ev.judge.issue
    if ev.hallucination and ev.hallucination.n_flagged:
        return "hallucination"
    return "low_similarity"
