"""Verdict logic: every branch, no LLM involved."""
import pytest

from chateval.config import Thresholds
from chateval.models import Claim, Evaluation, HallucinationResult, JudgeResult, Verdict
from chateval.verdict import decide

TH = Thresholds(similarity_pass=0.75, judge_pass=0.7, review_margin=0.1, max_judge_spread=0.25)


def ev(judge=0.95, sim=0.9, flagged=0, issue="none", spread=0.0, errors=None):
    claims = [Claim("x", "unsupported")] * flagged + [Claim("ok", "supported")]
    return Evaluation(similarity=sim, judge=JudgeResult(judge, issue, "because", spread, 3),
                      hallucination=HallucinationResult(claims), errors=errors or [])


@pytest.mark.parametrize("kwargs, verdict, reason", [
    (dict(), Verdict.PASS, "all_checks_agree"),
    (dict(judge=0.8), Verdict.PASS, "all_checks_agree"),                                    # exactly pass+margin
    (dict(sim=0.2), Verdict.PASS, "all_checks_agree"),                                      # paraphrase: low sim alone is not enough
    (dict(flagged=1), Verdict.REVIEW, "judge_pass_but_hallucination_flagged"),              # fluent-but-wrong suspicion
    (dict(flagged=1, sim=0.3, issue="hallucination"), Verdict.FAIL, "hallucination"),       # 2 of 3 checks object
    (dict(judge=0.2, issue="wrong_fact", flagged=1), Verdict.FAIL, "wrong_fact"),     # judge + fact-check agree
    (dict(judge=0.2, issue="none", flagged=1), Verdict.FAIL, "hallucination"),
    (dict(judge=0.2, issue="none", sim=0.3), Verdict.FAIL, "low_similarity"),
    (dict(judge=0.2, issue="refusal", sim=0.1), Verdict.FAIL, "refusal"),
    (dict(judge=0.2, sim=0.95, flagged=0), Verdict.REVIEW, "judge_fail_but_other_checks_pass"),  # judge may be too harsh
    (dict(judge=0.7), Verdict.REVIEW, "borderline_judge_score"),
    (dict(judge=0.65), Verdict.REVIEW, "borderline_judge_score"),
    (dict(judge=0.79), Verdict.REVIEW, "borderline_judge_score"),
    (dict(judge=0.6, sim=0.1, flagged=2), Verdict.FAIL, "hallucination"),                   # exactly pass-margin
    (dict(spread=0.4), Verdict.REVIEW, "unstable_judge"),
])
def test_decision_table(kwargs, verdict, reason):
    d = decide(ev(**kwargs), TH)
    assert (d.verdict, d.reason) == (verdict, reason)


def test_evaluator_error_always_goes_to_a_human():
    d = decide(Evaluation(similarity=0.9, judge=None, hallucination=None, errors=["judge: boom"]), TH)
    assert d.verdict is Verdict.REVIEW and d.reason == "evaluator_error" and "boom" in d.detail


def test_thresholds_are_respected():
    strict = Thresholds(judge_pass=0.9, review_margin=0.05)
    assert decide(ev(judge=0.9), strict).verdict is Verdict.REVIEW      # inside +/-0.05 of 0.9
    assert decide(ev(judge=0.96), strict).verdict is Verdict.PASS
