"""Evaluators with a scripted fake LLM (no network, no heuristics)."""
import asyncio

import pytest

from chateval.evaluators import cosine, evaluate, hallucination_check, llm_judge, semantic_similarity
from chateval.llm.base import LLMClient, LLMError, parse_json_object
from chateval.models import TestCase

CASE = TestCase("t1", "What is the fee?", "The fee is £5.")


class FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, judge_scores=(0.9,), claims=None, vectors=None, fail=None):
        super().__init__()
        self.judge_scores, self.claims, self.vectors, self.fail = list(judge_scores), claims or [], vectors, fail
        self.calls = 0

    async def complete_json(self, system, user):
        if self.fail and self.fail in system:
            raise LLMError("simulated outage")
        if system.startswith("TASK: judge"):
            score = self.judge_scores[self.calls % len(self.judge_scores)]
            self.calls += 1
            return {"score": score, "issue": "none", "reasoning": "ok"}
        return {"claims": self.claims}

    async def embed(self, texts):
        return self.vectors or [[1.0, 0.0], [1.0, 0.0]]


def run(coro):
    return asyncio.run(coro)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_similarity_is_clamped_and_uses_embeddings():
    assert run(semantic_similarity(FakeLLM(vectors=[[1, 0], [0, 1]]), CASE, "a")) == pytest.approx(0.0)
    assert run(semantic_similarity(FakeLLM(vectors=[[1, 0], [-1, 0]]), CASE, "a")) == 0.0  # negative clamped


def test_judge_median_and_spread_over_repeated_runs():
    res = run(llm_judge(FakeLLM(judge_scores=(0.9, 0.5, 0.8)), CASE, "a", runs=3))
    assert res.score == pytest.approx(0.8) and res.spread == pytest.approx(0.4) and res.runs == 3


@pytest.mark.parametrize("bad", ["high", None, 1.5, -0.1])
def test_judge_rejects_invalid_scores(bad):
    with pytest.raises(LLMError):
        run(llm_judge(FakeLLM(judge_scores=(bad,)), CASE, "a"))


def test_hallucination_flags_unsupported_and_contradicted_and_unknown_labels():
    claims = [{"claim": "a", "status": "supported"}, {"claim": "b", "status": "unsupported"},
              {"claim": "c", "status": "contradicted"}, {"claim": "d", "status": "weird"}]
    res = run(hallucination_check(FakeLLM(claims=claims), CASE, "x"))
    assert res.n_flagged == 3 and res.has_contradiction
    assert res.claims[3].status == "unsupported"   # unknown label fails safe


def test_evaluate_runs_all_three_and_isolates_failures():
    ok = run(evaluate(FakeLLM(), CASE, "a"))
    assert ok.errors == [] and ok.judge and ok.hallucination and ok.similarity is not None
    broken = run(evaluate(FakeLLM(fail="TASK: hallucination"), CASE, "a"))
    assert broken.hallucination is None and broken.judge is not None
    assert len(broken.errors) == 1 and "hallucination" in broken.errors[0]


def test_parse_json_object_tolerates_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_object("no json here")
