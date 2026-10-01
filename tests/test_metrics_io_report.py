import csv

import pytest

from chateval.io import load_cases, safe_cell, write_csv
from chateval.metrics import compute_metrics
from chateval.models import (BotReply, CaseResult, Decision, Evaluation, HallucinationResult,
                             JudgeResult, TestCase, Verdict, Claim)
from chateval.pipeline import check_gate
from chateval.report import render_report


def test_metrics_hand_computed():
    #  truth, predicted
    pairs = [("FAIL", "FAIL"), ("FAIL", "PASS"), ("FAIL", "REVIEW"), ("PASS", "PASS"),
             ("PASS", "FAIL"), ("PASS", "REVIEW"), ("PASS", "PASS")]
    m = compute_metrics(pairs)
    assert m["n"] == 7 and m["defects"] == 3 and m["auto_decided"] == 5
    assert m["accuracy_auto"] == pytest.approx(3 / 5)           # FAIL/FAIL, PASS/PASS x2
    assert m["precision"] == pytest.approx(1 / 2)               # 1 TP, 1 FP
    assert m["recall_auto"] == pytest.approx(1 / 3)
    assert m["defect_catch_incl_review"] == pytest.approx(2 / 3)
    assert m["false_pass_rate"] == pytest.approx(1 / 3)
    assert m["review_rate"] == pytest.approx(2 / 7)
    assert m["clean_sent_to_review_rate"] == pytest.approx(1 / 4)


def test_metrics_empty_denominators_are_none_not_zero():
    m = compute_metrics([("PASS", "PASS")])
    assert m["precision"] is None and m["false_pass_rate"] is None


def test_safe_cell_blocks_formula_injection():
    assert safe_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)" and safe_cell("hello") == "hello"


def test_load_cases_validates(tmp_path):
    good = tmp_path / "q.csv"
    good.write_text("question,groundtruth,id\nWhat is the overdraft fee?,£5,a1\n,,\n", encoding="utf-8")
    cases = load_cases(good, ("fee",))
    assert len(cases) == 1 and cases[0].id == "a1" and cases[0].high_risk
    bad = tmp_path / "bad.csv"
    bad.write_text("question,groundtruth\nQ?,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(bad)
    with pytest.raises(ValueError):
        load_cases(tmp_path / "x.txt")


def _result(verdict, reason="r", answer="a", hr=False):
    ev = Evaluation(0.5, JudgeResult(0.3, "wrong_fact", "judge says no"),
                    HallucinationResult([Claim("bad claim", "contradicted")]))
    return CaseResult(TestCase("1", "q?", "truth", high_risk=hr), BotReply(answer, 0.1), ev, Decision(verdict, reason, "d"))


META = {"run_at": "now", "target": "t", "provider": "p", "chat_model": "m", "prompt_version": "v1",
        "thresholds": {"judge_pass": 0.7, "review_margin": 0.1, "similarity_pass": 0.75}}


def test_report_escapes_untrusted_bot_output():
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    page = render_report([_result(Verdict.FAIL, "wrong_fact", answer=evil)], META)
    assert "<script>alert" not in page and "<img src=x" not in page and "&lt;script&gt;" in page


def test_csv_neutralises_formulas(tmp_path):
    out = tmp_path / "r.csv"
    write_csv([_result(Verdict.FAIL, "wrong_fact", answer="=1+1")], out, "m")
    assert list(csv.DictReader(out.open()))[0]["bot_answer"] == "'=1+1"


def test_gate_rules():
    results = [_result(Verdict.PASS), _result(Verdict.FAIL, "wrong_fact", hr=True), _result(Verdict.REVIEW)]
    problems = check_gate(results, {"min_pass_rate": 0.9, "max_review_rate": 0.25, "fail_on_high_risk_fail": True})
    assert len(problems) == 3
    assert check_gate(results, {}) == []
