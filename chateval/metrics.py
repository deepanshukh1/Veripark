"""Benchmark maths. Positive class = DEFECT (human label FAIL).

pairs: iterable of (truth, predicted) with truth in {PASS, FAIL} and predicted in {PASS, FAIL, REVIEW}.
REVIEW means the tool declined to decide and a human will, so it is excluded from accuracy/precision
but counted separately; the cost of REVIEW is reported as the share of clean answers sent to a human.
"""
from __future__ import annotations


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def compute_metrics(pairs) -> dict:
    pairs = list(pairs)
    n = len(pairs)
    defects = sum(t == "FAIL" for t, _ in pairs)
    clean = n - defects
    auto = [(t, p) for t, p in pairs if p != "REVIEW"]
    tp = sum(t == "FAIL" and p == "FAIL" for t, p in auto)
    fp = sum(t == "PASS" and p == "FAIL" for t, p in auto)
    false_pass = sum(t == "FAIL" and p == "PASS" for t, p in auto)
    defects_in_review = sum(t == "FAIL" and p == "REVIEW" for t, p in pairs)
    clean_in_review = sum(t == "PASS" and p == "REVIEW" for t, p in pairs)
    return {
        "n": n, "defects": defects, "clean": clean, "auto_decided": len(auto),
        "accuracy_auto": _ratio(sum(t == p for t, p in auto), len(auto)),
        "precision": _ratio(tp, tp + fp),
        "recall_auto": _ratio(tp, defects),
        "defect_catch_incl_review": _ratio(tp + defects_in_review, defects),
        "false_pass_rate": _ratio(false_pass, defects),
        "false_pass_count": false_pass,
        "review_rate": _ratio(n - len(auto), n),
        "clean_sent_to_review_rate": _ratio(clean_in_review, clean),
    }


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"
