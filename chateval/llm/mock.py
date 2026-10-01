"""Fully offline stand-in for an LLM, so reviewers can run everything with no API key.

IMPORTANT: this is a crude lexical heuristic (word overlap, number matching, refusal and
negation checks). It exists to exercise the pipeline end to end. Its results say NOTHING about
how well a real LLM judge performs, and benchmark runs in this mode are labelled as such.
"""
from __future__ import annotations

import html
import math
import re
import zlib
from collections import Counter

from .base import LLMClient

_STOP = set("""a an the is are was were be been being to of for and or in on at by with from your you we our it its
this that these those as can will may do does did has have had if there their they them i me my so than then
please also any all up out about into over""".split())
_SYN = {"charge": "fee", "charged": "fee", "charges": "fee", "cost": "fee", "costs": "fee",
        "phone": "call", "ring": "call", "within": "in"}
_NUMWORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_TOKEN = re.compile(r"\d+(?:\.\d+)?%?|[a-z]+")
_REFUSAL = re.compile(r"(can'?t help|cannot help|unable to (help|answer)|not able to (help|answer|give)|"
                      r"i'?m sorry, i|don'?t have (that )?information)", re.I)
_NEG = re.compile(r"\b(no|not|never|cannot|without)\b|n't", re.I)
_INSTR = re.compile(r"<(question|groundtruth|source|answer)>(.*?)</\1>", re.S)


def _norm(text: str) -> str:
    text = html.unescape(text).lower().replace("£", "").replace("$", "")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(_norm(text)):
        tok = _NUMWORDS.get(tok, tok)
        tok = _SYN.get(tok, tok)
        if not tok[0].isdigit() and len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _content(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOP]


def _numbers(text: str) -> set[str]:
    return {t for t in _tokens(text) if t[0].isdigit()}


def _fields(user: str) -> dict[str, str]:
    return {name: html.unescape(body) for name, body in _INSTR.findall(user)}


class MockLLM(LLMClient):
    name = "mock"
    DIM = 512

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        toks = _content(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        vec = [0.0] * self.DIM
        for gram, count in Counter(grams).items():
            vec[zlib.crc32(gram.encode()) % self.DIM] += math.sqrt(count)
        return vec

    async def complete_json(self, system: str, user: str) -> dict:
        self.usage["calls"] += 1
        fields = _fields(user)
        if system.startswith("TASK: judge"):
            return self._judge(fields)
        if system.startswith("TASK: hallucination"):
            return self._hallucination(fields)
        raise ValueError("MockLLM got an unknown task")

    # ---- heuristics -------------------------------------------------------
    def _judge(self, f: dict[str, str]) -> dict:
        gt, ans = f.get("groundtruth", ""), f.get("answer", "")
        if _REFUSAL.search(ans):
            if re.search(r"\b(decline|should (politely )?refuse|not give)\b", gt, re.I):
                return {"score": 0.9, "issue": "none", "reasoning": "Refusal is the expected behaviour here."}
            return {"score": 0.05, "issue": "refusal", "reasoning": "The bot refused a question it should answer."}
        gt_t, ans_t = set(_content(gt)), set(_content(ans))
        recall = len(gt_t & ans_t) / max(1, len(gt_t))
        score, issue, why = min(1.0, recall * 1.25), "none", "Answer covers the groundtruth facts."
        gt_n, ans_n = _numbers(gt), _numbers(ans)
        if gt_n - ans_n or ans_n - gt_n:
            score, issue = min(score, 0.25), "wrong_fact"
            why = f"Numbers differ from groundtruth (expected {sorted(gt_n)}, answer has {sorted(ans_n)})."
        elif bool(_NEG.search(gt)) != bool(_NEG.search(ans)):
            score, issue, why = min(score, 0.2), "wrong_fact", "Answer reverses the polarity (yes/no) of the groundtruth."
        elif recall < 0.5:
            issue, why = "missing_info", "Answer misses most of the groundtruth content."
        return {"score": round(score, 2), "issue": issue, "reasoning": why}

    def _hallucination(self, f: dict[str, str]) -> dict:
        ref = f.get("groundtruth", "") + " " + f.get("source", "")
        ref_t, ref_n = set(_content(ref)), _numbers(ref)
        claims = []
        sentences = re.split(r"(?<=[.!?])\s+", f.get("answer", "").strip())
        # Split into clauses so an added requirement ("..., and a reference letter") is judged on its own.
        clauses = [c for s in sentences if not _REFUSAL.search(s)
                   for c in re.split(r",\s+(?:and\s+)?|;\s*|\s+and\s+", s)]
        for sent in clauses:
            toks = _content(sent)
            if len(toks) < 2 or _REFUSAL.search(sent):
                continue
            support = sum(t in ref_t for t in toks) / len(toks)
            bad_nums = _numbers(sent) - ref_n
            if bad_nums:
                status = "contradicted" if ref_n else "unsupported"
            elif support < 0.4:
                status = "unsupported"
            else:
                status = "supported"
            claims.append({"claim": sent.strip()[:160], "status": status})
        return {"claims": claims}
