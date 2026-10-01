"""Plain data containers shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "NEEDS_HUMAN_REVIEW"


@dataclass
class TestCase:
    __test__ = False  # stop pytest trying to collect this as a test class
    id: str
    question: str
    groundtruth: str
    source: str = ""
    topic: str = ""
    high_risk: bool = False


@dataclass
class BotReply:
    answer: str
    latency_s: float
    error: str | None = None


@dataclass
class Claim:
    text: str
    status: str  # supported | unsupported | contradicted


@dataclass
class JudgeResult:
    score: float          # median across runs
    issue: str            # none | wrong_fact | missing_info | hallucination | outdated | refusal | other
    reasoning: str
    spread: float = 0.0   # max - min across repeated runs (0 for a single run)
    runs: int = 1


@dataclass
class HallucinationResult:
    claims: list[Claim] = field(default_factory=list)

    @property
    def flagged(self) -> list[Claim]:
        return [c for c in self.claims if c.status != "supported"]

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)

    @property
    def has_contradiction(self) -> bool:
        return any(c.status == "contradicted" for c in self.claims)


@dataclass
class Evaluation:
    similarity: float | None = None
    judge: JudgeResult | None = None
    hallucination: HallucinationResult | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class Decision:
    verdict: Verdict
    reason: str   # machine-readable code, see verdict.py
    detail: str = ""


@dataclass
class CaseResult:
    case: TestCase
    reply: BotReply
    evaluation: Evaluation
    decision: Decision
