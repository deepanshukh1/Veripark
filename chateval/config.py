"""Settings: YAML for behaviour, .env for secrets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Thresholds:
    similarity_pass: float = 0.75
    judge_pass: float = 0.7
    review_margin: float = 0.1
    max_judge_spread: float = 0.25


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    temperature: float = 0.0
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    judge_runs: int = 1
    concurrency: int = 5
    timeout_s: float = 60.0
    thresholds: Thresholds = field(default_factory=Thresholds)
    route_high_risk_passes_to_review: bool = False
    high_risk_keywords: tuple[str, ...] = ()
    gate: dict = field(default_factory=dict)
    pricing: dict = field(default_factory=dict)
    target: dict = field(default_factory=lambda: {"request_field": "question", "response_field": "answer"})


def load_settings(path: str | Path = "config.yaml", provider: str | None = None) -> Settings:
    load_dotenv()  # reads .env if present; harmless otherwise
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    llm = raw.get("llm", {})
    provider = provider or llm.get("provider", "openai")
    models = llm.get(provider, {})
    if not models and provider in {"gemini", "huggingface"}:
        models = llm.get("openai", {})
    th = {
        **raw.get("thresholds", {}),
        **raw.get("provider_overrides", {}).get(provider, {}).get("thresholds", {}),
    }
    policy = raw.get("policy", {})
    return Settings(
        provider=provider,
        temperature=float(llm.get("temperature", 0)),
        chat_model=models.get("chat_model", "mock-judge"),
        embedding_model=models.get("embedding_model", "mock-embed"),
        judge_runs=int(raw.get("judge_runs", 1)),
        concurrency=int(raw.get("concurrency", 5)),
        timeout_s=float(raw.get("timeout_s", 60)),
        thresholds=Thresholds(**th),
        route_high_risk_passes_to_review=bool(policy.get("route_high_risk_passes_to_review", False)),
        high_risk_keywords=tuple(k.lower() for k in policy.get("high_risk_keywords", [])),
        gate=raw.get("ci_gate", {}),
        pricing=raw.get("pricing", {}),
        target=raw.get("target", {"request_field": "question", "response_field": "answer"}),
    )
