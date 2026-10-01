"""Factory: pick a backend from settings."""
from __future__ import annotations

import os

from ..config import Settings
from .base import LLMClient, LLMError
from .gemini_client import GeminiClient
from .huggingface_client import HuggingFaceClient
from .mock import MockLLM
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

__all__ = ["make_llm", "LLMClient", "LLMError"]


def _provider_order(requested: str) -> list[str]:
    if requested == "mock":
        return ["mock"]
    if requested == "openai":
        return ["openai", "gemini", "huggingface"]
    if requested == "gemini":
        return ["gemini", "huggingface"]
    if requested == "huggingface":
        return ["huggingface"]
    if requested == "ollama":
        return ["ollama"]
    return [requested]


def make_llm(settings: Settings) -> LLMClient:
    for provider in _provider_order(settings.provider):
        if provider == "mock":
            return MockLLM()
        if provider == "ollama":
            return OllamaClient(os.getenv("OLLAMA_HOST", "http://localhost:11434"), settings.chat_model,
                                settings.embedding_model, settings.temperature, settings.timeout_s)
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if key:
                return OpenAIClient(key, settings.chat_model, settings.embedding_model,
                                    os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                    settings.temperature, settings.timeout_s)
            continue
        if provider == "gemini":
            key = os.getenv("GEMINI_API_KEY")
            if key:
                return GeminiClient(key, settings.chat_model, settings.embedding_model,
                                    os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
                                    settings.temperature, settings.timeout_s)
            continue
        if provider == "huggingface":
            token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
            if token or os.getenv("HUGGINGFACE_MODEL") or os.getenv("HF_MODEL"):
                return HuggingFaceClient(token, settings.chat_model, settings.embedding_model,
                                         os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co"),
                                         settings.temperature, settings.timeout_s)
            continue

    raise SystemExit(
        f"No usable LLM provider is configured for {settings.provider!r}. "
        "Set OPENAI_API_KEY, GEMINI_API_KEY, or HUGGINGFACE_API_TOKEN / HF_TOKEN, "
        "or use --mock-llm or --llm ollama."
    )
