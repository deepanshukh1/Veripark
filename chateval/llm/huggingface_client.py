"""Hugging Face Inference API backend. Works without an SDK and supports free hosted models."""
from __future__ import annotations

import httpx

from .base import LLMClient, LLMError, parse_json_object, post_json


def _extract_text(payload: object) -> str:
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "output_text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
            choice = payload["choices"][0]
            if isinstance(choice, dict):
                if "text" in choice and isinstance(choice["text"], str):
                    return choice["text"]
                if "message" in choice and isinstance(choice["message"], dict):
                    content = choice["message"].get("content")
                    if isinstance(content, str):
                        return content
        if "details" in payload and isinstance(payload["details"], dict):
            detail_text = payload["details"].get("generated_text") or payload["details"].get("text")
            if isinstance(detail_text, str):
                return detail_text
    elif isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return _extract_text(payload[0])
        if payload and isinstance(payload[0], list):
            return _extract_text(payload[0])
        if payload and isinstance(payload[0], str):
            return payload[0]
    raise LLMError(f"could not extract generated text from Hugging Face response: {payload!r}")


def _coerce_embeddings(payload: object) -> list[list[float]]:
    if isinstance(payload, dict):
        for key in ("embeddings", "embedding", "data"):
            if key in payload:
                return _coerce_embeddings(payload[key])
        raise LLMError(f"missing embeddings in Hugging Face response: {payload!r}")

    if isinstance(payload, list):
        if not payload:
            return []
        if isinstance(payload[0], (int, float)):
            return [[float(v) for v in payload]]
        if isinstance(payload[0], list):
            return [[float(v) for v in row] for row in payload]
        if isinstance(payload[0], dict):
            items = []
            for item in payload:
                values = item.get("embedding") or item.get("values")
                if isinstance(values, list):
                    items.append([float(v) for v in values])
            if items:
                return items

    raise LLMError(f"unsupported Hugging Face embedding shape: {payload!r}")


class HuggingFaceClient(LLMClient):
    name = "huggingface"

    def __init__(self, api_token: str | None, chat_model: str, embedding_model: str,
                 base_url: str = "https://api-inference.huggingface.co",
                 temperature: float = 0.0, timeout_s: float = 60.0) -> None:
        super().__init__()
        self._api_token = api_token
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._base = base_url.rstrip("/")
        self._temperature = temperature
        self._headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def complete_json(self, system: str, user: str) -> dict:
        prompt = f"{system}\n\n{user}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": self._temperature,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        data = await post_json(self._client, f"{self._base}/models/{self._chat_model}", payload, self._headers)
        self.usage["calls"] += 1
        return parse_json_object(_extract_text(data))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"inputs": texts[0] if len(texts) == 1 else texts}
        data = await post_json(self._client, f"{self._base}/pipeline/feature-extraction/{self._embedding_model}",
                              payload, self._headers)
        return _coerce_embeddings(data)

    async def aclose(self) -> None:
        await self._client.aclose()
