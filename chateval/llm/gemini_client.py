"""Google Gemini backend using the public REST API."""
from __future__ import annotations

import httpx

from .base import LLMClient, LLMError, parse_json_object, post_json


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, api_key: str, chat_model: str, embedding_model: str,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 temperature: float = 0.0, timeout_s: float = 60.0) -> None:
        super().__init__()
        self._api_key = api_key
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._base = base_url.rstrip("/")
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def complete_json(self, system: str, user: str) -> dict:
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self._temperature,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self._base}/models/{self._chat_model}:generateContent?key={self._api_key}"
        data = await post_json(self._client, url, payload)
        usage = data.get("usageMetadata", {})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(usage.get("promptTokenCount", 0))
        self.usage["completion_tokens"] += int(usage.get("candidatesTokenCount", 0) or usage.get("completionTokenCount", 0))

        try:
            response_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Gemini response: {data!r}") from exc
        return parse_json_object(response_text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base}/models/{self._embedding_model}:batchEmbedContents?key={self._api_key}"
        payload = {
            "requests": [
                {"model": f"models/{self._embedding_model}", "content": {"parts": [{"text": text}]}}
                for text in texts
            ]
        }
        data = await post_json(self._client, url, payload)
        embeddings: list[list[float]] = []
        for item in data.get("embeddings", []):
            values = item.get("values") or item.get("embedding", {}).get("values") or []
            embeddings.append([float(v) for v in values])
        return embeddings

    async def aclose(self) -> None:
        await self._client.aclose()
