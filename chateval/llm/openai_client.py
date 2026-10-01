"""OpenAI-compatible backend using plain HTTPS (no SDK dependency).
Works with api.openai.com, Azure/OpenAI-compatible gateways via OPENAI_BASE_URL."""
from __future__ import annotations

import httpx

from .base import LLMClient, parse_json_object, post_json


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, api_key: str, chat_model: str, embedding_model: str,
                 base_url: str = "https://api.openai.com/v1", temperature: float = 0.0,
                 timeout_s: float = 60.0) -> None:
        super().__init__()
        self._chat_model, self._embedding_model = chat_model, embedding_model
        self._base, self._temperature = base_url.rstrip("/"), temperature
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def complete_json(self, system: str, user: str) -> dict:
        data = await post_json(self._client, f"{self._base}/chat/completions", {
            "model": self._chat_model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, self._headers)
        usage = data.get("usage", {})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        return parse_json_object(data["choices"][0]["message"]["content"])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        data = await post_json(self._client, f"{self._base}/embeddings",
                               {"model": self._embedding_model, "input": texts}, self._headers)
        self.usage["embedding_tokens"] += data.get("usage", {}).get("total_tokens", 0)
        return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]

    async def aclose(self) -> None:
        await self._client.aclose()
