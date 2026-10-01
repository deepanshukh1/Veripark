"""Local-model backend (Ollama). Lets the tool run with no paid key and no data leaving the machine."""
from __future__ import annotations

import httpx

from .base import LLMClient, parse_json_object, post_json


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, host: str, chat_model: str, embedding_model: str,
                 temperature: float = 0.0, timeout_s: float = 120.0) -> None:
        super().__init__()
        self._host, self._chat_model, self._embedding_model = host.rstrip("/"), chat_model, embedding_model
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def complete_json(self, system: str, user: str) -> dict:
        data = await post_json(self._client, f"{self._host}/api/chat", {
            "model": self._chat_model, "stream": False, "format": "json",
            "options": {"temperature": self._temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += data.get("prompt_eval_count", 0)
        self.usage["completion_tokens"] += data.get("eval_count", 0)
        return parse_json_object(data["message"]["content"])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        data = await post_json(self._client, f"{self._host}/api/embed",
                               {"model": self._embedding_model, "input": texts})
        return data["embeddings"]

    async def aclose(self) -> None:
        await self._client.aclose()
