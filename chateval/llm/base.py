"""The one interface every model backend implements."""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod

import httpx


class LLMError(Exception):
    """Any failure talking to, or parsing output from, a model."""


class LLMClient(ABC):
    name = "base"

    def __init__(self) -> None:
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}

    @abstractmethod
    async def complete_json(self, system: str, user: str) -> dict:
        """Return the model's JSON object reply (temperature 0)."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    async def aclose(self) -> None:  # pragma: no cover - trivial
        return None


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model text, tolerating ```json fences and chatter."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise LLMError(f"model did not return JSON: {text[:120]!r}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"unparseable JSON from model: {text[:120]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError("model JSON was not an object")
    return data


async def post_json(client: httpx.AsyncClient, url: str, payload: dict,
                    headers: dict | None = None, retries: int = 3) -> dict:
    """POST with exponential backoff on rate limits / transient errors."""
    for attempt in range(retries):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.TransportError as exc:
            if attempt == retries - 1:
                raise LLMError(f"network error calling {url}: {exc}") from exc
            await asyncio.sleep(2 ** attempt)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"HTTP {exc.response.status_code} from {url}: {exc.response.text[:200]}") from exc
    raise LLMError("unreachable")  # pragma: no cover
