"""Where answers come from: an HTTP chatbot endpoint, or the bundled mock bot run in-process."""
from __future__ import annotations

import time

import httpx

from .models import BotReply


class HttpTarget:
    def __init__(self, url: str, request_field: str = "question", response_field: str = "answer",
                 timeout_s: float = 60.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._url, self._req, self._resp = url, request_field, response_field
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def ask(self, question: str) -> BotReply:
        start = time.perf_counter()
        try:
            resp = await self._client.post(self._url, json={self._req: question})
            resp.raise_for_status()
            answer = str(resp.json()[self._resp])
            return BotReply(answer, time.perf_counter() - start)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return BotReply("", time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")

    async def aclose(self) -> None:
        await self._client.aclose()


def make_target(spec: str, request_field: str = "question", response_field: str = "answer",
                timeout_s: float = 60.0) -> HttpTarget:
    if spec == "mock":
        # Same HTTP code path as a real bot, but served in-process (no port, no server to start).
        from mock_bot.app import app
        return HttpTarget("http://mockbot/chat", request_field, response_field, timeout_s,
                          transport=httpx.ASGITransport(app=app))
    if spec.startswith(("http://", "https://")):
        return HttpTarget(spec, request_field, response_field, timeout_s)
    raise SystemExit(f"--target must be 'mock' or an http(s) URL, got {spec!r}")
