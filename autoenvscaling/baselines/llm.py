"""A minimal OpenAI-compatible chat client for the tool-free baselines.

One request per call, no tools, no streaming. Transport failures and HTTP 408, 429, 5xx and 524
responses are retried up to three attempts with a 10, 20 second backoff; other HTTP errors stop
immediately. A provider content filter is reported as ``PolicyRefusal`` so the run can leave that
assignment out of the denominator, as the paper's three-way comparison does. These limits follow
the research transport (``exp1_authoring/refresh_common.one_shot``).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 524})
MAX_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 64_000
DEFAULT_TIMEOUT_SEC = 900.0


class PolicyRefusal(RuntimeError):
    """The provider refused the request (content filter)."""


class EndpointError(RuntimeError):
    """The endpoint failed in a way that is not the model's answer (transport, auth, quota)."""


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    wall_sec: float = 0.0


@dataclass
class ChatClient:
    """``POST {base_url}/chat/completions`` with a bearer key from ``api_key_env``."""

    base_url: str
    model: str
    api_key_env: str = "AES_PROPOSER_API_KEY"
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_effort: str | None = None
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, model: str, **kwargs: Any) -> "ChatClient":
        base_url = os.environ.get("AES_PROPOSER_BASE_URL", "").strip()
        if not base_url:
            raise ValueError("set AES_PROPOSER_BASE_URL to the proposer's "
                             "OpenAI-compatible endpoint")
        return cls(base_url=base_url, model=model, **kwargs)

    async def complete(self, prompt: str) -> Completion:
        body: dict[str, Any] = {"model": self.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": self.max_tokens, **self.extra_body}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        last_error = "no attempt made"
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            for attempt in range(MAX_ATTEMPTS):
                started = time.monotonic()
                try:
                    response = await client.post(url, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if response.status_code == 200:
                        return _parse(response.json(), time.monotonic() - started)
                    text = response.text[:2000]
                    if response.status_code == 422 and "cyber_policy" in text:
                        raise PolicyRefusal(f"HTTP 422: {text[:300]}")
                    if response.status_code not in RETRY_STATUS:
                        raise EndpointError(f"HTTP {response.status_code}: {text[:500]}")
                    last_error = f"HTTP {response.status_code}: {text[:300]}"
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(10 * (attempt + 1))
        raise EndpointError(f"endpoint failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _parse(data: dict[str, Any], wall_sec: float) -> Completion:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise EndpointError(f"malformed completion: {str(data)[:300]}") from exc
    finish = choice.get("finish_reason")
    text = (choice.get("message") or {}).get("content") or ""
    if finish == "content_filter":
        raise PolicyRefusal("provider finish_reason=content_filter")
    return Completion(text=text, finish_reason=finish, usage=data.get("usage") or {},
                      model=data.get("model"), wall_sec=round(wall_sec, 2))
