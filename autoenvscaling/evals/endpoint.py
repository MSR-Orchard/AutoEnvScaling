"""The OpenAI-compatible endpoint a solver or evaluated model is served from.

What model answered is part of the result. A served model name alone does not prove which
weights answered, and a router that silently falls back to another provider changes the model
mid-run without failing anything. So the endpoint records the exact model id, which environment
variables hold the URL and key (never the key itself), and, for routers such as OpenRouter, a
provider allow-list with fallback disabled. ``provenance()`` goes into every report.

The model runs inside the task sandbox (the Pi agent dials out from there), so the URL must be
reachable from the sandbox, not only from the machine that launches the job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class EndpointSpec:
    model: str                                   # exact served model id
    base_url_env: str = "AES_EVAL_BASE_URL"      # env var holding the base URL (ending in /v1)
    api_key_env: str = "AES_EVAL_API_KEY"        # env var holding the API key
    api: str = "openai-completions"              # Pi API style: openai-completions or openai-responses
    context_window: int = 262_144
    max_output_tokens: int = 32_768
    # Optional router pinning, sent as the request's "provider" field (OpenRouter's format),
    # e.g. {"order": ["SomeProvider"], "allow_fallbacks": False}.
    provider_routing: dict[str, Any] | None = None
    sampling: dict[str, float] | None = None     # decoding vector; None keeps server defaults
    extra: dict[str, Any] = field(default_factory=dict)

    def base_url(self, env: dict[str, str] | None = None) -> str:
        env = os.environ if env is None else env
        url = (env.get(self.base_url_env) or "").strip()
        if not url:
            raise RuntimeError(f"{self.base_url_env} is not set; it must hold the endpoint URL")
        return normalize_base_url(url)

    def provenance(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url_env": self.base_url_env,
            "api": self.api,
            "provider_routing": self.provider_routing,
            "sampling": self.sampling,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
        }


def normalize_base_url(url: str) -> str:
    """Strip trailing slashes and make sure the URL ends in /v1."""
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


@dataclass(frozen=True)
class PreflightResult:
    requested_model: str
    returned_model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int


def preflight(spec: EndpointSpec, *, env: dict[str, str] | None = None,
              transport: httpx.BaseTransport | None = None,
              timeout_s: float = 120.0) -> PreflightResult:
    """Send one tiny chat request and check that the endpoint serves the exact model.

    Run this once before a paid evaluation: an unreachable URL or a wrong model id otherwise
    shows up as hundreds of failed trials. It makes one billable request.
    """
    env = os.environ if env is None else env
    payload: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": "Reply with exactly: PREFLIGHT_OK"}],
        "max_tokens": 64,
        "temperature": 0,
    }
    if spec.provider_routing:
        payload["provider"] = spec.provider_routing
    headers = {}
    key = env.get(spec.api_key_env, "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(transport=transport, timeout=timeout_s) as client:
        response = client.post(f"{spec.base_url(env)}/chat/completions", json=payload,
                               headers=headers)
    response.raise_for_status()
    data = response.json()
    returned = data.get("model") or ""
    # Routers may append a date suffix to the canonical id (model-YYYYMMDD).
    if returned != spec.model and not returned.startswith(spec.model + "-"):
        raise RuntimeError(f"endpoint returned model {returned!r}, expected {spec.model!r}")
    if not (data.get("choices") or []):
        raise RuntimeError("endpoint returned no choices")
    usage = data.get("usage") or {}
    return PreflightResult(
        requested_model=spec.model,
        returned_model=returned,
        provider=data.get("provider"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )
