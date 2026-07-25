"""Model clients.

Both providers go through one interface with identical retry, timeout, and
temperature handling. This symmetry is a requirement of the experiment, not a
convenience: if one provider got different retry behaviour or a different
temperature, any measured difference between arms would be confounded.

Raw `requests` rather than the vendor SDKs, so the two paths stay visibly
parallel and the repo has one dependency.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests


class PermanentAPIError(RuntimeError):
    """A 4xx that will never succeed on retry — bad model name, malformed
    request, unsupported parameter. Retrying these wastes time and buries the
    response body, which is the only place the real reason appears."""


class TruncatedResponse(RuntimeError):
    """The model hit its output-token cap. For a reasoning model this often
    means the entire budget went to reasoning and the visible content is empty
    or cut mid-JSON. Treating that as 'no findings' would silently deflate the
    affected model's recall — so it raises instead."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class ModelClient:
    """Base interface. Subclasses implement `_call`."""

    def __init__(self, model: str, api_key: str, temperature: Optional[float] = 0.0,
                 max_tokens: int = 4096, timeout: int = 120, max_retries: int = 4,
                 reasoning_effort: str = "medium"):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    def complete(self, system: str, user: str) -> LLMResponse:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call(system, user)
            except (PermanentAPIError, TruncatedResponse):
                # Retrying with identical parameters cannot fix either of these.
                raise
            except Exception as exc:  # noqa: BLE001 - retry transport errors only
                last_err = exc
                if attempt == self.max_retries - 1:
                    break
                # Exponential backoff, identical for both providers.
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.model} failed after {self.max_retries} attempts: {last_err}")

    @staticmethod
    def _check(resp: "requests.Response", model: str) -> None:
        """Raise with the provider's own error text included.

        `raise_for_status()` alone gives 'Bad Request' and nothing else, which
        is useless for debugging an unsupported parameter.
        """
        if resp.status_code < 400:
            return
        try:
            detail = json.dumps(resp.json(), indent=2)[:1200]
        except Exception:  # noqa: BLE001
            detail = resp.text[:1200]
        message = f"{model}: HTTP {resp.status_code}\n{detail}"
        # 408/429 and 5xx are worth retrying; everything else in 4xx is not.
        if resp.status_code in (408, 429) or resp.status_code >= 500:
            raise RuntimeError(message)
        raise PermanentAPIError(message)

    def _call(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class AnthropicClient(ModelClient):
    ENDPOINT = "https://api.anthropic.com/v1/messages"

    def _payload(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        # Models with thinking enabled reject an explicit temperature. Set
        # `temperature: null` in config.yaml to omit it entirely.
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    def _call(self, system: str, user: str) -> LLMResponse:
        resp = requests.post(
            self.ENDPOINT,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=self._payload(system, user),
            timeout=self.timeout,
        )
        self._check(resp, self.model)
        data = resp.json()
        if data.get("stop_reason") == "max_tokens":
            raise TruncatedResponse(
                f"{self.model}: hit max_tokens ({self.max_tokens}) — output is "
                f"truncated. Raise max_tokens in config.yaml."
            )
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=self.model,
        )


class OpenAIClient(ModelClient):
    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def _call(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_tokens,
        }
        # Reasoning models reject an explicit temperature; everything else gets
        # the same value the Anthropic path uses.
        if self._is_reasoning_model():
            # Recorded explicitly rather than left to the provider default, so
            # the value used is part of the run's configuration.
            payload["reasoning_effort"] = self.reasoning_effort
        elif self.temperature is not None:
            payload["temperature"] = self.temperature

        resp = requests.post(
            self.ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        self._check(resp, self.model)
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        if choice.get("finish_reason") == "length":
            # For reasoning models, max_completion_tokens includes reasoning
            # tokens — the whole budget can be spent before any visible output.
            raise TruncatedResponse(
                f"{self.model}: hit max_completion_tokens ({self.max_tokens}) — "
                f"content is empty or truncated (reasoning tokens count against "
                f"the cap). Raise max_tokens in config.yaml."
            )
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )

    def _is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return m.startswith("o1") or m.startswith("o3") or m.startswith("o4") or m.startswith("gpt-5")


class MockClient(ModelClient):
    """Deterministic offline client, for testing the pipeline without keys.

    Returns canned findings so the full generate -> review -> normalize ->
    score path can be exercised in CI.
    """

    def __init__(self, model: str = "mock", **kwargs):
        super().__init__(model=model, api_key="mock", **kwargs)

    def _call(self, system: str, user: str) -> LLMResponse:
        upper = system.upper()
        filename = self._guess_filename(user)
        if "DEFECT LEDGER" in upper:
            body = json.dumps({
                "defects": [
                    {
                        "category": "missing-edge-case",
                        "line_start": 3,
                        "line_end": 4,
                        "severity": "major",
                        "description": "mock ground-truth defect",
                    },
                    {
                        "category": "null-boundary",
                        "line_start": 6,
                        "line_end": 6,
                        "severity": "critical",
                        "description": "mock boundary defect",
                    },
                ]
            })
        elif "REVIEW" in upper:
            body = json.dumps({
                "findings": [
                    {
                        "category": "missing-edge-case",
                        "file": filename,
                        "line": 3,
                        "severity": "major",
                        "rationale": f"mock finding from {self.model}",
                    },
                    {
                        "category": "logic-error",
                        "file": filename,
                        "line": 99,
                        "severity": "minor",
                        "rationale": f"mock false positive from {self.model}",
                    },
                ]
            })
        else:
            body = "```python\ndef solve(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs)\n```"
        return LLMResponse(text=body, input_tokens=100, output_tokens=50, model=self.model)

    @staticmethod
    def _guess_filename(user: str) -> str:
        # Anchor on the "File:" marker. Matching any backticked *.py picks up
        # filenames mentioned in the conventions block instead.
        import re as _re
        match = _re.search(r"File:\s*`([\w./-]+)`", user)
        return match.group(1) if match else "solution.py"


# --- Factory ---------------------------------------------------------------

PROVIDERS = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "mock": MockClient,
}

ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def build_client(provider: str, model: str, temperature: Optional[float] = 0.0,
                 max_tokens: int = 4096, reasoning_effort: str = "medium") -> ModelClient:
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}'; expected one of {list(PROVIDERS)}")
    if provider == "mock":
        return MockClient(model=model, temperature=temperature, max_tokens=max_tokens)

    env_key = ENV_KEYS[provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(
            f"Missing {env_key}. Set it in your environment or .env before running."
        )
    return PROVIDERS[provider](
        model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
