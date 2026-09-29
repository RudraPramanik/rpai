"""LiteLLM-backed embedding and LLM providers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Sequence

from litellm import completion, embedding


class ProviderConfigError(RuntimeError):
    """Raised when required provider credentials or settings are missing."""


def _ensure_gemini_credentials(gemini_api_key: str | None) -> None:
    """Ensure GEMINI_API_KEY is available to LiteLLM when using gemini/* models.

    Accepts Settings key, GEMINI_API_KEY, GEMINAI_API_KEY, or GENAI_API_KEY.
    """
    key = (
        gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINAI_API_KEY")
        or os.environ.get("GENAI_API_KEY")
    )
    if not key or not str(key).strip():
        raise ProviderConfigError(
            "GEMINI_API_KEY is not set. Add GEMINI_API_KEY (or GEMINAI_API_KEY / "
            "GENAI_API_KEY) to backend/.env (see .env.example). Do not hardcode keys."
        )
    os.environ["GEMINI_API_KEY"] = str(key).strip()


def _require_gemini_key_for_model(model: str) -> None:
    if model.startswith("gemini/") and not os.environ.get("GEMINI_API_KEY"):
        raise ProviderConfigError(
            "GEMINI_API_KEY is not set. Add GEMINI_API_KEY (or GEMINAI_API_KEY) to "
            "backend/.env before calling Gemini models."
        )


def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _delta_text(chunk: object) -> str:
    """Extract text delta from a LiteLLM streaming chunk (object or dict)."""
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices", [])
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.delta if hasattr(choice, "delta") else choice.get("delta", {})
    if delta is None:
        return ""
    content = delta.content if hasattr(delta, "content") else delta.get("content")
    return str(content or "")


class LiteLLMEmbeddingProvider:
    """Hosted embeddings via LiteLLM (default: gemini/gemini-embedding-001)."""

    def __init__(
        self,
        model: str,
        *,
        gemini_api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        if model.startswith("gemini/"):
            _ensure_gemini_credentials(gemini_api_key)

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        _require_gemini_key_for_model(self._model)
        kwargs: dict = {"model": self._model, "input": list(texts)}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = embedding(**kwargs)
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])
        if not data:
            raise RuntimeError(f"Empty embedding response from model {self._model!r}")
        # LiteLLM may return objects or dicts; sort by index for stability.
        items = sorted(
            data,
            key=lambda item: item["index"]
            if isinstance(item, dict)
            else getattr(item, "index", 0),
        )
        vectors: list[list[float]] = []
        for item in items:
            vec = item["embedding"] if isinstance(item, dict) else item.embedding
            vectors.append(list(vec))
        return vectors


class LiteLLMProvider:
    """Hosted chat/completion via LiteLLM (sync + streaming)."""

    def __init__(self, model: str, *, gemini_api_key: str | None = None) -> None:
        self._model = model
        if model.startswith("gemini/"):
            _ensure_gemini_credentials(gemini_api_key)

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        _require_gemini_key_for_model(self._model)
        messages = _build_messages(prompt, system)
        response = completion(model=self._model, messages=messages)
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"Empty completion response from model {self._model!r}")
        message = choices[0].message if hasattr(choices[0], "message") else choices[0]["message"]
        content = message.content if hasattr(message, "content") else message["content"]
        return str(content or "")

    def stream_complete(
        self, prompt: str, *, system: str | None = None
    ) -> Iterator[str]:
        _require_gemini_key_for_model(self._model)
        messages = _build_messages(prompt, system)
        stream = completion(model=self._model, messages=messages, stream=True)
        for chunk in stream:
            text = _delta_text(chunk)
            if text:
                yield text
