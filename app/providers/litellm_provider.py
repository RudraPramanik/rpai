"""LiteLLM-backed embedding and LLM providers."""

from __future__ import annotations

import os
from typing import Sequence

from litellm import completion, embedding


class ProviderConfigError(RuntimeError):
    """Raised when required provider credentials or settings are missing."""


def _ensure_gemini_credentials(gemini_api_key: str | None) -> None:
    """Ensure GEMINI_API_KEY is available to LiteLLM when using gemini/* models."""
    key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not str(key).strip():
        raise ProviderConfigError(
            "GEMINI_API_KEY is not set. Add it to backend/.env (see .env.example). "
            "Do not hardcode keys in source."
        )
    os.environ["GEMINI_API_KEY"] = str(key).strip()


class LiteLLMEmbeddingProvider:
    """Hosted embeddings via LiteLLM (default: Gemini text-embedding-004)."""

    def __init__(self, model: str, *, gemini_api_key: str | None = None) -> None:
        self._model = model
        if model.startswith("gemini/"):
            _ensure_gemini_credentials(gemini_api_key)

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model.startswith("gemini/") and not os.environ.get("GEMINI_API_KEY"):
            raise ProviderConfigError(
                "GEMINI_API_KEY is not set. Add it to backend/.env before embedding."
            )
        response = embedding(model=self._model, input=list(texts))
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
    """Hosted chat/completion via LiteLLM (Phase 5 façade)."""

    def __init__(self, model: str, *, gemini_api_key: str | None = None) -> None:
        self._model = model
        if model.startswith("gemini/"):
            _ensure_gemini_credentials(gemini_api_key)

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        if self._model.startswith("gemini/") and not os.environ.get("GEMINI_API_KEY"):
            raise ProviderConfigError(
                "GEMINI_API_KEY is not set. Add it to backend/.env before calling the LLM."
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = completion(model=self._model, messages=messages)
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"Empty completion response from model {self._model!r}")
        message = choices[0].message if hasattr(choices[0], "message") else choices[0]["message"]
        content = message.content if hasattr(message, "content") else message["content"]
        return str(content or "")
