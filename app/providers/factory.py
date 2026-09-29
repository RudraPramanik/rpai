"""Factories that resolve providers from application settings."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.litellm_provider import LiteLLMEmbeddingProvider, LiteLLMProvider


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider (LiteLLM-backed by default)."""
    cfg = settings or get_settings()
    return LiteLLMEmbeddingProvider(
        cfg.embedding_model,
        gemini_api_key=cfg.gemini_api_key,
    )


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the configured LLM provider (LiteLLM-backed by default)."""
    cfg = settings or get_settings()
    return LiteLLMProvider(
        cfg.llm_model,
        gemini_api_key=cfg.gemini_api_key,
    )
