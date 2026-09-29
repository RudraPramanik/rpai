"""Factories that resolve providers from application settings."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.base import EmbeddingProvider, LLMProvider, TranscriptionProvider
from app.providers.litellm_provider import LiteLLMEmbeddingProvider, LiteLLMProvider
from app.providers.transcription import OpenAICompatibleTranscriptionProvider


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider (LiteLLM-backed by default)."""
    cfg = settings or get_settings()
    return LiteLLMEmbeddingProvider(
        cfg.embedding_model,
        gemini_api_key=cfg.resolved_gemini_api_key(),
        dimensions=cfg.embedding_dimensions,
    )


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the configured LLM provider (sync + streaming via same instance)."""
    cfg = settings or get_settings()
    return LiteLLMProvider(
        cfg.llm_model,
        gemini_api_key=cfg.resolved_gemini_api_key(),
    )


def get_transcription_provider(
    settings: Settings | None = None,
) -> TranscriptionProvider:
    """Return the configured STT provider (NVIDIA NIM/NVCF OpenAI-compatible by default)."""
    cfg = settings or get_settings()
    return OpenAICompatibleTranscriptionProvider(
        cfg.stt_model,
        api_key=cfg.nvidia_nim_api_key,
        api_base=cfg.stt_api_base,
        language=cfg.stt_language,
        send_model_field=cfg.stt_send_model_field,
    )
