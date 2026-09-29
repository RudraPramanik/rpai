"""Protocols for embedding, LLM, and transcription providers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embed one or more texts into dense vectors."""

    @property
    def model_id(self) -> str:
        """Configured embedding model identifier."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """LLM completion façade used by Phase 5 chat (sync + streaming)."""

    @property
    def model_id(self) -> str:
        """Configured LLM model identifier."""
        ...

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return a single completion string for the prompt."""
        ...

    def stream_complete(
        self, prompt: str, *, system: str | None = None
    ) -> Iterator[str]:
        """Yield incremental text deltas for the prompt."""
        ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Speech-to-text façade used by voice transcription."""

    @property
    def model_id(self) -> str:
        """Configured transcription model identifier."""
        ...

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str | None = None,
    ) -> str:
        """Return transcribed text for the audio payload."""
        ...
