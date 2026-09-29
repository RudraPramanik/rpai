"""Protocols for embedding and LLM providers."""

from __future__ import annotations

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
    """Thin completion façade for Phase 5 chat reuse."""

    @property
    def model_id(self) -> str:
        """Configured LLM model identifier."""
        ...

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return a single completion string for the prompt."""
        ...
