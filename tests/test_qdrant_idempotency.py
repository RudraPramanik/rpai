"""Qdrant upsert idempotency tests using a fake embedding provider (no Gemini)."""

from __future__ import annotations

from typing import Sequence

from app.config import Settings
from app.rag.chunking import chunk_markdown_file
from app.rag.qdrant_store import (
    ensure_collection,
    get_qdrant_client,
    point_id_for,
    upsert_chunks,
)
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_KNOWLEDGE = _BACKEND / "data" / "knowledge"
_TEST_COLLECTION = "portfolio_knowledge_test"


class FakeEmbeddingProvider:
    model_id = "fake/test-embed"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Deterministic tiny vectors from text length / hash
        out: list[list[float]] = []
        for t in texts:
            h = float(sum(ord(c) for c in t) % 1000) / 1000.0
            out.append([h, 1.0 - h, 0.5, 0.25])
        return out


def _test_settings() -> Settings:
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION=_TEST_COLLECTION,
        EMBEDDING_DIMENSIONS=4,
        EMBEDDING_MODEL="fake/test-embed",
    )


def test_point_id_deterministic() -> None:
    a = point_id_for("projects/oms.md", 0)
    b = point_id_for("projects/oms.md", 0)
    c = point_id_for("projects/oms.md", 1)
    assert a == b
    assert a != c


def test_reingest_does_not_double_count() -> None:
    settings = _test_settings()
    provider = FakeEmbeddingProvider()
    client = get_qdrant_client(settings)
    if client.collection_exists(_TEST_COLLECTION):
        client.delete_collection(_TEST_COLLECTION)

    ensure_collection(client, settings, provider)
    files = sorted(_KNOWLEDGE.rglob("*.md"))
    chunks = []
    for path in files:
        chunks.extend(chunk_markdown_file(path, _KNOWLEDGE))
    assert chunks

    vectors = provider.embed([c.text for c in chunks])
    upsert_chunks(
        client, settings, chunks, vectors, embedding_model=provider.model_id
    )
    count1 = client.count(collection_name=_TEST_COLLECTION, exact=True).count

    upsert_chunks(
        client, settings, chunks, vectors, embedding_model=provider.model_id
    )
    count2 = client.count(collection_name=_TEST_COLLECTION, exact=True).count

    sources = {c.source for c in chunks}
    file_sources = {p.relative_to(_KNOWLEDGE).as_posix() for p in files}
    assert sources == file_sources
    assert count1 == len(chunks)
    assert count2 == count1

    client.delete_collection(_TEST_COLLECTION)
