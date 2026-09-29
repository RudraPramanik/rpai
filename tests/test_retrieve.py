"""Retrieve filter smoke tests with fake embeddings (no Gemini)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.config import Settings
from app.rag.chunking import chunk_markdown_file
from app.rag.qdrant_store import ensure_collection, get_qdrant_client, upsert_chunks
from app.rag.retrieve import retrieve

_BACKEND = Path(__file__).resolve().parents[1]
_KNOWLEDGE = _BACKEND / "data" / "knowledge"
_TEST_COLLECTION = "portfolio_knowledge_retrieve_test"


class FakeEmbeddingProvider:
    model_id = "fake/test-embed"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # Bias OMS project chunks toward a distinctive direction
            lower = t.lower()
            if "project: oms" in lower or "oms" in lower:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "project: laddu" in lower:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                h = float(sum(ord(c) for c in t) % 1000) / 1000.0
                out.append([0.0, 0.0, h, 1.0 - h])
        return out


def _settings() -> Settings:
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION=_TEST_COLLECTION,
        EMBEDDING_DIMENSIONS=4,
        EMBEDDING_MODEL="fake/test-embed",
    )


def test_filtered_retrieve_project_oms() -> None:
    settings = _settings()
    provider = FakeEmbeddingProvider()
    client = get_qdrant_client(settings)
    if client.collection_exists(_TEST_COLLECTION):
        client.delete_collection(_TEST_COLLECTION)

    ensure_collection(client, settings, provider)
    chunks = []
    for path in sorted(_KNOWLEDGE.rglob("*.md")):
        chunks.extend(chunk_markdown_file(path, _KNOWLEDGE))
    vectors = provider.embed([c.text for c in chunks])
    upsert_chunks(
        client, settings, chunks, vectors, embedding_model=provider.model_id
    )

    filtered = retrieve(
        "How does architecture work?",
        top_k=5,
        filters={"project": "oms"},
        settings=settings,
        embedding_provider=provider,
    )
    assert filtered
    assert all(r.metadata.get("project") == "oms" for r in filtered)

    # Query vector biased to OMS direction via fake provider
    unfiltered = retrieve(
        "Project: oms office management mobile",
        top_k=5,
        settings=settings,
        embedding_provider=provider,
    )
    assert unfiltered
    assert any(r.metadata.get("project") == "oms" for r in unfiltered[:3])

    client.delete_collection(_TEST_COLLECTION)
