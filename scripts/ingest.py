#!/usr/bin/env python3
"""Ingest knowledge Markdown into Qdrant (idempotent upserts)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/ingest.py` from backend/
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import get_settings
from app.providers.factory import get_embedding_provider
from app.rag.chunking import Chunk, chunk_markdown_file
from app.rag.qdrant_store import (
    ensure_collection,
    get_qdrant_client,
    upsert_chunks,
)

EMBED_BATCH_SIZE = 16


def iter_knowledge_files(knowledge_root: Path) -> list[Path]:
    if not knowledge_root.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_root}")
    return sorted(knowledge_root.rglob("*.md"))


def embed_batches(
    provider,
    chunks: list[Chunk],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors.extend(provider.embed([c.text for c in batch]))
    return vectors


def main() -> int:
    settings = get_settings()
    knowledge_root = settings.data_dir / "knowledge"
    files = iter_knowledge_files(knowledge_root)
    if not files:
        print(f"No Markdown files under {knowledge_root}")
        return 1

    provider = get_embedding_provider(settings)
    client = get_qdrant_client(settings)
    vector_size = ensure_collection(client, settings, provider)

    all_chunks: list[Chunk] = []
    for path in files:
        all_chunks.extend(chunk_markdown_file(path, knowledge_root))

    print(
        f"Ingesting {len(files)} files -> {len(all_chunks)} chunks "
        f"into {settings.qdrant_collection!r} "
        f"(model={provider.model_id}, dim={vector_size})"
    )

    vectors = embed_batches(provider, all_chunks)
    upserted = upsert_chunks(
        client,
        settings,
        all_chunks,
        vectors,
        embedding_model=provider.model_id,
    )

    count = client.count(collection_name=settings.qdrant_collection, exact=True)
    sources = sorted({c.source for c in all_chunks})
    print(f"Upserted {upserted} points; collection count={count.count}")
    print(f"Distinct sources ({len(sources)}): {', '.join(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
