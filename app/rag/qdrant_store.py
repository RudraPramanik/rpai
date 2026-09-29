"""Qdrant collection helpers and deterministic upserts."""

from __future__ import annotations

import uuid
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings, get_settings
from app.providers.base import EmbeddingProvider
from app.rag.chunking import Chunk

# Stable namespace for deterministic point IDs
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def point_id_for(source: str, chunk_index: int) -> str:
    """Deterministic UUID string from source + chunk index."""
    return str(uuid.uuid5(_POINT_NS, f"{source}:{chunk_index}"))


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    cfg = settings or get_settings()
    kwargs: dict = {"url": cfg.qdrant_url}
    if cfg.qdrant_api_key:
        kwargs["api_key"] = cfg.qdrant_api_key
    return QdrantClient(**kwargs)


def resolve_vector_size(
    settings: Settings,
    embedding_provider: EmbeddingProvider,
) -> int:
    """Use configured dimensions or probe with a one-shot embed."""
    if settings.embedding_dimensions is not None:
        return int(settings.embedding_dimensions)
    vectors = embedding_provider.embed(["dimension probe"])
    if not vectors or not vectors[0]:
        raise RuntimeError("Failed to probe embedding dimensions")
    return len(vectors[0])


def ensure_collection(
    client: QdrantClient,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
) -> int:
    """
    Ensure the portfolio knowledge collection exists with the expected vector size.
    Returns the vector size used.
    """
    name = settings.qdrant_collection
    vector_size = resolve_vector_size(settings, embedding_provider)
    exists = client.collection_exists(name)
    if exists:
        info = client.get_collection(name)
        # qdrant-client exposes vectors config differently by version
        vectors_cfg = info.config.params.vectors
        if isinstance(vectors_cfg, qmodels.VectorParams):
            current_size = vectors_cfg.size
        elif isinstance(vectors_cfg, dict):
            # named vectors — take first
            first = next(iter(vectors_cfg.values()))
            current_size = first.size
        else:
            current_size = getattr(vectors_cfg, "size", None)
        if current_size is not None and int(current_size) != vector_size:
            raise RuntimeError(
                f"Collection {name!r} has vector size {current_size}, but "
                f"configured/probed size is {vector_size}. Recreate the collection "
                f"and re-ingest after changing EMBEDDING_MODEL / EMBEDDING_DIMENSIONS."
            )
        return vector_size

    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )
    return vector_size


def upsert_chunks(
    client: QdrantClient,
    settings: Settings,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    embedding_model: str,
) -> int:
    """Upsert chunks with deterministic IDs. Returns number of points upserted."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    if not chunks:
        return 0

    points: list[qmodels.PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        payload = {
            "text": chunk.text,
            "source": chunk.source,
            "category": chunk.category,
            "project": chunk.project,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "embedding_model": embedding_model,
        }
        points.append(
            qmodels.PointStruct(
                id=point_id_for(chunk.source, chunk.chunk_index),
                vector=list(vector),
                payload=payload,
            )
        )

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)
