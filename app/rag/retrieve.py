"""Semantic retrieve over the portfolio knowledge collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qdrant_client.http import models as qmodels

from app.config import Settings, get_settings
from app.providers.base import EmbeddingProvider
from app.providers.factory import get_embedding_provider
from app.rag.qdrant_store import get_qdrant_client


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    metadata: dict[str, Any]
    score: float


def _build_filter(filters: Mapping[str, Any] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    must: list[qmodels.FieldCondition] = []
    for key, value in filters.items():
        if value is None:
            continue
        must.append(
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
        )
    if not must:
        return None
    return qmodels.Filter(must=must)


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Mapping[str, Any] | None = None,
    *,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    """
    Embed the query with the same provider used at ingest and search Qdrant.

    filters: optional payload filters, e.g. {"project": "oms"}.
    """
    cfg = settings or get_settings()
    provider = embedding_provider or get_embedding_provider(cfg)
    client = get_qdrant_client(cfg)

    vectors = provider.embed([query])
    if not vectors or not vectors[0]:
        return []

    query_filter = _build_filter(filters)
    response = client.query_points(
        collection_name=cfg.qdrant_collection,
        query=vectors[0],
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    hits = response.points if hasattr(response, "points") else response

    results: list[RetrievalResult] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        text = str(payload.pop("text", ""))
        results.append(
            RetrievalResult(
                text=text,
                metadata=payload,
                score=float(hit.score) if hit.score is not None else 0.0,
            )
        )
    return results
