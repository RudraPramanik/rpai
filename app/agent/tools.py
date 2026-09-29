"""Portfolio tools. Each function is callable without the agent graph."""

from __future__ import annotations

from typing import Any, Mapping

from app.config import Settings
from app.rag.retrieve import RetrievalResult, retrieve
from app.services.content import ContentStore


def get_profile(store: ContentStore) -> dict[str, Any]:
    """Return stored profile facts, including null fields."""
    return dict(store.get_profile())


def get_projects(store: ContentStore) -> list[dict[str, Any]]:
    """Return the canonical project list."""
    return store.get_projects()


def get_project(store: ContentStore, slug: str) -> dict[str, Any]:
    """Return one project, or a not-found payload that does not substitute another."""
    project = store.get_project(slug)
    if project is None:
        return {"found": False, "slug": slug}
    return {"found": True, "slug": slug, "project": dict(project)}


def search_knowledge(
    query: str,
    filters: Mapping[str, Any] | None = None,
    *,
    top_k: int = 5,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Semantic search over portfolio knowledge.

    Returns retrieval hits plus citation sources deduped the same way as chat.
    """
    results: list[RetrievalResult] = retrieve(
        query,
        top_k=top_k,
        filters=filters,
        settings=settings,
    )
    # Lazy import: chat imports the agent package, so a module-level import cycles.
    from app.services.chat import dedupe_sources

    return {
        "results": results,
        "sources": dedupe_sources(results),
    }
