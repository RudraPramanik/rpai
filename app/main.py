"""FastAPI entrypoint for the portfolio read API (Phase 2)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.portfolio import router as portfolio_router
from app.config import get_settings
from app.services.content import ContentLoadError, ContentStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    store = ContentStore(settings.data_dir)
    try:
        store.load_all()
    except ContentLoadError as exc:
        raise RuntimeError(f"Failed to load portfolio content: {exc}") from exc
    app.state.content_store = store
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Portfolio API",
        description="Read-only portfolio facts from Phase-1 JSON content.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(portfolio_router)
    return app


app = create_app()
