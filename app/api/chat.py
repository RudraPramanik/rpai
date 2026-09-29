"""POST /api/chat — JSON and SSE modes."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from qdrant_client.http.exceptions import UnexpectedResponse

from app.providers.litellm_provider import ProviderConfigError
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.chat import run_chat, stream_chat

router = APIRouter(tags=["chat"])


def _wants_stream(stream: bool | None, accept: str | None) -> bool:
    if stream is True:
        return True
    if stream is False:
        return False
    if accept and "text/event-stream" in accept.lower():
        return True
    return False


def _sse_event(event: str, data: str) -> str:
    # SSE: each field on its own line; blank line terminates the event.
    safe = data.replace("\r\n", "\n").replace("\r", "\n")
    lines = safe.split("\n")
    payload = "\n".join(f"data: {line}" for line in lines) if lines else "data: "
    return f"event: {event}\n{payload}\n\n"


def _sources_json(sources: list[ChatSource]) -> str:
    return json.dumps([s.model_dump() for s in sources], ensure_ascii=False)


def _map_dependency_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ProviderConfigError):
        return HTTPException(status_code=503, detail=str(exc))
    detail = str(exc) or exc.__class__.__name__
    return HTTPException(
        status_code=503,
        detail=f"Chat backend unavailable: {detail}",
    )


def _sse_stream(
    body: ChatRequest,
) -> Iterator[str]:
    try:
        sources, tokens = stream_chat(body.query, history=body.history or None)
        for token in tokens:
            yield _sse_event("token", token)
        yield _sse_event("sources", _sources_json(sources))
        yield _sse_event("done", "")
    except ProviderConfigError as exc:
        yield _sse_event("error", str(exc))
    except (UnexpectedResponse, ConnectionError, TimeoutError, OSError) as exc:
        yield _sse_event("error", f"Chat backend unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 — surface as SSE error for clients
        yield _sse_event("error", f"Chat failed: {exc}")


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        200: {
            "description": "JSON answer+sources, or SSE stream when stream=true / Accept: text/event-stream",
        },
        422: {"description": "Invalid or blank query"},
        503: {"description": "Missing credentials or vector store unavailable"},
    },
)
def chat(
    body: ChatRequest,
    stream: bool | None = Query(
        default=None,
        description="Force SSE when true; force JSON when false. "
        "If omitted, Accept: text/event-stream selects SSE.",
    ),
    accept: str | None = Header(default=None, alias="Accept"),
):
    """RAG chat: retrieve knowledge, then complete via the configured LLM provider."""
    if _wants_stream(stream, accept):
        return StreamingResponse(
            _sse_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = run_chat(body.query, history=body.history or None)
    except ProviderConfigError as exc:
        raise _map_dependency_error(exc) from exc
    except (UnexpectedResponse, ConnectionError, TimeoutError, OSError) as exc:
        raise _map_dependency_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise _map_dependency_error(exc) from exc

    return ChatResponse(answer=result.answer, sources=result.sources)
