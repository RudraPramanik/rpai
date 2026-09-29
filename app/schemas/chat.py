"""Chat request/response schemas for POST /api/chat."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_QUERY_CHARS = 2000
MAX_HISTORY_TURNS = 10
MAX_HISTORY_CONTENT_CHARS = 1000


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_HISTORY_CONTENT_CHARS)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    history: list[ChatHistoryMessage] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_be_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped

    @field_validator("history")
    @classmethod
    def cap_history(cls, value: list[ChatHistoryMessage]) -> list[ChatHistoryMessage]:
        if len(value) > MAX_HISTORY_TURNS:
            return value[-MAX_HISTORY_TURNS:]
        return value


class ChatSource(BaseModel):
    title: str
    source: str
    section: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
