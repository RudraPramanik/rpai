"""State carried through the portfolio agent graph."""

from __future__ import annotations

from typing import Any, TypedDict

from app.schemas.chat import ChatSource


class ToolCall(TypedDict, total=False):
    name: str
    slug: str
    query: str
    project: str


class AgentState(TypedDict, total=False):
    query: str
    history: list[dict[str, str]]
    question_class: str
    tool_plan: list[ToolCall]
    tool_results: list[dict[str, Any]]
    sources: list[ChatSource]
    answer: str
