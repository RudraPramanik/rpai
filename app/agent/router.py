"""Router prompt, JSON plan parsing, and class guards."""

from __future__ import annotations

import json
from typing import Any

from app.agent.state import ToolCall

ROUTER_SYSTEM = """You route portfolio questions to tools. Reply with one JSON object and nothing else.
Schema:
{"question_class": "profile" | "project" | "comparative", "slugs": ["slug"], "query": "short search query"}

Classes:
- profile: facts about the person (name, headline, location, email, links, tech stack, years of experience). slugs must be empty.
- project: exactly one portfolio project. Put its catalog slug in slugs.
- comparative: two or more projects (compare, contrast, or ask about several). Put each catalog slug in slugs.

Use only slugs from the catalog in the user message.

Examples:
Question: How many years of experience does Rudra have?
{"question_class": "profile", "slugs": [], "query": "years of experience"}

Question: What is Rudra's tech stack?
{"question_class": "profile", "slugs": [], "query": "tech stack"}

Question: How does the OMS mobile app work?
{"question_class": "project", "slugs": ["oms"], "query": "How does the OMS mobile app work?"}

Question: Compare OMS and Laddu
{"question_class": "comparative", "slugs": ["oms", "laddu"], "query": "Compare OMS and Laddu"}
"""


class InvalidPlanError(ValueError):
    """Router output could not be parsed or normalized."""


class AgentRouteError(RuntimeError):
    """Router failed to produce a valid plan after one retry."""


def build_router_user_prompt(
    query: str,
    history: list[dict[str, str]] | None,
    catalog: list[dict[str, str]],
) -> str:
    lines = ["Project catalog (use these slugs only):"]
    for item in catalog:
        lines.append(f"- {item['slug']}: {item['title']}")
    if history:
        lines.append("Prior conversation:")
        for message in history:
            lines.append(f"- {message['role']}: {message['content']}")
    lines.append(f"Question: {query}")
    lines.append("JSON:")
    return "\n".join(lines)


def parse_router_output(text: str) -> dict[str, Any]:
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise InvalidPlanError("router output has no JSON object")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InvalidPlanError("router JSON is invalid") from exc
    if not isinstance(data, dict):
        raise InvalidPlanError("router JSON must be an object")
    return data


def _collect_slugs(raw: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str):
            slug = value.strip()
            if slug and slug not in found:
                found.append(slug)

    for key in ("slugs", "projects"):
        value = raw.get(key)
        if isinstance(value, list):
            for item in value:
                add(item)
    add(raw.get("slug"))
    tools = raw.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                add(tool.get("slug") or tool.get("project"))
    return found


def normalize_plan(raw: dict[str, Any], *, user_query: str) -> tuple[str, list[ToolCall]]:
    """
    Force tool plans to match the question class.

    Profile plans drop knowledge search. Project plans are one lookup plus a
    filtered search. Comparative plans cover at least two slugs.
    """
    if not isinstance(raw, dict):
        raise InvalidPlanError("plan must be an object")
    question_class = raw.get("question_class")
    if question_class not in {"profile", "project", "comparative"}:
        raise InvalidPlanError("unknown question_class")

    if question_class == "profile":
        return "profile", [{"name": "get_profile"}]

    slugs = _collect_slugs(raw)
    search_query = user_query.strip() or str(raw.get("query") or "").strip()
    if question_class == "project":
        if not slugs:
            raise InvalidPlanError("project plan missing slug")
        slug = slugs[0]
        return "project", [
            {"name": "get_project", "slug": slug},
            {"name": "search_knowledge", "query": search_query, "project": slug},
        ]

    if len(slugs) < 2:
        raise InvalidPlanError("comparative plan needs two slugs")
    tools: list[ToolCall] = []
    for slug in slugs[:4]:
        tools.append({"name": "get_project", "slug": slug})
        tools.append(
            {"name": "search_knowledge", "query": search_query, "project": slug}
        )
    return "comparative", tools
