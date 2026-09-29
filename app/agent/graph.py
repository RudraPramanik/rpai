"""Single portfolio agent: route → execute_tools → synthesize."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.router import (
    ROUTER_SYSTEM,
    AgentRouteError,
    InvalidPlanError,
    build_router_user_prompt,
    normalize_plan,
    parse_router_output,
)
from app.agent.state import AgentState
from app.agent.tools import get_profile, get_project, get_projects, search_knowledge
from app.config import Settings
from app.providers.base import LLMProvider
from app.schemas.chat import ChatHistoryMessage, ChatSource
from app.services.content import ContentStore

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """You are the portfolio assistant for this site.
Answer ONLY using the TOOL RESULTS in the user message.
If a field is null, missing, or a tool result says found=false, say that information is not documented.
Do not invent projects, employers, technologies, education, clients, years of experience, or personal facts.
When knowledge-search chunks are present, cite their labels (for example [1]) when helpful.
If no tool result supports the question, say you do not have enough documented information.
Keep answers concise and professional."""


def _history_dicts(
    history: list[ChatHistoryMessage] | None,
) -> list[dict[str, str]]:
    if not history:
        return []
    return [{"role": message.role, "content": message.content} for message in history]


def _catalog(store: ContentStore) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    for project in store.get_projects():
        slug = project.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        title = project.get("title")
        catalog.append(
            {"slug": slug, "title": str(title) if title is not None else slug}
        )
    return catalog


def build_synthesis_prompt(
    query: str,
    history: list[dict[str, str]] | None,
    tool_results: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    if history:
        parts.append("Prior conversation (for continuity only; prefer TOOL RESULTS for facts):")
        for message in history:
            parts.append(f"- {message['role']}: {message['content']}")
        parts.append("")
    parts.append(f"Question: {query}")
    parts.append(
        "TOOL RESULTS (null, missing, or found=false means that fact is not documented):"
    )
    parts.append(json.dumps(tool_results, ensure_ascii=False, indent=2))
    return "\n".join(parts)


class PortfolioAgent:
    """Compiled three-node graph. Streaming reuses the same route and tool nodes."""

    def __init__(
        self,
        llm: LLMProvider,
        settings: Settings,
        store: ContentStore,
        *,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.store = store
        self.top_k = top_k
        self.graph = self._compile()

    def _compile(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self.route)
        graph.add_node("execute_tools", self.execute_tools)
        graph.add_node("synthesize", self.synthesize)
        graph.add_edge(START, "route")
        graph.add_edge("route", "execute_tools")
        graph.add_edge("execute_tools", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    def route(self, state: AgentState) -> dict[str, Any]:
        catalog = _catalog(self.store)
        prompt = build_router_user_prompt(
            state["query"],
            state.get("history") or [],
            catalog,
        )
        last_error: Exception | None = None
        question_class = ""
        tool_plan: list[dict[str, Any]] = []
        for _attempt in range(2):
            raw = self.llm.complete(prompt, system=ROUTER_SYSTEM)
            try:
                parsed = parse_router_output(raw)
                question_class, tool_plan = normalize_plan(
                    parsed,
                    user_query=state["query"],
                )
                last_error = None
                break
            except InvalidPlanError as exc:
                last_error = exc
        if last_error is not None:
            raise AgentRouteError(
                f"Router returned an invalid plan: {last_error}"
            ) from last_error
        logger.info("agent route question_class=%s", question_class)
        return {"question_class": question_class, "tool_plan": tool_plan}

    def execute_tools(self, state: AgentState) -> dict[str, Any]:
        from app.services.chat import dedupe_sources

        tool_results: list[dict[str, Any]] = []
        hits = []
        label = 1
        for call in state.get("tool_plan") or []:
            name = call.get("name")
            if name == "get_profile":
                args: dict[str, Any] = {}
                payload: Any = get_profile(self.store)
            elif name == "get_projects":
                args = {}
                payload = get_projects(self.store)
            elif name == "get_project":
                slug = str(call.get("slug") or "")
                args = {"slug": slug}
                payload = get_project(self.store, slug)
            elif name == "search_knowledge":
                project = call.get("project")
                filters = {"project": project} if project else None
                query = str(call.get("query") or state.get("query") or "")
                args = {"query": query, "filters": filters}
                found = search_knowledge(
                    query,
                    filters,
                    top_k=self.top_k,
                    settings=self.settings,
                )
                hits.extend(found["results"])
                chunks = []
                for item in found["results"]:
                    chunks.append(
                        {
                            "label": f"[{label}]",
                            "text": item.text,
                            "source": item.metadata.get("source"),
                            "section": item.metadata.get("section"),
                            "project": item.metadata.get("project"),
                        }
                    )
                    label += 1
                payload = {"chunks": chunks}
            else:
                continue
            logger.info("agent tool name=%s args=%s", name, args)
            tool_results.append({"tool": name, "arguments": args, "result": payload})

        return {
            "tool_results": tool_results,
            "sources": dedupe_sources(hits),
        }

    def synthesize(self, state: AgentState) -> dict[str, str]:
        prompt = build_synthesis_prompt(
            state.get("query") or "",
            state.get("history") or [],
            state.get("tool_results") or [],
        )
        answer = self.llm.complete(prompt, system=SYNTHESIS_SYSTEM).strip()
        if not answer:
            from app.services.chat import INSUFFICIENT_EVIDENCE_ANSWER

            answer = INSUFFICIENT_EVIDENCE_ANSWER
        return {"answer": answer}

    def _initial(self, query: str, history: list[ChatHistoryMessage] | None) -> AgentState:
        return {"query": query, "history": _history_dicts(history)}

    def run(
        self,
        query: str,
        history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[str, list[ChatSource]]:
        final = self.graph.invoke(self._initial(query, history))
        sources = list(final.get("sources") or [])
        answer = str(final.get("answer") or "")
        return answer, sources

    def stream(
        self,
        query: str,
        history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[list[ChatSource], Iterator[str]]:
        """
        Run route and tools to completion, then stream synthesis tokens.

        The compiled graph still owns synthesize for non-streaming invoke.
        Streaming calls the same route and execute_tools nodes, then
        stream_complete with the same synthesis prompt, so tokens leave
        before the HTTP handler emits sources and done.
        """
        state: AgentState = dict(self._initial(query, history))
        state.update(self.route(state))
        state.update(self.execute_tools(state))
        sources = list(state.get("sources") or [])
        prompt = build_synthesis_prompt(
            state.get("query") or "",
            state.get("history") or [],
            state.get("tool_results") or [],
        )

        def tokens() -> Iterator[str]:
            yielded = False
            for delta in self.llm.stream_complete(prompt, system=SYNTHESIS_SYSTEM):
                if delta:
                    yielded = True
                    yield delta
            if not yielded:
                from app.services.chat import INSUFFICIENT_EVIDENCE_ANSWER

                yield INSUFFICIENT_EVIDENCE_ANSWER

        return sources, tokens()
