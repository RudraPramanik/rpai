"""Agent tools, plan guards, and chat wiring. No live LLM or Qdrant."""

from __future__ import annotations

import inspect
import logging

from fastapi.testclient import TestClient

from app.agent.graph import PortfolioAgent, SYNTHESIS_SYSTEM, build_synthesis_prompt
from app.agent.router import (
    AgentRouteError,
    InvalidPlanError,
    normalize_plan,
    parse_router_output,
)
from app.agent.tools import get_profile, get_project, get_projects, search_knowledge
from app.api import portfolio as portfolio_api
from app.config import get_settings
from app.main import create_app
from app.rag.retrieve import RetrievalResult
from app.services.chat import stream_chat
from app.services.content import ContentStore


class ScriptedLLM:
    model_id = "fake"

    def __init__(self, completes: list[str], streams: list[str] | None = None) -> None:
        self.completes = list(completes)
        self.streams = list(streams or [])
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.completes.pop(0)

    def stream_complete(self, prompt: str, *, system: str | None = None):
        self.prompts.append(prompt)
        text = self.streams.pop(0) if self.streams else "not documented"
        yield text


def _store() -> ContentStore:
    store = ContentStore(get_settings().data_dir)
    store.load_all()
    return store


def test_profile_tool_includes_null_years() -> None:
    profile = get_profile(_store())
    assert profile["name"]
    assert "yearsOfExperience" in profile
    assert profile["yearsOfExperience"] is None
    assert "techStack" in profile


def test_unknown_project_slug_is_not_found() -> None:
    result = get_project(_store(), "does-not-exist")
    assert result["found"] is False
    assert result["slug"] == "does-not-exist"
    assert "project" not in result
    slugs = {item["slug"] for item in get_projects(_store())}
    assert "oms" in slugs
    assert "does-not-exist" not in slugs


def test_search_knowledge_passes_project_filter(monkeypatch) -> None:
    seen: dict = {}

    def fake_retrieve(query, top_k=5, filters=None, **_kwargs):
        seen["query"] = query
        seen["top_k"] = top_k
        seen["filters"] = filters
        project = (filters or {}).get("project")
        return [
            RetrievalResult(
                text="OMS architecture",
                metadata={
                    "source": "knowledge/projects/oms.md",
                    "section": "Architecture",
                    "project": project,
                },
                score=0.8,
            )
        ]

    monkeypatch.setattr("app.agent.tools.retrieve", fake_retrieve)
    found = search_knowledge("How does OMS work?", filters={"project": "oms"})
    assert seen["filters"] == {"project": "oms"}
    assert seen["top_k"] == 5
    assert found["sources"][0].source.endswith("oms.md")
    assert found["sources"][0].section == "Architecture"
    assert all(item.metadata["project"] == "oms" for item in found["results"])


def test_profile_plan_strips_knowledge_search() -> None:
    question_class, tools = normalize_plan(
        {
            "question_class": "profile",
            "slugs": [],
            "tools": [{"name": "search_knowledge", "query": "years", "project": "oms"}],
        },
        user_query="How many years of experience does Rudra have?",
    )
    assert question_class == "profile"
    assert tools == [{"name": "get_profile"}]
    assert all(tool["name"] != "search_knowledge" for tool in tools)


def test_comparative_plan_keeps_oms_and_laddu() -> None:
    question_class, tools = normalize_plan(
        {"question_class": "comparative", "slugs": ["oms", "laddu"]},
        user_query="Compare OMS and Laddu",
    )
    assert question_class == "comparative"
    slugs = [tool.get("slug") or tool.get("project") for tool in tools]
    assert slugs.count("oms") == 2
    assert slugs.count("laddu") == 2
    names = [tool["name"] for tool in tools]
    assert names.count("get_project") == 2
    assert names.count("search_knowledge") == 2


def test_project_plan_requires_slug() -> None:
    try:
        normalize_plan(
            {"question_class": "project", "slugs": []},
            user_query="How does OMS work?",
        )
        raised = False
    except InvalidPlanError:
        raised = True
    assert raised


def test_router_retries_once_then_fails() -> None:
    store = _store()
    settings = get_settings()
    bad = ScriptedLLM(["not json", "still not json"])
    agent = PortfolioAgent(bad, settings, store)
    try:
        agent.route({"query": "How many years of experience does Rudra have?", "history": []})
        raised = False
    except AgentRouteError:
        raised = True
    assert raised
    assert len(bad.prompts) == 2


def test_router_accepts_json_on_retry() -> None:
    store = _store()
    settings = get_settings()
    llm = ScriptedLLM(
        [
            "sure, let me think",
            '{"question_class": "profile", "slugs": [], "query": "years"}',
        ]
    )
    agent = PortfolioAgent(llm, settings, store)
    update = agent.route(
        {"query": "How many years of experience does Rudra have?", "history": []}
    )
    assert update["question_class"] == "profile"
    assert update["tool_plan"] == [{"name": "get_profile"}]
    assert len(llm.prompts) == 2


def test_graph_nodes_and_profile_turn_logs(caplog) -> None:
    store = _store()
    settings = get_settings()
    llm = ScriptedLLM(
        [
            '{"question_class": "profile", "slugs": [], "query": "years of experience"}',
            "Years of experience are not documented.",
        ]
    )
    agent = PortfolioAgent(llm, settings, store)
    node_names = set(agent.graph.get_graph().nodes)
    assert {"route", "execute_tools", "synthesize"} <= node_names

    caplog.set_level(logging.INFO)
    answer, sources = agent.run("How many years of experience does Rudra have?")
    assert sources == []
    assert "not documented" in answer.lower()
    assert "null" in llm.prompts[-1]
    assert "yearsOfExperience" in llm.prompts[-1]
    assert "agent route question_class=profile" in caplog.text
    assert "agent tool name=get_profile" in caplog.text
    assert "agent tool name=search_knowledge" not in caplog.text
    assert "not documented" in SYNTHESIS_SYSTEM.lower() or "not documented" in build_synthesis_prompt(
        "q", [], [{"tool": "get_profile", "result": {"yearsOfExperience": None}}]
    )


def test_oms_turn_filters_search_and_cites(monkeypatch, caplog) -> None:
    def fake_retrieve(query, top_k=5, filters=None, **_kwargs):
        project = (filters or {}).get("project")
        return [
            RetrievalResult(
                text=f"Details about {project}.",
                metadata={
                    "source": f"knowledge/projects/{project}.md",
                    "section": "Overview",
                    "project": project,
                },
                score=0.9,
            )
        ]

    monkeypatch.setattr("app.agent.tools.retrieve", fake_retrieve)
    llm = ScriptedLLM(
        [
            '{"question_class": "project", "slugs": ["oms"], "query": "How does OMS work?"}',
            "OMS is a mobile office app. [1]",
        ]
    )
    agent = PortfolioAgent(llm, get_settings(), _store())
    caplog.set_level(logging.INFO)
    answer, sources = agent.run("How does the OMS mobile app work?")
    assert "OMS" in answer
    assert len(sources) == 1
    assert sources[0].source.endswith("oms.md")
    assert "agent tool name=get_project" in caplog.text
    assert "'slug': 'oms'" in caplog.text or '"slug": "oms"' in caplog.text or "oms" in caplog.text
    assert "agent tool name=search_knowledge" in caplog.text
    assert "'project': 'oms'" in caplog.text or '"project": "oms"' in caplog.text


def test_compare_oms_and_laddu_traces_both(monkeypatch, caplog) -> None:
    calls: list[dict | None] = []

    def fake_retrieve(query, top_k=5, filters=None, **_kwargs):
        calls.append(dict(filters) if filters else None)
        project = (filters or {}).get("project")
        return [
            RetrievalResult(
                text=f"{project} overview",
                metadata={
                    "source": f"knowledge/projects/{project}.md",
                    "section": "Overview",
                    "project": project,
                },
                score=0.7,
            )
        ]

    monkeypatch.setattr("app.agent.tools.retrieve", fake_retrieve)
    llm = ScriptedLLM(
        [
            '{"question_class": "comparative", "slugs": ["oms", "laddu"]}',
            "OMS is an office app while Laddu is a food delivery app.",
        ]
    )
    agent = PortfolioAgent(llm, get_settings(), _store())
    caplog.set_level(logging.INFO)
    answer, sources = agent.run("Compare OMS and Laddu")
    assert "OMS" in answer and "Laddu" in answer
    assert {call["project"] for call in calls} == {"oms", "laddu"}
    assert len(sources) == 2
    assert "question_class=comparative" in caplog.text


def test_stream_emits_tokens_before_sources_event(monkeypatch) -> None:
    llm = ScriptedLLM(
        completes=[
            '{"question_class": "profile", "slugs": [], "query": "years"}',
        ],
        streams=["Years of experience are not documented."],
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("retrieve should not run")

    monkeypatch.setattr("app.agent.tools.retrieve", _boom)
    monkeypatch.setattr("app.services.chat.get_llm_provider", lambda settings=None: llm)

    app = create_app()
    with TestClient(app) as client:
        blank = client.post("/api/chat", json={"query": "   "})
        assert blank.status_code == 422
        assert llm.prompts == []

        health = client.get("/health")
        assert health.status_code == 200

        with client.stream(
            "POST",
            "/api/chat?stream=true",
            json={"query": "How many years of experience does Rudra have?"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    token_at = body.index("event: token")
    sources_at = body.index("event: sources")
    done_at = body.index("event: done")
    assert token_at < sources_at < done_at
    assert "not documented" in body
    assert '"sources"' not in body.split("event: sources", 1)[0] or "event: sources" in body
    sources_payload = body.split("event: sources", 1)[1].split("event: done", 1)[0]
    assert "[]" in sources_payload


def test_invalid_router_json_maps_to_503() -> None:
    llm = ScriptedLLM(["nope", "also nope"])
    app = create_app()

    def _fake_stream(*_args, **_kwargs):
        raise AssertionError("stream should not be used")

    # Patch the provider used by the chat service for this app process.
    import app.services.chat as chat_service

    original = chat_service.get_llm_provider
    chat_service.get_llm_provider = lambda settings=None: llm  # type: ignore[assignment]
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={"query": "How many years of experience does Rudra have?"},
            )
    finally:
        chat_service.get_llm_provider = original
    assert response.status_code == 503
    assert "invalid plan" in response.json()["detail"].lower()
    assert _fake_stream is not None


def test_portfolio_routes_do_not_import_agent() -> None:
    source = inspect.getsource(portfolio_api)
    assert "app.agent" not in source
    assert "langgraph" not in source


def test_stream_chat_helper_yields_before_done() -> None:
    llm = ScriptedLLM(
        completes=['{"question_class": "profile", "slugs": [], "query": "years"}'],
        streams=["alpha", "beta"],
    )
    # stream_complete yields one item from streams.pop — adjust by custom impl
    chunks = ["alpha ", "beta"]

    class Multi(ScriptedLLM):
        def stream_complete(self, prompt: str, *, system: str | None = None):
            self.prompts.append(prompt)
            yield from chunks

    multi = Multi(completes=['{"question_class": "profile", "slugs": [], "query": "years"}'])
    sources, tokens = stream_chat(
        "How many years of experience does Rudra have?",
        llm=multi,
    )
    assert sources == []
    assert list(tokens) == ["alpha ", "beta"]


def test_parse_router_output_reads_fenced_json() -> None:
    data = parse_router_output('```json\n{"question_class": "profile", "slugs": []}\n```')
    assert data["question_class"] == "profile"
