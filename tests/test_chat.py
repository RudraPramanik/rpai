"""Unit tests for chat prompt assembly and short-circuit (no live LLM/Qdrant)."""

from __future__ import annotations

from app.rag.retrieve import RetrievalResult
from app.schemas.chat import ChatHistoryMessage, ChatRequest
from app.services.chat import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
    dedupe_sources,
    run_chat,
)


class _FakeLLM:
    model_id = "fake"

    def __init__(self) -> None:
        self.called = False

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.called = True
        return "fake-answer"

    def stream_complete(self, prompt: str, *, system: str | None = None):
        self.called = True
        yield "fake"


def test_chat_request_rejects_blank_query() -> None:
    try:
        ChatRequest(query="   ")
        raised = False
    except Exception:
        raised = True
    assert raised


def test_dedupe_sources_by_source_section() -> None:
    results = [
        RetrievalResult(
            text="a",
            metadata={"source": "knowledge/projects/oms.md", "section": "Overview"},
            score=0.9,
        ),
        RetrievalResult(
            text="b",
            metadata={"source": "knowledge/projects/oms.md", "section": "Overview"},
            score=0.8,
        ),
        RetrievalResult(
            text="c",
            metadata={"source": "knowledge/projects/oms.md", "section": "Architecture"},
            score=0.7,
        ),
    ]
    sources = dedupe_sources(results)
    assert len(sources) == 2
    assert sources[0].section == "Overview"
    assert sources[1].section == "Architecture"


def test_build_user_prompt_includes_context_and_history() -> None:
    results = [
        RetrievalResult(
            text="OMS is a mobile office app.",
            metadata={
                "source": "knowledge/projects/oms.md",
                "section": "Overview",
                "project": "oms",
            },
            score=0.9,
        )
    ]
    prompt = build_user_prompt(
        "How does OMS work?",
        results,
        history=[ChatHistoryMessage(role="user", content="hi")],
    )
    assert "Question: How does OMS work?" in prompt
    assert "[1]" in prompt
    assert "OMS is a mobile office app." in prompt
    assert "Prior conversation" in prompt


def test_system_prompt_grounding_rules() -> None:
    assert "ONLY" in SYSTEM_PROMPT.upper() or "only" in SYSTEM_PROMPT
    assert "invent" in SYSTEM_PROMPT.lower()


def test_run_chat_short_circuits_without_llm(monkeypatch) -> None:
    fake = _FakeLLM()

    def _empty_retrieve(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.services.chat.retrieve", _empty_retrieve)
    result = run_chat("What is Rudra's favorite food?", llm=fake)
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []
    assert fake.called is False
