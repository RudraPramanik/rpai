"""Chat service: portfolio agent graph, then complete/stream the synthesis."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.agent.graph import PortfolioAgent
from app.config import Settings, get_settings
from app.providers.base import LLMProvider
from app.providers.factory import get_llm_provider
from app.providers.litellm_provider import ProviderConfigError
from app.rag.retrieve import RetrievalResult
from app.schemas.chat import ChatHistoryMessage, ChatSource
from app.services.content import ContentStore

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough documented information in the portfolio knowledge base "
    "to answer that. Please ask about documented projects, experience, or profile topics."
)

SYSTEM_PROMPT = """You are the portfolio assistant for this site.
Answer ONLY using the CONTEXT blocks provided by the user message.
If the context is missing, empty, or insufficient, say you do not have enough documented information.
Do not invent projects, employers, technologies, education, clients, or personal facts.
When you use a fact from context, cite the source label (e.g. [1], [2]) inline when helpful.
Keep answers concise and professional."""


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[ChatSource]


def _source_title(metadata: dict) -> str:
    section = metadata.get("section")
    source = str(metadata.get("source") or "")
    if section:
        return str(section)
    if source:
        return Path(source).stem.replace("-", " ") or source
    return "knowledge"


def dedupe_sources(results: list[RetrievalResult]) -> list[ChatSource]:
    """Dedupe by (source, section); preserve retrieve order."""
    seen: set[tuple[str, str | None]] = set()
    sources: list[ChatSource] = []
    for item in results:
        source = str(item.metadata.get("source") or "")
        section_raw = item.metadata.get("section")
        section = str(section_raw) if section_raw is not None else None
        key = (source, section)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            ChatSource(
                title=_source_title(item.metadata),
                source=source,
                section=section,
            )
        )
    return sources


def _format_history(history: list[ChatHistoryMessage] | None) -> str:
    if not history:
        return ""
    lines = ["Prior conversation (for continuity only; prefer CONTEXT for facts):"]
    for msg in history:
        lines.append(f"- {msg.role}: {msg.content}")
    return "\n".join(lines) + "\n\n"


def build_user_prompt(
    query: str,
    results: list[RetrievalResult],
    history: list[ChatHistoryMessage] | None = None,
) -> str:
    parts: list[str] = []
    hist = _format_history(history)
    if hist:
        parts.append(hist)
    parts.append(f"Question: {query}\n")
    if not results:
        parts.append("CONTEXT:\n(none)\n")
        return "\n".join(parts)

    parts.append("CONTEXT:")
    for index, item in enumerate(results, start=1):
        source = item.metadata.get("source") or "unknown"
        section = item.metadata.get("section") or ""
        project = item.metadata.get("project") or ""
        header = f"[{index}] source={source}"
        if section:
            header += f" section={section}"
        if project:
            header += f" project={project}"
        parts.append(header)
        parts.append(item.text)
        parts.append("")
    return "\n".join(parts)


def _content_store(settings: Settings) -> ContentStore:
    store = ContentStore(settings.data_dir)
    store.load_all()
    return store


def _agent(
    *,
    settings: Settings | None,
    llm: LLMProvider | None,
    top_k: int,
) -> PortfolioAgent:
    cfg = settings or get_settings()
    provider = llm or get_llm_provider(cfg)
    return PortfolioAgent(provider, cfg, _content_store(cfg), top_k=top_k)


def run_chat(
    query: str,
    *,
    history: list[ChatHistoryMessage] | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
) -> ChatResult:
    """Non-streaming chat: agent graph (route → tools → synthesize)."""
    answer, sources = _agent(settings=settings, llm=llm, top_k=top_k).run(
        query,
        history,
    )
    return ChatResult(answer=answer, sources=sources)


def stream_chat(
    query: str,
    *,
    history: list[ChatHistoryMessage] | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
) -> tuple[list[ChatSource], Iterator[str]]:
    """
    Streaming chat.

    Returns (sources, token_iterator). Routing and tools finish first; the
    iterator yields synthesis tokens only.
    """
    return _agent(settings=settings, llm=llm, top_k=top_k).stream(query, history)


__all__ = [
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "SYSTEM_PROMPT",
    "ChatResult",
    "ProviderConfigError",
    "build_user_prompt",
    "dedupe_sources",
    "run_chat",
    "stream_chat",
]
