"""Unit tests for Markdown chunking (no Qdrant / Gemini required)."""

from __future__ import annotations

from pathlib import Path

from app.rag.chunking import (
    chunk_markdown_file,
    chunk_markdown_text,
    estimate_tokens,
)

_BACKEND = Path(__file__).resolve().parents[1]
_KNOWLEDGE = _BACKEND / "data" / "knowledge"


def test_short_section_one_chunk() -> None:
    md = "## Overview\n\nShort body only.\n"
    chunks = chunk_markdown_text(
        md,
        source="profile.md",
        category="profile",
        project=None,
    )
    assert len(chunks) == 1
    assert chunks[0].section == "Overview"
    assert "Doc: profile" in chunks[0].text
    assert "Section: Overview" in chunks[0].text
    assert "Short body only." in chunks[0].text


def test_long_section_subsplit_with_overlap() -> None:
    # Force tiny max so a medium paragraph splits
    words = " ".join(f"word{i}" for i in range(200))
    md = f"## Architecture\n\n{words}\n"
    chunks = chunk_markdown_text(
        md,
        source="projects/demo.md",
        category="project",
        project="demo",
        max_tokens=40,
        overlap_tokens=10,
    )
    assert len(chunks) > 1
    assert all(c.section == "Architecture" for c in chunks)
    assert all("Project: demo" in c.text for c in chunks)
    # Overlap: consecutive chunks share some words
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_oms_file_project_metadata() -> None:
    path = _KNOWLEDGE / "projects" / "oms.md"
    assert path.is_file(), f"missing {path}"
    chunks = chunk_markdown_file(path, _KNOWLEDGE)
    assert chunks
    assert all(c.project == "oms" for c in chunks)
    assert all(c.category == "project" for c in chunks)
    assert all(c.source == "projects/oms.md" for c in chunks)
    sections = {c.section for c in chunks}
    assert "Overview" in sections or "Architecture" in sections


def test_estimate_tokens_nonempty() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") >= 1
