"""Markdown-aware chunking with parent context and metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ~600–700 tokens target; ~50–100 overlap (word-based estimator ≈ 0.75 words/token)
DEFAULT_MAX_TOKENS = 650
DEFAULT_OVERLAP_TOKENS = 75

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One embeddable knowledge chunk with metadata."""

    text: str
    source: str
    category: str
    project: str | None
    section: str
    chunk_index: int


def estimate_tokens(text: str) -> int:
    """Lightweight token estimate (~4 chars/token, floor of 1 for non-empty)."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _split_with_overlap(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into overlapping windows by approximate token size."""
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []

    words = text.split()
    if not words:
        return []

    # Approximate words per window from chars/token heuristic
    avg_word_len = max(1, sum(len(w) for w in words) // len(words))
    words_per_chunk = max(1, (max_tokens * 4) // (avg_word_len + 1))
    overlap_words = max(1, (overlap_tokens * 4) // (avg_word_len + 1))

    parts: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        piece = " ".join(words[start:end]).strip()
        if piece:
            parts.append(piece)
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
        if start >= end:
            start = end
    return parts


def derive_path_metadata(path: Path, knowledge_root: Path) -> tuple[str, str, str | None]:
    """
    Derive (source, category, project) from a knowledge file path.

    source is relative to knowledge_root using forward slashes.
    """
    rel = path.resolve().relative_to(knowledge_root.resolve())
    source = rel.as_posix()
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "projects":
        slug = Path(parts[-1]).stem
        return source, "project", slug
    stem = path.stem
    if stem in {"profile", "experience", "education", "courses"}:
        return source, stem, None
    return source, "other", None


def _parent_context(project: str | None, category: str, section: str) -> str:
    if project:
        header = f"Project: {project}"
    else:
        header = f"Doc: {category}"
    return f"{header}\nSection: {section}\n\n"


def chunk_markdown_text(
    markdown: str,
    *,
    source: str,
    category: str,
    project: str | None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk Markdown by `##` headings, then sub-split oversized sections."""
    matches = list(_HEADING_RE.finditer(markdown))
    sections: list[tuple[str, str]] = []

    if not matches:
        body = markdown.strip()
        if body:
            sections.append(("Overview", body))
    else:
        # Content before first ## heading (rare) → Overview
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.append(("Overview", preamble))
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            body = markdown[start:end].strip()
            sections.append((title, body))

    chunks: list[Chunk] = []
    index = 0
    for section_title, body in sections:
        if not body:
            continue
        pieces = _split_with_overlap(
            body,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for piece in pieces:
            embed_text = _parent_context(project, category, section_title) + piece
            chunks.append(
                Chunk(
                    text=embed_text,
                    source=source,
                    category=category,
                    project=project,
                    section=section_title,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks


def chunk_markdown_file(
    path: Path,
    knowledge_root: Path,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Read a Markdown file and return chunks with path-derived metadata."""
    source, category, project = derive_path_metadata(path, knowledge_root)
    text = path.read_text(encoding="utf-8")
    return chunk_markdown_text(
        text,
        source=source,
        category=category,
        project=project,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
