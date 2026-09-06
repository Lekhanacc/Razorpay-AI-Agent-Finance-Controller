"""Deterministic, heading-aware chunking for the knowledge corpus.

This module implements Phase 2's chunking strategy:
  - Split a markdown document into sections at its headings.
  - Keep each heading attached to the content that follows it.
  - Merge unusually small sections together so we don't emit tiny, low-signal chunks.
  - For sections longer than the configured chunk size, split into overlapping
    word-count windows (the heading is repeated on every window so a chunk is always
    self-describing, even mid-document).

No summarization or rewriting happens here -- only splitting/merging of existing text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Section:
    heading: str | None
    body: str


def _words(text: str) -> list[str]:
    return text.split()


def split_into_sections(markdown_text: str) -> list[Section]:
    """Split a markdown document into (heading, body) sections.

    Content appearing before the first heading (if any) is returned as a
    section with heading=None.
    """
    matches = list(_HEADING_PATTERN.finditer(markdown_text))
    if not matches:
        stripped = markdown_text.strip()
        return [Section(heading=None, body=stripped)] if stripped else []

    sections: list[Section] = []

    leading = markdown_text[: matches[0].start()].strip()
    if leading:
        sections.append(Section(heading=None, body=leading))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[body_start:body_end].strip()
        sections.append(Section(heading=heading_text, body=body))

    return sections


def _merge_small_sections(sections: list[Section], min_chunk_words: int, chunk_size_words: int) -> list[Section]:
    """Merge sections smaller than min_chunk_words into a running buffer.

    Prevents emitting near-empty chunks (e.g. a heading followed by one short
    sentence) as their own retrieval unit.
    """
    merged: list[Section] = []
    buffer_headings: list[str] = []
    buffer_bodies: list[str] = []
    buffer_words = 0

    def flush() -> None:
        nonlocal buffer_headings, buffer_bodies, buffer_words
        if buffer_bodies:
            merged.append(
                Section(
                    heading=" / ".join(buffer_headings) if buffer_headings else None,
                    body="\n\n".join(buffer_bodies),
                )
            )
        buffer_headings, buffer_bodies, buffer_words = [], [], 0

    for section in sections:
        word_count = len(_words(section.body))
        if word_count == 0:
            continue
        fits_in_buffer = word_count < min_chunk_words and buffer_words + word_count <= chunk_size_words
        if fits_in_buffer:
            if section.heading:
                buffer_headings.append(section.heading)
            buffer_bodies.append(f"{section.heading}\n{section.body}" if section.heading else section.body)
            buffer_words += word_count
            continue
        flush()
        merged.append(section)

    flush()
    return merged


def _chunk_section(section: Section, chunk_size_words: int, overlap_words: int) -> list[str]:
    prefix = f"{section.heading}\n" if section.heading else ""
    body_words = _words(section.body)
    if not body_words:
        return [prefix.strip()] if prefix.strip() else []

    if len(body_words) <= chunk_size_words:
        text = (prefix + section.body).strip()
        return [text] if text else []

    step = max(chunk_size_words - overlap_words, 1)
    chunks: list[str] = []
    start = 0
    while start < len(body_words):
        window = body_words[start : start + chunk_size_words]
        chunk_text = (prefix + " ".join(window)).strip()
        chunks.append(chunk_text)
        if start + chunk_size_words >= len(body_words):
            break
        start += step
    return chunks


def chunk_document(
    markdown_text: str,
    chunk_size_words: int = 150,
    overlap_words: int = 30,
    min_chunk_words: int = 20,
) -> list[str]:
    """Chunk a full markdown document into a list of self-contained text chunks."""
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    sections = split_into_sections(markdown_text)
    if not sections:
        return []

    merged_sections = _merge_small_sections(sections, min_chunk_words, chunk_size_words)

    chunks: list[str] = []
    for section in merged_sections:
        chunks.extend(_chunk_section(section, chunk_size_words, overlap_words))
    return [c for c in chunks if c.strip()]
