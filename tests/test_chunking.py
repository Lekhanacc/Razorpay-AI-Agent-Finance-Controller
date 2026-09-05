import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from razorpay_ai.chunking import chunk_document, split_into_sections


def test_split_into_sections_preserves_headings():
    text = "# Title\nintro text\n## Section A\ncontent a\n## Section B\ncontent b\n"
    sections = split_into_sections(text)
    headings = [s.heading for s in sections]
    assert "Title" in headings
    assert "Section A" in headings
    assert "Section B" in headings


def test_split_into_sections_handles_leading_content_with_no_heading():
    text = "just some text before any heading\n## First Heading\nbody"
    sections = split_into_sections(text)
    assert sections[0].heading is None
    assert "just some text" in sections[0].body


def test_split_into_sections_empty_document_returns_no_sections():
    assert split_into_sections("") == []
    assert split_into_sections("   \n  ") == []


def test_chunk_document_keeps_heading_attached_to_its_content():
    text = "## Refund Policy\nRefunds are processed within five to seven business days."
    chunks = chunk_document(text, chunk_size_words=150, overlap_words=30, min_chunk_words=5)
    assert len(chunks) == 1
    assert "Refund Policy" in chunks[0]
    assert "five to seven business days" in chunks[0]


def test_chunk_document_merges_small_sections_instead_of_emitting_tiny_chunks():
    text = "## A\nshort\n## B\nalso short\n## C\n" + ("word " * 5)
    chunks = chunk_document(text, chunk_size_words=150, overlap_words=30, min_chunk_words=20)
    # All three tiny sections should be merged into a single chunk, not three.
    assert len(chunks) == 1


def test_chunk_document_splits_long_sections_with_overlap():
    long_body = " ".join(f"word{i}" for i in range(400))
    text = f"## Long Section\n{long_body}"
    chunks = chunk_document(text, chunk_size_words=100, overlap_words=20, min_chunk_words=10)
    assert len(chunks) > 1
    # Every chunk should carry the heading for self-contained context.
    assert all("Long Section" in c for c in chunks)
    # Consecutive chunks should overlap: the tail of chunk N appears in chunk N+1.
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    overlap_candidate = " ".join(first_words[-10:])
    assert overlap_candidate in " ".join(second_words)


def test_chunk_document_rejects_overlap_greater_than_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_document("## X\nsome text here", chunk_size_words=50, overlap_words=50)


def test_chunk_document_on_empty_text_returns_no_chunks():
    assert chunk_document("") == []
