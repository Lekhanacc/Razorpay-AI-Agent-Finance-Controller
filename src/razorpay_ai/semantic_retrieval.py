"""Semantic retrieval: local sentence-transformers embeddings + cosine similarity.

Deliberately mirrors razorpay_ai.retrieval.KnowledgeRetriever's interface
(retrieve() -> list[RetrievedChunk] with identical fields) so the two retrieval
backends can be swapped and compared without touching any caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    SEMANTIC_CHUNKS_PATH,
    SEMANTIC_EMBEDDINGS_PATH,
    SEMANTIC_SIMILARITY_THRESHOLD,
    SEMANTIC_TOP_K,
)
from .embeddings import LocalEmbeddingModel
from .retrieval import RetrievedChunk


def load_semantic_chunks(chunks_path: Path = SEMANTIC_CHUNKS_PATH) -> list[dict]:
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class SemanticRetriever:
    """Ranks knowledge chunks against a query using sentence-embedding cosine similarity."""

    def __init__(self, embedding_model: LocalEmbeddingModel, embeddings: np.ndarray, chunks: list[dict]):
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings row count must match number of chunks")
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.chunks = chunks

    @classmethod
    def load(
        cls,
        embeddings_path: Path = SEMANTIC_EMBEDDINGS_PATH,
        chunks_path: Path = SEMANTIC_CHUNKS_PATH,
    ) -> "SemanticRetriever":
        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Semantic index not found ({embeddings_path}, {chunks_path}). "
                "Run scripts/build_semantic_index.py first."
            )
        embedding_model = LocalEmbeddingModel()
        embeddings = np.load(embeddings_path)
        chunks = load_semantic_chunks(chunks_path)
        return cls(embedding_model, embeddings, chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = SEMANTIC_TOP_K,
        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks scoring at or above similarity_threshold.

        Both query and chunk embeddings are L2-normalized, so cosine similarity
        reduces to a plain dot product.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not self.chunks:
            return []

        query_embedding = self.embedding_model.encode([query])[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < similarity_threshold:
                continue
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    text=chunk["text"],
                    score=score,
                    source_id=chunk["source_id"],
                    title=chunk["title"],
                    url=chunk["url"],
                    product=chunk["product"],
                    topic=chunk["topic"],
                    provenance_status=chunk["provenance_status"],
                    chunk_id=chunk["chunk_id"],
                )
            )
        return results
