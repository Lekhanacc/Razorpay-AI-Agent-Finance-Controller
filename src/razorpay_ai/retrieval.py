"""TF-IDF + cosine similarity retrieval over the ingested knowledge index.

This is RAG V1: no vector database, no embeddings. The interface (retrieve() ->
list[RetrievedChunk]) is deliberately narrow so the TF-IDF implementation can later
be swapped for an embedding/vector-search backend without changing any caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CHUNKS_PATH,
    MATRIX_PATH,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    RETRIEVAL_TOP_K,
    VECTORIZER_PATH,
)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source_id: str
    title: str
    url: str
    product: str
    topic: str
    provenance_status: str
    chunk_id: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "product": self.product,
            "topic": self.topic,
            "provenance_status": self.provenance_status,
            "chunk_id": self.chunk_id,
        }


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class KnowledgeRetriever:
    """Ranks knowledge chunks against a query using TF-IDF cosine similarity."""

    def __init__(self, vectorizer, matrix, chunks: list[dict]):
        if matrix.shape[0] != len(chunks):
            raise ValueError("matrix row count must match number of chunks")
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def load(
        cls,
        vectorizer_path: Path = VECTORIZER_PATH,
        matrix_path: Path = MATRIX_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> "KnowledgeRetriever":
        vectorizer = joblib.load(vectorizer_path)
        matrix = joblib.load(matrix_path)
        chunks = load_chunks(chunks_path)
        return cls(vectorizer, matrix, chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks scoring at or above similarity_threshold.

        Returns an empty list if nothing clears the threshold -- callers must not
        assume the top-scoring chunk is automatically relevant.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not self.chunks:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1][:top_k]

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
