import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from razorpay_ai.semantic_retrieval import SemanticRetriever


class _FakeEmbeddingModel:
    """Deterministic fake embedder for pure ranking/threshold/metadata tests,
    so those tests don't depend on loading the real (large) local model."""

    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self._vectors_by_text = vectors_by_text

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors_by_text[t] for t in texts], dtype=np.float32)


def _make_retriever():
    # 2D toy embedding space: axis 0 ~ "payment links", axis 1 ~ "subscriptions".
    chunk_texts = ["payment link chunk", "subscription chunk"]
    chunk_vectors = {
        "payment link chunk": [1.0, 0.0],
        "subscription chunk": [0.0, 1.0],
    }
    embeddings = np.array([chunk_vectors[t] for t in chunk_texts], dtype=np.float32)
    chunks = [
        {
            "chunk_id": "c0",
            "source_id": "payment_links",
            "title": "Payment Links",
            "url": "https://example.com/payment-links",
            "product": "Payment Links",
            "topic": "payment_links",
            "provenance_status": "VERIFIED",
            "text": "payment link chunk",
        },
        {
            "chunk_id": "c1",
            "source_id": "subscriptions",
            "title": "Subscriptions",
            "url": "https://example.com/subscriptions",
            "product": "Subscriptions",
            "topic": "subscriptions",
            "provenance_status": "VERIFIED",
            "text": "subscription chunk",
        },
    ]
    query_vectors = {
        "how do I create a payment link": [1.0, 0.0],
        "how do I set up a subscription plan": [0.0, 1.0],
        "totally unrelated query": [0.01, 0.01],
    }
    fake_model = _FakeEmbeddingModel(query_vectors)
    return SemanticRetriever(fake_model, embeddings, chunks)


def test_retrieve_ranks_the_semantically_closest_chunk_first():
    retriever = _make_retriever()
    results = retriever.retrieve("how do I create a payment link", top_k=2, similarity_threshold=0.0)
    assert results[0].source_id == "payment_links"


def test_retrieve_respects_top_k():
    retriever = _make_retriever()
    results = retriever.retrieve("how do I set up a subscription plan", top_k=1, similarity_threshold=0.0)
    assert len(results) == 1


def test_retrieve_applies_similarity_threshold():
    retriever = _make_retriever()
    results = retriever.retrieve("totally unrelated query", top_k=2, similarity_threshold=0.5)
    assert results == []


def test_retrieve_preserves_full_provenance_metadata():
    retriever = _make_retriever()
    results = retriever.retrieve("how do I create a payment link", top_k=1, similarity_threshold=0.0)
    chunk_dict = results[0].to_dict()
    assert set(chunk_dict) == {
        "text", "score", "source_id", "title", "url", "product", "topic", "provenance_status", "chunk_id",
    }
    assert chunk_dict["source_id"] == "payment_links"


def test_retrieve_rejects_empty_query():
    retriever = _make_retriever()
    with pytest.raises(ValueError):
        retriever.retrieve("", top_k=2)


def test_retriever_rejects_mismatched_embeddings_and_chunks():
    embeddings = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        SemanticRetriever(_FakeEmbeddingModel({}), embeddings, chunks=[{"source_id": "only_one"}])


def test_semantic_retriever_load_raises_file_not_found_when_index_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        SemanticRetriever.load(
            embeddings_path=tmp_path / "missing_embeddings.npy",
            chunks_path=tmp_path / "missing_chunks.jsonl",
        )


@pytest.fixture(scope="module")
def real_semantic_retriever():
    # Integration test against the actual built index + real local model.
    return SemanticRetriever.load()


def test_real_semantic_retriever_returns_relevant_result_for_a_real_corpus_query(real_semantic_retriever):
    results = real_semantic_retriever.retrieve("What is the test mode limit for payment links?", top_k=3, similarity_threshold=0.0)
    assert results
    assert results[0].source_id == "sl_payment_links_create_standard"


def test_real_semantic_retriever_preserves_provenance_end_to_end(real_semantic_retriever):
    results = real_semantic_retriever.retrieve("payment link expiry", top_k=1, similarity_threshold=0.0)
    assert results
    chunk = results[0]
    assert chunk.source_id
    assert chunk.url.startswith("https://")
    assert chunk.provenance_status in {"VERIFIED", "MISMATCH_FLAGGED", "VERIFIED_REGIONAL_VARIANT"}
