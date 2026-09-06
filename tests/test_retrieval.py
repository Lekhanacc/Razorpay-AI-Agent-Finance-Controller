import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from razorpay_ai.retrieval import KnowledgeRetriever


def _build_retriever(texts: list[str], metadata: list[dict]) -> KnowledgeRetriever:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    chunks = [
        {
            "chunk_id": f"chunk{i}",
            "source_id": meta["source_id"],
            "title": meta.get("title", "Title"),
            "url": meta.get("url", "https://example.com"),
            "product": meta.get("product", "Product"),
            "topic": meta.get("topic", "topic"),
            "provenance_status": meta.get("provenance_status", "VERIFIED"),
            "text": text,
        }
        for i, (text, meta) in enumerate(zip(texts, metadata))
    ]
    return KnowledgeRetriever(vectorizer, matrix, chunks)


@pytest.fixture
def sample_retriever():
    texts = [
        "How to create a payment link with amount and expiry date",
        "Refund policy and refund processing timelines for captured payments",
        "Subscription plans define billing interval and period",
        "Webhook events notify you about payment status changes",
    ]
    metadata = [
        {"source_id": "payment_links", "topic": "payment_links"},
        {"source_id": "refunds", "topic": "refunds"},
        {"source_id": "subscriptions", "topic": "subscriptions"},
        {"source_id": "webhooks", "topic": "webhooks"},
    ]
    return _build_retriever(texts, metadata)


def test_retrieve_ranks_the_most_relevant_chunk_first(sample_retriever):
    results = sample_retriever.retrieve("How do I set an expiry date for a payment link?", top_k=5, similarity_threshold=0.0)
    assert results
    assert results[0].source_id == "payment_links"


def test_retrieve_respects_top_k(sample_retriever):
    results = sample_retriever.retrieve("payment", top_k=2, similarity_threshold=0.0)
    assert len(results) <= 2


def test_retrieve_applies_similarity_threshold(sample_retriever):
    # An unrelated query should score low against every chunk; a high threshold
    # must filter everything out rather than returning a spuriously "best" match.
    results = sample_retriever.retrieve("unrelated query about something else entirely", top_k=5, similarity_threshold=0.9)
    assert results == []


def test_retrieve_returns_empty_results_object_shape_when_nothing_passes_threshold(sample_retriever):
    results = sample_retriever.retrieve("completely unrelated gibberish zzz", top_k=5, similarity_threshold=0.99)
    assert results == []


def test_retrieve_preserves_full_provenance_metadata(sample_retriever):
    results = sample_retriever.retrieve("refund processing timeline", top_k=1, similarity_threshold=0.0)
    assert results
    chunk_dict = results[0].to_dict()
    assert set(chunk_dict) == {
        "text", "score", "source_id", "title", "url", "product", "topic", "provenance_status", "chunk_id",
    }
    assert chunk_dict["source_id"] == "refunds"


def test_retrieve_rejects_empty_query(sample_retriever):
    with pytest.raises(ValueError):
        sample_retriever.retrieve("", top_k=5)
    with pytest.raises(ValueError):
        sample_retriever.retrieve("   ", top_k=5)


def test_retriever_rejects_mismatched_matrix_and_chunks():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["apple banana", "cherry date", "elderberry fig"])
    with pytest.raises(ValueError):
        KnowledgeRetriever(vectorizer, matrix, chunks=[{"source_id": "only_one"}])


def test_retriever_on_empty_chunk_set_returns_no_results():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["placeholder"])
    import scipy.sparse

    empty_matrix = scipy.sparse.csr_matrix((0, len(vectorizer.vocabulary_)))
    retriever = KnowledgeRetriever(vectorizer, empty_matrix, chunks=[])
    assert retriever.retrieve("anything", top_k=5, similarity_threshold=0.0) == []
