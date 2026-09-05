import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from razorpay_ai.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_semantic_retriever_returns_expected_response_shape(client):
    response = client.post(
        "/v1/rag/retrieve",
        json={"text": "What is the test mode limit for payment links?", "retriever": "semantic"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["retriever"] == "semantic"
    assert {"query", "domain", "retriever", "retrieval_status", "results"} <= set(body)
    for chunk in body["results"]:
        assert {"text", "score", "source_id", "title", "url", "product", "topic"} <= set(chunk)


def test_default_retriever_is_still_tfidf_when_field_omitted(client):
    # Backward compatibility: existing Phase 2 callers that never send "retriever"
    # must see identical behavior to before this phase.
    response = client.post("/v1/rag/retrieve", json={"text": "How do I create a payment link?"})
    assert response.status_code == 200
    assert response.json()["retriever"] == "tfidf"


def test_semantic_retriever_skips_retrieval_for_out_of_domain_query(client):
    response = client.post(
        "/v1/rag/retrieve",
        json={"text": "Why did my Stripe payment fail?", "retriever": "semantic"},
    )
    body = response.json()
    assert body["domain"] == "OUT_OF_DOMAIN"
    assert body["retrieval_status"] == "SKIPPED_OUT_OF_DOMAIN"
    assert body["results"] == []


def test_invalid_retriever_value_is_rejected_with_422(client):
    response = client.post(
        "/v1/rag/retrieve",
        json={"text": "How do I create a payment link?", "retriever": "not_a_real_retriever"},
    )
    assert response.status_code == 422


def test_semantic_retriever_returns_503_when_unavailable(client, monkeypatch):
    # Simulate the semantic index/model not being available, without touching
    # the real loaded state used by other tests in this module.
    original = client.app.state.rag_pipeline.semantic_retriever
    client.app.state.rag_pipeline.semantic_retriever = None
    try:
        response = client.post(
            "/v1/rag/retrieve",
            json={"text": "How do I create a payment link?", "retriever": "semantic"},
        )
        assert response.status_code == 503
    finally:
        client.app.state.rag_pipeline.semantic_retriever = original


def test_tfidf_retriever_still_works_when_semantic_is_unavailable(client):
    original = client.app.state.rag_pipeline.semantic_retriever
    client.app.state.rag_pipeline.semantic_retriever = None
    try:
        response = client.post(
            "/v1/rag/retrieve",
            json={"text": "How do I create a payment link?", "retriever": "tfidf"},
        )
        assert response.status_code == 200
        assert response.json()["retriever"] == "tfidf"
    finally:
        client.app.state.rag_pipeline.semantic_retriever = original


def test_existing_nlu_endpoint_still_unaffected_by_phase_3(client):
    response = client.post("/v1/nlu/analyze", json={"text": "How do I create a payment link?"})
    assert response.status_code == 200
    body = response.json()
    assert "retriever" not in body
    assert "results" not in body
