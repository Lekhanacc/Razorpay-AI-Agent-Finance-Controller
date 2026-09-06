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


def test_rag_retrieve_returns_expected_response_shape(client):
    response = client.post("/v1/rag/retrieve", json={"text": "How do I create a payment link?"})
    assert response.status_code == 200
    body = response.json()
    assert {"query", "domain", "retrieval_status", "results"} <= set(body)
    assert body["query"] == "How do I create a payment link?"
    for chunk in body["results"]:
        assert {"text", "score", "source_id", "title", "url", "product", "topic"} <= set(chunk)


def test_rag_retrieve_skips_retrieval_for_out_of_domain_query(client):
    response = client.post("/v1/rag/retrieve", json={"text": "Why did my Stripe payment fail?"})
    body = response.json()
    assert body["domain"] == "OUT_OF_DOMAIN"
    assert body["retrieval_status"] == "SKIPPED_OUT_OF_DOMAIN"
    assert body["results"] == []


def test_rag_retrieve_skips_retrieval_for_ambiguous_query(client):
    response = client.post("/v1/rag/retrieve", json={"text": "Why did my payment fail?"})
    body = response.json()
    assert body["domain"] == "AMBIGUOUS"
    assert body["retrieval_status"] == "SKIPPED_AMBIGUOUS"
    assert body["results"] == []


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 123}])
def test_rag_retrieve_rejects_missing_empty_or_invalid_text(client, payload):
    assert client.post("/v1/rag/retrieve", json=payload).status_code == 422


def test_existing_nlu_endpoint_is_unaffected_by_rag_addition(client):
    # Guards against Phase 2 regressing Phase 1's contract.
    response = client.post("/v1/nlu/analyze", json={"text": "How do I create a payment link?"})
    assert response.status_code == 200
    body = response.json()
    assert {"text", "domain", "domain_confidence", "intent", "intent_confidence", "entities", "entity_extraction_status", "confidence_decision"} <= set(body)
    assert "results" not in body
    assert "retrieval_status" not in body


def test_health_endpoint_still_works_alongside_rag_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "razorpay-ai-nlu"}
