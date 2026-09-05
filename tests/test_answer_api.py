import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from razorpay_ai.api import app
from razorpay_ai.generation import GenerationFailedError, MissingAPIKeyError


class _FakeGeminiClient:
    def __init__(self, response_text=None, exception=None):
        self._response_text = response_text
        self._exception = exception

    def generate(self, system_instruction, user_prompt):
        if self._exception is not None:
            raise self._exception
        return self._response_text


# A query verified (see tests/test_answer_pipeline.py) to reach PROCEED with real
# retrievable evidence in the small 7-document corpus.
PROCEED_QUERY = "The customer says they paid. Can I check the transaction status? for a customer"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_answer_endpoint_out_of_domain(client):
    response = client.post("/v1/rag/answer", json={"text": "Why did my Stripe payment fail?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OUT_OF_DOMAIN"
    assert body["sources"] == []


def test_answer_endpoint_clarify(client):
    response = client.post("/v1/rag/answer", json={"text": "Why did my payment fail?"})
    assert response.status_code == 200
    assert response.json()["status"] == "CLARIFY"


def test_answer_endpoint_rejects_empty_text(client):
    assert client.post("/v1/rag/answer", json={"text": ""}).status_code == 422
    assert client.post("/v1/rag/answer", json={}).status_code == 422


def test_answer_endpoint_missing_api_key_uses_evidence_only_fallback(client):
    # Gemini is optional: missing server-side credentials must not break an
    # otherwise evidence-grounded UI response.
    import os

    assert os.environ.get("GEMINI_API_KEY") in (None, "")
    response = client.post("/v1/rag/answer", json={"text": PROCEED_QUERY})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ANSWERED"
    assert body["reason"] == "local_grounded_fallback: missing_api_key"
    assert body["answer"]
    assert body["sources"]


def test_answer_endpoint_with_mocked_successful_generation(client):
    original = client.app.state.answer_pipeline.gemini_client
    client.app.state.answer_pipeline.gemini_client = _FakeGeminiClient(response_text="Mocked grounded answer.")
    try:
        response = client.post("/v1/rag/answer", json={"text": PROCEED_QUERY})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ANSWERED"
        assert body["answer"] == "Mocked grounded answer."
        assert body["sources"]
    finally:
        client.app.state.answer_pipeline.gemini_client = original


def test_answer_endpoint_with_mocked_generation_failure(client):
    original = client.app.state.answer_pipeline.gemini_client
    client.app.state.answer_pipeline.gemini_client = _FakeGeminiClient(exception=GenerationFailedError("simulated failure"))
    try:
        response = client.post("/v1/rag/answer", json={"text": PROCEED_QUERY})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "GENERATION_FAILED"
        assert body["answer"] is None
        assert body["sources"]
    finally:
        client.app.state.answer_pipeline.gemini_client = original


def test_answer_endpoint_returns_503_when_index_unavailable(client):
    original = client.app.state.answer_pipeline
    client.app.state.answer_pipeline = None
    try:
        response = client.post("/v1/rag/answer", json={"text": "How do I create a payment link?"})
        assert response.status_code == 503
    finally:
        client.app.state.answer_pipeline = original


def test_existing_endpoints_unaffected_by_phase_4(client):
    health = client.get("/health")
    assert health.status_code == 200

    analyze = client.post("/v1/nlu/analyze", json={"text": "How do I create a payment link?"})
    assert analyze.status_code == 200
    assert "answer" not in analyze.json()

    retrieve = client.post("/v1/rag/retrieve", json={"text": "How do I create a payment link?"})
    assert retrieve.status_code == 200
    assert "answer" not in retrieve.json()
