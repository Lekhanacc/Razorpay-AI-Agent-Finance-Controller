import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from razorpay_ai.api import app
from razorpay_ai.pipeline import RazorpayAIPipeline


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_healthy_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "razorpay-ai-nlu"}


def test_analyze_returns_the_existing_pipeline_contract(client):
    response = client.post("/v1/nlu/analyze", json={"text": "How do I create a payment link?"})
    assert response.status_code == 200
    body = response.json()
    assert {"text", "domain", "domain_confidence", "intent", "intent_confidence", "entities", "entity_extraction_status"} <= set(body)
    assert body["confidence_decision"] == "CLARIFY"
    pipeline_result = RazorpayAIPipeline.load().predict("How do I create a payment link?")
    assert {key: body[key] for key in pipeline_result} == pipeline_result


@pytest.mark.parametrize(
    ("text", "domain", "routing_status", "confidence_decision"),
    [
        ("Why did my Razorpay payment fail?", "RAZORPAY", None, "CLARIFY"),
        ("Why did my payment fail?", "AMBIGUOUS", "CLARIFICATION_REQUIRED", "CLARIFY"),
        ("Why did my Stripe payment fail?", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN", "OUT_OF_DOMAIN"),
    ],
)
def test_analyze_preserves_existing_routing(client, text, domain, routing_status, confidence_decision):
    body = client.post("/v1/nlu/analyze", json={"text": text}).json()
    assert body["domain"] == domain
    if routing_status is None:
        assert "routing_status" not in body
    else:
        assert body["routing_status"] == routing_status
    assert body["confidence_decision"] == confidence_decision


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 123}])
def test_analyze_rejects_missing_empty_or_invalid_text(client, payload):
    assert client.post("/v1/nlu/analyze", json=payload).status_code == 422


def test_analyze_preserves_entity_extraction(client):
    text = "Check pay_ABC123 for Rs. 250 by UPI on 2026-09-01"
    body = client.post("/v1/nlu/analyze", json={"text": text}).json()
    assert body["entities"] == {
        "payment_id": "pay_ABC123",
        "date": "2026-09-01",
        "amount": 250.0,
        "currency": "INR",
        "payment_method": "UPI",
    }
    assert body["confidence_decision"] in {"PROCEED", "CLARIFY", "OUT_OF_DOMAIN"}
