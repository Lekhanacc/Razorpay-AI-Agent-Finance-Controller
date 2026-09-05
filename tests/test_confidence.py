import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from razorpay_ai.confidence import ConfidenceDecision, evaluate_confidence
from razorpay_ai.config import DOMAIN_CONFIDENCE_THRESHOLD, INTENT_CONFIDENCE_THRESHOLD


def razorpay_result(domain_confidence, intent_confidence):
    return {
        "domain": "RAZORPAY",
        "domain_confidence": domain_confidence,
        "intent_confidence": intent_confidence,
    }


def test_confident_razorpay_result_proceeds():
    assert evaluate_confidence(
        razorpay_result(DOMAIN_CONFIDENCE_THRESHOLD, INTENT_CONFIDENCE_THRESHOLD)
    ) == ConfidenceDecision.PROCEED


def test_low_razorpay_domain_confidence_requires_clarification():
    assert evaluate_confidence(
        razorpay_result(DOMAIN_CONFIDENCE_THRESHOLD - 0.01, INTENT_CONFIDENCE_THRESHOLD)
    ) == ConfidenceDecision.CLARIFY


def test_low_razorpay_intent_confidence_requires_clarification():
    assert evaluate_confidence(
        razorpay_result(DOMAIN_CONFIDENCE_THRESHOLD, INTENT_CONFIDENCE_THRESHOLD - 0.01)
    ) == ConfidenceDecision.CLARIFY


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("AMBIGUOUS", ConfidenceDecision.CLARIFY),
        ("OUT_OF_DOMAIN", ConfidenceDecision.OUT_OF_DOMAIN),
    ],
)
def test_non_razorpay_domains_preserve_existing_routing_semantics(domain, expected):
    assert evaluate_confidence(
        {"domain": domain, "domain_confidence": 1.0, "intent_confidence": None}
    ) == expected
