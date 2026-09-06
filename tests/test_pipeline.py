import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from razorpay_ai.data import load_dataset, normalize_domain_labels, split_dataset
from razorpay_ai.domain import capped_balanced_class_weights, train_domain_model
from razorpay_ai.entities import extract_entities
from razorpay_ai.intent import train_intent_model
from razorpay_ai.pipeline import RazorpayAIPipeline


@pytest.fixture(scope="module")
def trained_pipeline():
    data = load_dataset().data
    domain_train, _, _ = split_dataset(data, "domain")
    razorpay = data[data.domain == "RAZORPAY"].reset_index(drop=True)
    intent_train, _, _ = split_dataset(razorpay, "intent")
    return RazorpayAIPipeline(
        train_domain_model(domain_train.text, domain_train.domain),
        train_intent_model(intent_train.text, intent_train.intent),
    )


def test_dataset_loading_and_validation():
    bundle = load_dataset()
    assert len(bundle.data) == 770
    assert len(bundle.taxonomy["intents"]) == 21
    assert not bundle.data.text.duplicated().any()


def test_label_normalization():
    bundle = load_dataset()
    assert set(bundle.data.domain) == {"RAZORPAY", "AMBIGUOUS", "OUT_OF_DOMAIN"}
    assert normalize_domain_labels(bundle.data).domain.equals(bundle.data.domain)


def test_model_training(trained_pipeline):
    assert set(trained_pipeline.domain_model.classes_) == {"RAZORPAY", "AMBIGUOUS", "OUT_OF_DOMAIN"}
    assert len(trained_pipeline.intent_model.classes_) == 21


def test_domain_class_weights_are_capped_for_rare_labels():
    weights = capped_balanced_class_weights(["RAZORPAY"] * 60 + ["AMBIGUOUS"] * 3 + ["OUT_OF_DOMAIN"])
    assert weights["RAZORPAY"] == 1.0
    assert weights["AMBIGUOUS"] == pytest.approx(64 / 9)
    assert weights["OUT_OF_DOMAIN"] == 10.0


def test_ood_routing(trained_pipeline):
    result = trained_pipeline.predict("Why did my Stripe payment fail?")
    assert result["domain"] == "OUT_OF_DOMAIN"
    assert result["intent"] is None


def test_ambiguous_routing(trained_pipeline):
    result = trained_pipeline.predict("Why did my payment fail?")
    assert result["domain"] == "AMBIGUOUS"
    assert result["intent"] is None
    assert result["routing_status"] == "CLARIFICATION_REQUIRED"


def test_razorpay_routing(trained_pipeline):
    result = trained_pipeline.predict("Why did my Razorpay payment fail?")
    assert result["domain"] == "RAZORPAY"
    assert result["intent"] == "FAILURE_TROUBLESHOOT"


def test_output_schema(trained_pipeline):
    result = trained_pipeline.predict("How do I create a payment link?")
    assert {"text", "domain", "domain_confidence", "intent", "intent_confidence", "entities", "entity_extraction_status"} <= set(result)
    assert 0 <= result["domain_confidence"] <= 1
    assert result["entity_extraction_status"] == "RULE_BASED_V1"


def test_explicit_entity_extraction():
    result = extract_entities("Refund pay_ABC123 of Rs. 1,250.50 by UPI to a@merchant.example on 2026-09-01")
    assert result == {"payment_id": "pay_ABC123", "email": "a@merchant.example", "date": "2026-09-01", "amount": 1250.5, "currency": "INR", "payment_method": "UPI"}


def test_entity_extraction_omits_unmentioned_values():
    assert extract_entities("How do I create a payment link?") == {}
