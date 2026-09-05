"""Inference pipeline with domain-first routing."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import DOMAIN_MODEL_PATH, ENTITY_EXTRACTION_STATUS, INTENT_MODEL_PATH
from .entities import extract_entities


class RazorpayAIPipeline:
    def __init__(self, domain_model, intent_model):
        self.domain_model = domain_model
        self.intent_model = intent_model

    @classmethod
    def load(cls, domain_path: Path = DOMAIN_MODEL_PATH, intent_path: Path = INTENT_MODEL_PATH) -> "RazorpayAIPipeline":
        return cls(joblib.load(domain_path), joblib.load(intent_path))

    @staticmethod
    def _prediction(model, text: str) -> tuple[str, float]:
        probabilities = model.predict_proba([text])[0]
        index = probabilities.argmax()
        return str(model.classes_[index]), float(probabilities[index])

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        domain, domain_confidence = self._prediction(self.domain_model, text)
        result = {
            "text": text,
            "domain": domain,
            "domain_confidence": domain_confidence,
            "intent": None,
            "intent_confidence": None,
            "entities": extract_entities(text),
            "entity_extraction_status": ENTITY_EXTRACTION_STATUS,
        }
        if domain == "RAZORPAY":
            intent, intent_confidence = self._prediction(self.intent_model, text)
            result["intent"] = intent
            result["intent_confidence"] = intent_confidence
        elif domain == "AMBIGUOUS":
            result["routing_status"] = "CLARIFICATION_REQUIRED"
        else:
            result["routing_status"] = "OUT_OF_DOMAIN"
        return result
