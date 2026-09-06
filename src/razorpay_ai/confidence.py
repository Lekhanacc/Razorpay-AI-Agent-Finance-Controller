"""Explicit, provisional confidence policy for NLU routing decisions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from .config import DOMAIN_CONFIDENCE_THRESHOLD, INTENT_CONFIDENCE_THRESHOLD


class ConfidenceDecision(StrEnum):
    PROCEED = "PROCEED"
    CLARIFY = "CLARIFY"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


def evaluate_confidence(result: Mapping[str, Any]) -> ConfidenceDecision:
    """Apply policy thresholds to an existing pipeline result.

    Thresholds are operational safeguards only. They require calibration against
    representative held-out data before being treated as probability estimates.
    """
    domain = result["domain"]
    if domain == "OUT_OF_DOMAIN":
        return ConfidenceDecision.OUT_OF_DOMAIN
    if domain == "AMBIGUOUS":
        return ConfidenceDecision.CLARIFY
    if domain != "RAZORPAY":
        return ConfidenceDecision.CLARIFY

    domain_confidence = result["domain_confidence"]
    intent_confidence = result["intent_confidence"]
    if domain_confidence < DOMAIN_CONFIDENCE_THRESHOLD:
        return ConfidenceDecision.CLARIFY
    if intent_confidence is None or intent_confidence < INTENT_CONFIDENCE_THRESHOLD:
        return ConfidenceDecision.CLARIFY
    return ConfidenceDecision.PROCEED
