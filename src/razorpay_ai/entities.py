"""Deterministic extraction for explicit Razorpay entities in user text.

This is intentionally conservative. The supplied training data has no entity
annotations, so a learned entity model would have no labels from which to learn.
"""

from __future__ import annotations

import re
from typing import Any


_IDENTIFIER_PATTERNS = {
    "payment_id": r"\bpay_[A-Za-z0-9]+\b", "order_id": r"\border_[A-Za-z0-9]+\b",
    "refund_id": r"\brfnd_[A-Za-z0-9]+\b", "payout_id": r"\bpout_[A-Za-z0-9]+\b",
    "transfer_id": r"\btrf_[A-Za-z0-9]+\b", "invoice_id": r"\binv_[A-Za-z0-9]+\b",
    "subscription_id": r"\bsub_[A-Za-z0-9]+\b", "payment_link_id": r"\bplink_[A-Za-z0-9]+\b",
    "customer_id": r"\bcust_[A-Za-z0-9]+\b",
}
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+91[ -]?)?[6-9]\d{9}(?!\w)")
_ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_PATTERN = re.compile(r"(?:₹\s*|\b(?:inr|rs\.?|usd|\$)\s*)(\d+(?:,\d{3})*(?:\.\d{1,2})?)|(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:inr|rs\.?|usd)\b", re.IGNORECASE)
_CURRENCY_PATTERNS = {"INR": re.compile(r"₹|\b(?:inr|rs\.?)\b", re.IGNORECASE), "USD": re.compile(r"\$|\busd\b", re.IGNORECASE)}
_PAYMENT_METHODS = {"upi": "UPI", "credit card": "CARD", "debit card": "CARD", "card": "CARD", "netbanking": "NETBANKING", "net banking": "NETBANKING", "wallet": "WALLET"}


def _first_match(pattern: str | re.Pattern[str], text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE) if isinstance(pattern, str) else pattern.search(text)
    return match.group(0) if match else None


def extract_entities(text: str) -> dict[str, Any]:
    """Return explicitly supplied values, omitting absent fields."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    entities: dict[str, Any] = {}
    for entity_name, pattern in _IDENTIFIER_PATTERNS.items():
        if value := _first_match(pattern, text):
            entities[entity_name] = value
    if email := _first_match(_EMAIL_PATTERN, text): entities["email"] = email
    if phone := _first_match(_PHONE_PATTERN, text): entities["phone"] = re.sub(r"[ -]", "", phone)
    if date := _first_match(_ISO_DATE_PATTERN, text): entities["date"] = date
    if amount_match := _AMOUNT_PATTERN.search(text):
        entities["amount"] = float((amount_match.group(1) or amount_match.group(2)).replace(",", ""))
    for code, pattern in _CURRENCY_PATTERNS.items():
        if pattern.search(text):
            entities["currency"] = code
            break
    normalized_text = text.lower()
    for phrase, method in _PAYMENT_METHODS.items():
        if phrase in normalized_text:
            entities["payment_method"] = method
            break
    return entities
