"""Guardrails for consequential Razorpay action authorizations.

This project deliberately has no Razorpay credential or execution adapter.  This
module therefore authorizes or rejects a requested action; it never silently
creates an order or payment link.  A future executor must consume only an
``APPROVED`` decision and perform the provider call separately.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


SUPPORTED_ACTIONS = frozenset({"CREATE_ORDER", "CREATE_PAYMENT_LINK"})


class AuditLogger:
    """Append non-secret action decisions as JSON Lines for operational review."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, record: dict) -> str:
        audit_id = str(uuid4())
        entry = {"audit_id": audit_id, "timestamp": datetime.now(UTC).isoformat(), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return audit_id


class ActionGuard:
    """Authorize only known, confirmed, within-limit consequential actions."""

    def __init__(self, audit_logger: AuditLogger, max_amount_paise: int, max_actions_per_conversation: int):
        self.audit_logger = audit_logger
        self.max_amount_paise = max_amount_paise
        self.max_actions_per_conversation = max_actions_per_conversation
        self._approved_counts: dict[str, int] = {}

    def authorize(
        self, action: str, amount_paise: int | None, confirmed: bool, conversation_id: str | None
    ) -> dict:
        conversation_key = conversation_id or "anonymous"
        status, reason = "APPROVED", None
        if action not in SUPPORTED_ACTIONS:
            status, reason = "REJECTED_UNSUPPORTED", "action is not in the supported allowlist"
        elif not confirmed:
            status, reason = "CONFIRMATION_REQUIRED", "consequential actions require confirmed=true"
        elif amount_paise is not None and (amount_paise <= 0 or amount_paise > self.max_amount_paise):
            status, reason = "REJECTED_LIMIT", "amount exceeds the configured action limit"
        elif self._approved_counts.get(conversation_key, 0) >= self.max_actions_per_conversation:
            status, reason = "REJECTED_LIMIT", "conversation action limit reached"

        if status == "APPROVED":
            self._approved_counts[conversation_key] = self._approved_counts.get(conversation_key, 0) + 1
        audit_id = self.audit_logger.record(
            {
                "event": "action_authorization",
                "action": action,
                "amount_paise": amount_paise,
                "conversation_id": conversation_id,
                "confirmed": confirmed,
                "status": status,
                "reason": reason,
            }
        )
        return {"status": status, "reason": reason, "audit_id": audit_id}
