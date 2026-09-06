import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.actions import ActionGuard, AuditLogger
from razorpay_ai.api import authorize_action
from razorpay_ai.schemas import ActionAuthorizationRequest


def guard(tmp_path, limit=100, count=2):
    return ActionGuard(AuditLogger(tmp_path / "audit.jsonl"), limit, count)


def test_unsupported_action_is_rejected_and_audited(tmp_path):
    result = guard(tmp_path).authorize("DELETE_ACCOUNT", None, True, "one")
    assert result["status"] == "REJECTED_UNSUPPORTED"
    entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert entry["status"] == "REJECTED_UNSUPPORTED"
    assert entry["audit_id"] == result["audit_id"]


def test_consequential_action_requires_confirmation(tmp_path):
    result = guard(tmp_path).authorize("CREATE_ORDER", 50, False, "one")
    assert result["status"] == "CONFIRMATION_REQUIRED"


def test_limits_and_count_are_enforced(tmp_path):
    policy = guard(tmp_path, limit=100, count=1)
    assert policy.authorize("CREATE_ORDER", 101, True, "one")["status"] == "REJECTED_LIMIT"
    assert policy.authorize("CREATE_ORDER", 100, True, "one")["status"] == "APPROVED"
    assert policy.authorize("CREATE_PAYMENT_LINK", 1, True, "one")["status"] == "REJECTED_LIMIT"


def test_api_authorization_boundary_uses_the_guard(tmp_path):
    class State:
        action_guard = guard(tmp_path)

    class Request:
        app = type("App", (), {"state": State()})()

    response = authorize_action(ActionAuthorizationRequest(action="create_order", amount_paise=10), Request())
    assert response["status"] == "CONFIRMATION_REQUIRED"
