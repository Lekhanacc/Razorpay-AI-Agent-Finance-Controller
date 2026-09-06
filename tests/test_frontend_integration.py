import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.api import app


def test_part_one_frontend_is_served_by_the_current_api():
    """The retained UI must use the Part 2 API, never the old Express/Gemini route."""
    with TestClient(app) as client:
        index = client.get("/")
        script = client.get("/script.js")
        config = client.get("/config.js")

    assert index.status_code == 200
    assert "Razorpay AI Agent" in index.text
    assert script.status_code == 200
    assert "/v1/rag/answer" in script.text
    assert 'conversation_id: conversationId' in script.text
    assert 'fetch("/agent"' not in script.text
    assert "apiBaseUrl" in config.text
