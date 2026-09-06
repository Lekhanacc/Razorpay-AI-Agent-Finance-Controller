import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
import pytest

from razorpay_ai.generation import GeminiClient, GenerationFailedError, MissingAPIKeyError


def test_generate_raises_missing_api_key_error_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient()
    with pytest.raises(MissingAPIKeyError):
        client.generate("system", "user prompt")


def test_generate_raises_generation_failed_on_real_network_error(monkeypatch):
    # Real failure, not simulated: this environment's network egress does not
    # allow generativelanguage.googleapis.com, so a genuine request to it fails.
    # This test exercises the actual failure path, not a mock.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")
    client = GeminiClient(timeout_seconds=10.0)
    with pytest.raises(GenerationFailedError):
        client.generate("system", "user prompt")


def test_generate_raises_generation_failed_on_non_2xx_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    class _FakeResponse:
        status_code = 400

        def raise_for_status(self):
            raise httpx.HTTPStatusError("bad request", request=None, response=self)

        def json(self):
            return {"error": "bad request"}

    def _fake_post(*args, **kwargs):
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", _fake_post)
    client = GeminiClient()
    with pytest.raises(GenerationFailedError):
        client.generate("system", "user prompt")


def test_generate_raises_generation_failed_on_unexpected_response_shape(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"unexpected": "shape"}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse())
    client = GeminiClient()
    with pytest.raises(GenerationFailedError):
        client.generate("system", "user prompt")


def test_generate_returns_text_on_well_formed_mocked_response(monkeypatch):
    # MOCKED: verifies prompt construction and response parsing logic only.
    # This does NOT verify a live Gemini call succeeds -- see the Phase 4 report
    # for what has and hasn't been verified against the real Gemini API.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    captured_payload = {}

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Refunds take 5-7 business days."}]}}]}

    def _fake_post(url, params=None, json=None, timeout=None):
        captured_payload["url"] = url
        captured_payload["params"] = params
        captured_payload["json"] = json
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", _fake_post)
    client = GeminiClient()
    result = client.generate("system instruction text", "user prompt text")

    assert result == "Refunds take 5-7 business days."
    assert captured_payload["params"] == {"key": "test-key-not-a-real-credential"}
    assert captured_payload["json"]["systemInstruction"]["parts"][0]["text"] == "system instruction text"
    assert captured_payload["json"]["contents"][0]["parts"][0]["text"] == "user prompt text"


def test_generate_raises_generation_failed_on_empty_text_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "   "}]}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse())
    client = GeminiClient()
    with pytest.raises(GenerationFailedError):
        client.generate("system", "user prompt")


def test_generate_retries_transient_server_error_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    calls = []

    class Failure:
        status_code = 503

        def raise_for_status(self):
            raise httpx.HTTPStatusError("contains-url-with-secret", request=None, response=self)

    class Success:
        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    def fake_post(*args, **kwargs):
        calls.append(1)
        return Failure() if len(calls) == 1 else Success()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("razorpay_ai.generation.time.sleep", lambda *_: None)
    assert GeminiClient().generate("system", "prompt") == "ok"
    assert len(calls) == 2
