"""Minimal Gemini REST client for grounded RAG answer generation.

Deliberately thin: no SDK dependency, just httpx against the documented
generateContent REST endpoint. Never hard-codes an API key -- always read from
the environment. Raises typed, catchable errors instead of letting arbitrary
exceptions escape, so callers can implement the "insufficient evidence /
generation unavailable" behavior required by Phase 4 without guessing at
exception types.

NOTE ON VERIFICATION: this environment's network egress does not allow
generativelanguage.googleapis.com (verified: HTTP 403,
x-deny-reason: host_not_allowed). This client implements the real, documented
Gemini REST contract, but a live successful call has not been exercised from
this sandbox. See the Phase 4 report for exactly what was and wasn't verified.
"""

from __future__ import annotations

import os
import re
import time

import httpx

from .config import (
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_MODEL_NAME,
    GEMINI_REQUEST_TIMEOUT_SECONDS,
    GEMINI_TRANSIENT_RETRY_COUNT,
)


class MissingAPIKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is not set in the environment."""


class GenerationFailedError(RuntimeError):
    """Raised when the Gemini API call fails (network error, non-2xx, bad response shape)."""


def _get_api_key() -> str:
    api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
    if not api_key or not api_key.strip():
        raise MissingAPIKeyError(
            f"{GEMINI_API_KEY_ENV_VAR} is not set. Set it in the environment to enable answer generation."
        )
    return api_key


def _safe_gemini_error_detail(response: httpx.Response) -> str:
    """Return useful Google error metadata without exposing credentials or prompts."""
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return ""
    if not isinstance(error, dict):
        return ""
    fields = [str(error[field]) for field in ("code", "status", "message") if error.get(field)]
    detail = " | ".join(fields)
    # Defense in depth: redact common query-key and Google API-key forms before
    # exposing a provider error through the internal API response.
    detail = re.sub(r"(?i)(key=)[^\s&]+", r"\1[REDACTED]", detail)
    detail = re.sub(r"AIza[\w-]+", "[REDACTED]", detail)
    return detail


class GeminiClient:
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        base_url: str = GEMINI_API_BASE_URL,
        timeout_seconds: float = GEMINI_REQUEST_TIMEOUT_SECONDS,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """Call Gemini's generateContent endpoint and return the text response.

        Raises MissingAPIKeyError if no API key is configured, or
        GenerationFailedError for any network/HTTP/response-shape failure.
        """
        api_key = _get_api_key()
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        for attempt in range(GEMINI_TRANSIENT_RETRY_COUNT + 1):
            try:
                response = httpx.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                break
            except httpx.HTTPStatusError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if status_code is not None and status_code >= 500 and attempt < GEMINI_TRANSIENT_RETRY_COUNT:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                # Do not include str(exc): HTTPX embeds the query string and API key.
                detail = f"HTTP {status_code}" if status_code is not None else "an HTTP error"
                provider_detail = _safe_gemini_error_detail(exc.response)
                if provider_detail:
                    detail = f"{detail}: {provider_detail}"
                raise GenerationFailedError(f"Gemini request failed with {detail}.") from exc
            except httpx.RequestError as exc:
                raise GenerationFailedError("Gemini request failed due to a network error.") from exc
            except ValueError as exc:
                raise GenerationFailedError(f"Gemini returned a non-JSON response: {exc}") from exc
        else:  # pragma: no cover - loop either breaks or raises
            raise GenerationFailedError("Gemini request failed after transient retries.")

        try:
            candidates = body["candidates"]
            parts = candidates[0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationFailedError(f"Unexpected Gemini response shape: {body!r}") from exc

        if not text.strip():
            raise GenerationFailedError("Gemini returned an empty response.")
        return text
