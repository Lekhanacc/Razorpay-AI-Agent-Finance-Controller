import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from razorpay_ai.answer_pipeline import AnswerPipeline
from razorpay_ai.generation import GenerationFailedError, MissingAPIKeyError
from razorpay_ai.rag_pipeline import RAGRetrievalPipeline


class _FakeGeminiClient:
    def __init__(self, response_text=None, exception=None):
        self._response_text = response_text
        self._exception = exception
        self.last_call = None

    def generate(self, system_instruction, user_prompt):
        self.last_call = (system_instruction, user_prompt)
        if self._exception is not None:
            raise self._exception
        return self._response_text


@pytest.fixture(scope="module")
def real_rag_pipeline():
    # Uses the real, already-built Phase 2/3 indexes -- consistent with how
    # test_rag_pipeline.py tests the retrieval layer against real artifacts.
    return RAGRetrievalPipeline.load()


def test_out_of_domain_query_never_reaches_retrieval_or_generation(real_rag_pipeline):
    fake_gemini = _FakeGeminiClient(response_text="should never be returned")
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    result = pipeline.answer("Why did my Stripe payment fail?")
    assert result["status"] == "OUT_OF_DOMAIN"
    assert result["sources"] == []
    assert fake_gemini.last_call is None


def test_ambiguous_query_never_reaches_retrieval_or_generation(real_rag_pipeline):
    fake_gemini = _FakeGeminiClient(response_text="should never be returned")
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    result = pipeline.answer("Why did my payment fail?")
    assert result["status"] == "CLARIFY"
    assert result["sources"] == []
    assert fake_gemini.last_call is None


def test_supported_query_with_evidence_calls_gemini_and_returns_grounded_answer(real_rag_pipeline):
    fake_gemini = _FakeGeminiClient(response_text="Test mode allows up to 30 payment links per business.")
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    # Verified against the real, trained classifier to actually reach PROCEED with
    # retrievable evidence -- see the Phase 4 report for why many naturally-phrased
    # questions instead reach CLARIFY at the current confidence thresholds.
    result = pipeline.answer("The customer says they paid. Can I check the transaction status? for a customer")
    assert result["status"] == "ANSWERED"
    assert result["answer"] == "Test mode allows up to 30 payment links per business."
    assert result["retriever_used"] == "semantic"
    assert result["sources"]
    for source in result["sources"]:
        assert source["source_id"]
        assert source["url"]


def test_grounded_prompt_actually_contains_the_retrieved_evidence_text(real_rag_pipeline):
    fake_gemini = _FakeGeminiClient(response_text="answer")
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    result = pipeline.answer("The customer says they paid. Can I check the transaction status? for a customer")
    assert result["status"] == "ANSWERED"
    _, user_prompt = fake_gemini.last_call
    assert "RETRIEVED RAZORPAY DOCUMENTATION" in user_prompt
    assert "USER QUESTION" in user_prompt
    # At least one retrieved source's own text must actually appear in the prompt sent to Gemini.
    assert any(source["text"] in user_prompt for source in result["sources"])


def test_query_with_no_retrieval_evidence_never_calls_gemini(real_rag_pipeline):
    fake_gemini = _FakeGeminiClient(response_text="should never be returned")
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    # Verified: reaches PROCEED (real classifier) but has no evidence in this narrow
    # 7-document corpus (refund status is not one of the ingested topics).
    result = pipeline.answer("Where can I see the current state of refund rfnd_DEF456? from my backend")
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["sources"] == []
    assert fake_gemini.last_call is None


def test_missing_api_key_uses_local_grounded_fallback_but_keeps_retrieved_sources(real_rag_pipeline):
    # Consistent with tests/test_answer_api.py's endpoint-level coverage of the
    # same scenario: a missing API key falls back to the evidence-gated local
    # renderer rather than surfacing a bare "unavailable" status with no answer.
    fake_gemini = _FakeGeminiClient(exception=MissingAPIKeyError("no key set"))
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    result = pipeline.answer("The customer says they paid. Can I check the transaction status? for a customer")
    assert result["status"] == "ANSWERED"
    assert result["answer"]
    assert result["reason"] == "local_grounded_fallback"
    assert result["sources"]


def test_gemini_failure_returns_generation_failed_but_keeps_retrieved_sources(real_rag_pipeline):
    fake_gemini = _FakeGeminiClient(exception=GenerationFailedError("network error"))
    pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
    result = pipeline.answer("The customer says they paid. Can I check the transaction status? for a customer")
    assert result["status"] == "GENERATION_FAILED"
    assert result["answer"] is None
    assert "network error" in result["reason"]
    assert result["sources"]


def test_falls_back_to_tfidf_when_semantic_retriever_unavailable(real_rag_pipeline):
    original_semantic = real_rag_pipeline.semantic_retriever
    real_rag_pipeline.semantic_retriever = None
    try:
        fake_gemini = _FakeGeminiClient(response_text="answer")
        pipeline = AnswerPipeline(real_rag_pipeline, fake_gemini)
        result = pipeline.answer("The customer says they paid. Can I check the transaction status? for a customer")
        assert result["retriever_used"] == "tfidf"
    finally:
        real_rag_pipeline.semantic_retriever = original_semantic
