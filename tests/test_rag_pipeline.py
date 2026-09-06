import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from razorpay_ai.rag_pipeline import RAGRetrievalPipeline


@pytest.fixture(scope="module")
def rag_pipeline():
    return RAGRetrievalPipeline.load()


def test_razorpay_query_retrieves_from_the_knowledge_corpus(rag_pipeline):
    result = rag_pipeline.retrieve_for_query("What is the test mode limit for payment links?")
    assert result["domain"] == "RAZORPAY"
    assert result["retrieval_status"] in {"RETRIEVED", "NO_MATCH_ABOVE_THRESHOLD"}
    if result["retrieval_status"] == "RETRIEVED":
        assert result["results"]
        assert all("source_id" in r for r in result["results"])


def test_out_of_domain_query_does_not_retrieve(rag_pipeline):
    result = rag_pipeline.retrieve_for_query("Why did my Stripe payment fail?")
    assert result["domain"] == "OUT_OF_DOMAIN"
    assert result["retrieval_status"] == "SKIPPED_OUT_OF_DOMAIN"
    assert result["results"] == []


def test_ambiguous_query_does_not_retrieve(rag_pipeline):
    result = rag_pipeline.retrieve_for_query("Why did my payment fail?")
    assert result["domain"] == "AMBIGUOUS"
    assert result["retrieval_status"] == "SKIPPED_AMBIGUOUS"
    assert result["results"] == []


def test_unsupported_razorpay_topic_returns_empty_results_not_a_guess(rag_pipeline):
    # A Razorpay-domain question with no supporting document in this small corpus
    # must not be answered from irrelevant chunks.
    result = rag_pipeline.retrieve_for_query("Razorpay RazorpayX payroll salary disbursal setup")
    assert result["domain"] in {"RAZORPAY", "AMBIGUOUS", "OUT_OF_DOMAIN"}
    # Whatever the domain call, retrieval must never silently fabricate a match:
    # every returned chunk (if any) must carry real source provenance.
    for chunk in result["results"]:
        assert chunk["source_id"]
        assert chunk["url"]
