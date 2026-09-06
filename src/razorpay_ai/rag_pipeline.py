"""Composes the existing NLU pipeline with conditional knowledge retrieval.

This module does not modify RazorpayAIPipeline. It calls it as-is and applies a
routing rule on top of its output:

  domain == RAZORPAY      -> retrieve from the knowledge corpus
  domain == AMBIGUOUS     -> do NOT retrieve; clarification is required first
  domain == OUT_OF_DOMAIN -> do NOT retrieve from the Razorpay corpus

Retrieval-only: this does not generate an answer. That is a later phase.

Phase 3 adds a second retrieval backend (semantic embeddings) alongside the
Phase 2 TF-IDF backend. The caller selects which one to use; TF-IDF remains the
default so existing callers see identical behavior unless they opt in.
"""

from __future__ import annotations

from .config import RETRIEVAL_SIMILARITY_THRESHOLD, RETRIEVAL_TOP_K, SEMANTIC_SIMILARITY_THRESHOLD, SEMANTIC_TOP_K
from .pipeline import RazorpayAIPipeline
from .retrieval import KnowledgeRetriever
from .semantic_retrieval import SemanticRetriever


class RAGRetrievalPipeline:
    def __init__(
        self,
        nlu_pipeline: RazorpayAIPipeline,
        tfidf_retriever: KnowledgeRetriever,
        semantic_retriever: SemanticRetriever | None = None,
    ):
        self.nlu_pipeline = nlu_pipeline
        self.tfidf_retriever = tfidf_retriever
        self.semantic_retriever = semantic_retriever

    @classmethod
    def load(cls) -> "RAGRetrievalPipeline":
        try:
            semantic_retriever = SemanticRetriever.load()
        except FileNotFoundError:
            semantic_retriever = None
        return cls(RazorpayAIPipeline.load(), KnowledgeRetriever.load(), semantic_retriever)

    def retrieve_for_query(
        self,
        text: str,
        retriever: str = "tfidf",
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> dict:
        if retriever not in ("tfidf", "semantic"):
            raise ValueError(f"Unknown retriever '{retriever}'. Expected 'tfidf' or 'semantic'.")
        if retriever == "semantic" and self.semantic_retriever is None:
            raise RuntimeError("Semantic retriever is not available (index not built or model not loaded).")

        nlu_result = self.nlu_pipeline.predict(text)
        domain = nlu_result["domain"]

        if domain == "OUT_OF_DOMAIN":
            return {
                "query": text,
                "domain": domain,
                "retriever": retriever,
                "retrieval_status": "SKIPPED_OUT_OF_DOMAIN",
                "results": [],
            }
        if domain == "AMBIGUOUS":
            return {
                "query": text,
                "domain": domain,
                "retriever": retriever,
                "retrieval_status": "SKIPPED_AMBIGUOUS",
                "results": [],
            }

        if retriever == "tfidf":
            backend = self.tfidf_retriever
            default_top_k = RETRIEVAL_TOP_K
            default_threshold = RETRIEVAL_SIMILARITY_THRESHOLD
        else:
            backend = self.semantic_retriever
            default_top_k = SEMANTIC_TOP_K
            default_threshold = SEMANTIC_SIMILARITY_THRESHOLD

        retrieved = backend.retrieve(
            text,
            top_k=top_k if top_k is not None else default_top_k,
            similarity_threshold=similarity_threshold if similarity_threshold is not None else default_threshold,
        )
        retrieval_status = "RETRIEVED" if retrieved else "NO_MATCH_ABOVE_THRESHOLD"
        return {
            "query": text,
            "domain": domain,
            "retriever": retriever,
            "retrieval_status": retrieval_status,
            "results": [chunk.to_dict() for chunk in retrieved],
        }
