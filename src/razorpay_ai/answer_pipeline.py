"""Grounded RAG answer generation.

Pipeline (matches the architecture given for Phase 4):

  NLU predict -> evaluate_confidence (PROCEED / CLARIFY / OUT_OF_DOMAIN)
    -> OUT_OF_DOMAIN: reject, no retrieval, no generation
    -> CLARIFY: ask for clarification, no retrieval, no generation
    -> PROCEED: retrieve evidence (semantic primary, TF-IDF fallback)
         -> no evidence above threshold: return INSUFFICIENT_EVIDENCE, no Gemini call
         -> evidence found: build a grounded prompt, call Gemini, attach real
            provenance from the retrieved chunks (not LLM-generated citations)

This module does not modify RAGRetrievalPipeline, RazorpayAIPipeline, or the
confidence policy -- it composes them.
"""

from __future__ import annotations

from .confidence import ConfidenceDecision, evaluate_confidence
from .config import ANSWER_FALLBACK_RETRIEVER, ANSWER_PRIMARY_RETRIEVER
from .generation import GeminiClient, GenerationFailedError, MissingAPIKeyError
from .rag_pipeline import RAGRetrievalPipeline

SYSTEM_INSTRUCTION = """You are a Razorpay support assistant. You answer ONLY using the \
"RETRIEVED RAZORPAY DOCUMENTATION" provided in the user message below.

Rules you must follow strictly:
1. Do not use any knowledge about Razorpay, payments, or APIs beyond what is in the \
provided documentation excerpts, even if you believe you know the answer.
2. If the provided excerpts do not contain enough information to answer the \
question, say so plainly instead of guessing or filling gaps with assumptions.
3. Do not invent policy details, numbers, field names, error codes, or timelines \
that are not explicitly present in the excerpts.
4. Keep the answer concise and directly address the user's question.
5. Where useful, mention which part of the documentation supports your answer, but \
do not fabricate a source that was not given to you."""


OUT_OF_DOMAIN_MESSAGE = "I'm designed to help with Razorpay and payment-related questions. That looks outside my scope."
CLARIFY_MESSAGE = "I'm not fully sure what you're asking about. Could you clarify -- for example, which Razorpay product or action this concerns?"
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have verified Razorpay documentation covering this specific question in my current knowledge base, "
    "so I don't want to guess. Could you rephrase, or point me to the specific Razorpay product this concerns?"
)


def _local_grounded_fallback(chunks: list[dict]) -> str:
    """Produce a safe, extractive answer when optional remote generation is unavailable.

    This is deliberately not another model or retriever: it renders the highest
    ranked, already evidence-gated documentation excerpt with its real title.
    It keeps the UI useful without a Gemini key while preserving the same NLU,
    confidence, retrieval, and provenance boundaries as generated answers.
    """
    best = chunks[0]
    excerpt = " ".join(best["text"].split())
    return f"According to Razorpay documentation — {best['title']}: {excerpt}"


def _build_user_prompt(query: str, chunks: list[dict]) -> str:
    evidence_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        evidence_blocks.append(f"[Source {i}: {chunk['title']} ({chunk['url']})]\n{chunk['text']}")
    evidence_text = "\n\n".join(evidence_blocks)
    return f"RETRIEVED RAZORPAY DOCUMENTATION:\n\n{evidence_text}\n\nUSER QUESTION:\n{query}"


class AnswerPipeline:
    def __init__(self, rag_pipeline: RAGRetrievalPipeline, gemini_client: GeminiClient):
        self.rag_pipeline = rag_pipeline
        self.gemini_client = gemini_client

    @classmethod
    def load(cls) -> "AnswerPipeline":
        return cls(RAGRetrievalPipeline.load(), GeminiClient())

    def _select_retriever(self) -> tuple[object, str]:
        """Return (retriever, name_used), preferring the primary, falling back if unavailable."""
        primary = getattr(self.rag_pipeline, f"{ANSWER_PRIMARY_RETRIEVER}_retriever", None)
        if primary is not None:
            return primary, ANSWER_PRIMARY_RETRIEVER
        fallback = getattr(self.rag_pipeline, f"{ANSWER_FALLBACK_RETRIEVER}_retriever", None)
        if fallback is not None:
            return fallback, ANSWER_FALLBACK_RETRIEVER
        return None, "none"

    def answer(self, text: str) -> dict:
        nlu_result = self.rag_pipeline.nlu_pipeline.predict(text)
        decision = evaluate_confidence(nlu_result)

        if decision == ConfidenceDecision.OUT_OF_DOMAIN:
            return {
                "query": text,
                "status": "OUT_OF_DOMAIN",
                "answer": OUT_OF_DOMAIN_MESSAGE,
                "retriever_used": None,
                "sources": [],
            }
        if decision == ConfidenceDecision.CLARIFY and nlu_result["domain"] != "RAZORPAY":
            return {
                "query": text,
                "status": "CLARIFY",
                "answer": CLARIFY_MESSAGE,
                "retriever_used": None,
                "sources": [],
            }

        retriever, retriever_name = self._select_retriever()
        if retriever is None:
            return {
                "query": text,
                "status": "RETRIEVAL_UNAVAILABLE",
                "answer": None,
                "retriever_used": None,
                "sources": [],
            }

        retrieved = retriever.retrieve(text)
        if not retrieved:
            if decision == ConfidenceDecision.CLARIFY:
                return {
                    "query": text,
                    "status": "CLARIFY",
                    "answer": CLARIFY_MESSAGE,
                    "retriever_used": retriever_name,
                    "sources": [],
                }
            return {
                "query": text,
                "status": "INSUFFICIENT_EVIDENCE",
                "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
                "retriever_used": retriever_name,
                "sources": [],
            }

        chunk_dicts = [chunk.to_dict() for chunk in retrieved]
        user_prompt = _build_user_prompt(text, chunk_dicts)

        try:
            generated_text = self.gemini_client.generate(SYSTEM_INSTRUCTION, user_prompt)
        except MissingAPIKeyError:
            return {
                "query": text,
                "status": "ANSWERED",
                "answer": _local_grounded_fallback(chunk_dicts),
                "retriever_used": retriever_name,
                "sources": chunk_dicts,
                "reason": "local_grounded_fallback",
            }
        except GenerationFailedError as exc:
            return {
                "query": text,
                "status": "GENERATION_FAILED",
                "answer": None,
                "retriever_used": retriever_name,
                "sources": chunk_dicts,
                "reason": str(exc),
            }

        return {
            "query": text,
            "status": "ANSWERED",
            "answer": generated_text,
            "retriever_used": retriever_name,
            "sources": chunk_dicts,
        }
