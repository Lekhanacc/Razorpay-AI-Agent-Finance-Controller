"""HTTP request and response schemas for the NLU service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator

from .confidence import ConfidenceDecision


class AnalyzeRequest(BaseModel):
    """The text accepted by the existing NLU pipeline."""

    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be a non-empty string")
        return value


class AnalyzeResponse(BaseModel):
    """The current RazorpayAIPipeline.predict response contract."""

    text: str
    domain: Literal["RAZORPAY", "AMBIGUOUS", "OUT_OF_DOMAIN"]
    domain_confidence: float
    intent: str | None
    intent_confidence: float | None
    entities: dict[str, Any]
    entity_extraction_status: str
    routing_status: str | None = None
    confidence_decision: ConfidenceDecision


class HealthResponse(BaseModel):
    """Service availability response."""

    status: Literal["healthy"]
    service: Literal["razorpay-ai-nlu"]


class RAGRetrieveRequest(BaseModel):
    """Request body for the retrieval-only RAG endpoint."""

    text: str
    retriever: Literal["tfidf", "semantic"] = "tfidf"

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be a non-empty string")
        return value


class RetrievedChunkResponse(BaseModel):
    """A single retrieved knowledge chunk with full provenance."""

    text: str
    score: float
    source_id: str
    title: str
    url: str
    product: str
    topic: str
    provenance_status: str
    chunk_id: str


class RAGRetrieveResponse(BaseModel):
    """Response body for the retrieval-only RAG endpoint."""

    query: str
    domain: Literal["RAZORPAY", "AMBIGUOUS", "OUT_OF_DOMAIN"]
    retriever: Literal["tfidf", "semantic"]
    retrieval_status: str
    results: list[RetrievedChunkResponse]


class AnswerRequest(BaseModel):
    """Request body for the grounded answer-generation endpoint."""

    text: str
    conversation_id: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be a non-empty string")
        return value

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("conversation_id must not be blank")
        return value


class AnswerResponse(BaseModel):
    """Response body for the grounded answer-generation endpoint."""

    query: str
    status: Literal[
        "ANSWERED",
        "OUT_OF_DOMAIN",
        "CLARIFY",
        "INSUFFICIENT_EVIDENCE",
        "GENERATION_UNAVAILABLE",
        "GENERATION_FAILED",
        "RETRIEVAL_UNAVAILABLE",
    ]
    answer: str | None
    retriever_used: Literal["semantic", "tfidf"] | None
    sources: list[RetrievedChunkResponse]
    reason: str | None = None
    conversation_id: str | None = None
    context_applied: bool = False


class ActionAuthorizationRequest(BaseModel):
    action: str
    amount_paise: int | None = None
    confirmed: bool = False
    conversation_id: str | None = None

    @field_validator("action")
    @classmethod
    def action_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("action must not be blank")
        return value.upper()


class ActionAuthorizationResponse(BaseModel):
    status: Literal["APPROVED", "CONFIRMATION_REQUIRED", "REJECTED_UNSUPPORTED", "REJECTED_LIMIT"]
    reason: str | None
    audit_id: str
