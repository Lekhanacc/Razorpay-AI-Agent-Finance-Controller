"""FastAPI boundary for the persisted Razorpay NLU pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .answer_pipeline import AnswerPipeline
from .actions import ActionGuard, AuditLogger
from .confidence import evaluate_confidence
from .config import AUDIT_LOG_PATH, CONVERSATION_MEMORY_MAX_TURNS, MAX_ACTION_AMOUNT_PAISE, MAX_ACTIONS_PER_CONVERSATION
from .conversation import ConversationMemory
from .generation import GeminiClient
from .pipeline import RazorpayAIPipeline
from .rag_pipeline import RAGRetrievalPipeline
from .retrieval import KnowledgeRetriever
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnswerRequest,
    AnswerResponse,
    ActionAuthorizationRequest,
    ActionAuthorizationResponse,
    HealthResponse,
    RAGRetrieveRequest,
    RAGRetrieveResponse,
)
from .semantic_retrieval import SemanticRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted models once when the service starts.

    Both retrievers are loaded best-effort: if the TF-IDF index or the semantic
    index/model has not been built yet, the existing NLU endpoints must still
    start and work. Only the affected retriever mode of /v1/rag/retrieve is
    unavailable in that case (returns 503, not a startup crash).
    """
    app.state.pipeline = RazorpayAIPipeline.load()
    app.state.conversation_memory = ConversationMemory(CONVERSATION_MEMORY_MAX_TURNS)
    app.state.action_guard = ActionGuard(
        AuditLogger(AUDIT_LOG_PATH), MAX_ACTION_AMOUNT_PAISE, MAX_ACTIONS_PER_CONVERSATION
    )

    try:
        tfidf_retriever = KnowledgeRetriever.load()
    except FileNotFoundError:
        tfidf_retriever = None

    try:
        semantic_retriever = SemanticRetriever.load()
    except (FileNotFoundError, ModuleNotFoundError):
        # Keep the service usable with its persisted TF-IDF fallback when the
        # optional local semantic runtime is unavailable. A normal install from
        # requirements.txt provides sentence-transformers; this also makes the
        # documented best-effort startup behavior true for minimal deployments.
        semantic_retriever = None

    if tfidf_retriever is None:
        app.state.rag_pipeline = None
        app.state.answer_pipeline = None
    else:
        app.state.rag_pipeline = RAGRetrievalPipeline(app.state.pipeline, tfidf_retriever, semantic_retriever)
        # GeminiClient itself never raises at construction time; the missing-API-key
        # case is handled lazily, per-request, inside AnswerPipeline.answer().
        app.state.answer_pipeline = AnswerPipeline(app.state.rag_pipeline, GeminiClient())
    yield


app = FastAPI(
    title="Razorpay AI NLU Service",
    version="1.0.0",
    lifespan=lifespan,
)

# The interface is served by this application for the simplest local setup, but
# it may also be hosted separately (for example with ``python -m http.server``).
# Keep the permitted browser origins configurable rather than coupling API
# access to a particular development port.
_default_cors_origins = "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:5173,http://localhost:5173"
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", service="razorpay-ai-nlu")


@app.post(
    "/v1/nlu/analyze",
    response_model=AnalyzeResponse,
    response_model_exclude_none=True,
)
def analyze(payload: AnalyzeRequest, request: Request) -> dict:
    """Analyze text using the loaded domain, intent, and entity pipeline."""
    result = request.app.state.pipeline.predict(payload.text)
    result["confidence_decision"] = evaluate_confidence(result).value
    return result


@app.post("/v1/rag/retrieve", response_model=RAGRetrieveResponse)
def rag_retrieve(payload: RAGRetrieveRequest, request: Request) -> dict:
    """Retrieval-only endpoint: domain-gated knowledge lookup, no answer generation.

    `retriever` selects the backend: "tfidf" (default, Phase 2) or "semantic"
    (Phase 3, local sentence-transformers embeddings).
    """
    rag_pipeline = request.app.state.rag_pipeline
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Knowledge index not built. Run scripts/ingest_knowledge.py first.")
    if payload.retriever == "semantic" and rag_pipeline.semantic_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic retriever is not available. Run scripts/build_semantic_index.py first.",
        )
    return rag_pipeline.retrieve_for_query(payload.text, retriever=payload.retriever)


@app.post("/v1/rag/answer", response_model=AnswerResponse)
def rag_answer(payload: AnswerRequest, request: Request) -> dict:
    """Grounded answer generation: NLU -> confidence -> retrieval -> Gemini.

    Only calls Gemini when retrieval evidence is available and the confidence
    policy says PROCEED. Never answers from unsupported knowledge.
    """
    answer_pipeline = request.app.state.answer_pipeline
    if answer_pipeline is None:
        raise HTTPException(status_code=503, detail="Knowledge index not built. Run scripts/ingest_knowledge.py first.")
    resolved_text, context_applied = request.app.state.conversation_memory.contextualize(
        payload.conversation_id, payload.text
    )
    result = answer_pipeline.answer(resolved_text)
    # The external contract always reflects the user's original wording, rather
    # than exposing the internal query expansion sent to NLU/retrieval.
    result["query"] = payload.text
    result["conversation_id"] = payload.conversation_id
    result["context_applied"] = context_applied
    request.app.state.conversation_memory.record(payload.conversation_id, payload.text, result)
    return result


@app.post("/v1/actions/authorize", response_model=ActionAuthorizationResponse)
def authorize_action(payload: ActionAuthorizationRequest, request: Request) -> dict:
    """Apply action guardrails; this endpoint never performs a provider-side action."""
    return request.app.state.action_guard.authorize(
        payload.action, payload.amount_paise, payload.confirmed, payload.conversation_id
    )


# The Part 1 Razorpay-like UI is intentionally a static, dependency-free
# frontend. The explicit asset routes keep the page portable: the same files
# also work when frontend/public is hosted by a separate static server.
# This removes the previous Express endpoint that called Gemini from the
# browser-facing project.
_frontend_dir = Path(__file__).resolve().parents[2] / "frontend" / "public"
if _frontend_dir.is_dir():
    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(_frontend_dir / "index.html")

    @app.get("/style.css", include_in_schema=False)
    def frontend_styles() -> FileResponse:
        return FileResponse(_frontend_dir / "style.css", media_type="text/css")

    @app.get("/config.js", include_in_schema=False)
    def frontend_config() -> FileResponse:
        return FileResponse(_frontend_dir / "config.js", media_type="application/javascript")

    @app.get("/script.js", include_in_schema=False)
    def frontend_script() -> FileResponse:
        return FileResponse(_frontend_dir / "script.js", media_type="application/javascript")
