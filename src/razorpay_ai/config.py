"""Project paths and reproducibility settings."""

from pathlib import Path

RANDOM_SEED = 42
DOMAIN_CLASS_WEIGHT_MAX = 10.0
# Provisional operational policy thresholds, not calibrated probabilities.
DOMAIN_CONFIDENCE_THRESHOLD = 0.80
INTENT_CONFIDENCE_THRESHOLD = 0.40
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "dataset.csv"
TAXONOMY_PATH = DATA_DIR / "taxonomy_v1.json"
DOMAIN_MODEL_PATH = MODELS_DIR / "domain_model.joblib"
INTENT_MODEL_PATH = MODELS_DIR / "intent_model.joblib"

REQUIRED_COLUMNS = ("id", "text", "domain", "product", "intent", "operation", "entities")
DOMAIN_LABEL_MAP = {
    "razorpay": "RAZORPAY",
    "ambiguous": "AMBIGUOUS",
    "out_of_domain": "OUT_OF_DOMAIN",
}
ENTITY_EXTRACTION_STATUS = "RULE_BASED_V1"

# --- Phase 2: RAG V1 (TF-IDF retrieval) settings ---
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
KNOWLEDGE_MANIFEST_PATH = DATA_DIR / "knowledge_manifest.jsonl"
KNOWLEDGE_INDEX_DIR = DATA_DIR / "knowledge_index"
CHUNKS_PATH = KNOWLEDGE_INDEX_DIR / "chunks.jsonl"
VECTORIZER_PATH = KNOWLEDGE_INDEX_DIR / "vectorizer.joblib"
MATRIX_PATH = KNOWLEDGE_INDEX_DIR / "matrix.joblib"
INDEX_MANIFEST_PATH = KNOWLEDGE_INDEX_DIR / "index_manifest.json"
RETRIEVAL_EVAL_PATH = DATA_DIR / "retrieval_eval.jsonl"

# Deterministic chunking configuration (word-based, not token-based).
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30
MIN_CHUNK_WORDS = 20

# Provisional retrieval policy settings, not calibrated against a large eval set.
# Tuned against the small evaluation set in data/retrieval_eval.jsonl; revisit as the
# corpus grows. Mirrors the "provisional, not calibrated" stance already used for the
# domain/intent confidence thresholds in this project.
RETRIEVAL_TOP_K = 5
RETRIEVAL_SIMILARITY_THRESHOLD = 0.12

# --- Phase 3: Semantic Retrieval V2 (local sentence-transformers embeddings) ---
# Model files were provided locally by the user (see README/Phase 3 report for
# provenance) because this environment's network egress does not allow
# huggingface.co. Loaded fully offline; no network calls are made.
SEMANTIC_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_EMBEDDING_DIMENSION = 384

SEMANTIC_INDEX_DIR = DATA_DIR / "knowledge_index_semantic"
SEMANTIC_CHUNKS_PATH = SEMANTIC_INDEX_DIR / "chunks.jsonl"
SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_INDEX_DIR / "embeddings.npy"
SEMANTIC_INDEX_MANIFEST_PATH = SEMANTIC_INDEX_DIR / "index_manifest.json"

SEMANTIC_TOP_K = 5
# Cosine similarity threshold for normalized sentence embeddings. Provisional,
# evaluated (not blindly tuned) against data/retrieval_eval.jsonl -- see the
# Phase 3 report for the actual measured trade-offs at this value.
SEMANTIC_SIMILARITY_THRESHOLD = 0.35

# --- Phase 4: Gemini-powered RAG answer generation ---
# Verified in this environment: generativelanguage.googleapis.com returns
# HTTP 403 (x-deny-reason: host_not_allowed) -- the same class of network
# restriction documented for huggingface.co in Phase 3. The client below is a
# real implementation of the actual Gemini REST contract; live end-to-end
# generation has not been verified from this sandbox. See the Phase 4 report.
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
# Verified with a real generateContent call on 2026-09-05. The previous
# gemini-2.0-flash and versioned aliases returned 404 for this API key.
GEMINI_MODEL_NAME = "gemini-flash-latest"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_REQUEST_TIMEOUT_SECONDS = 20.0
GEMINI_TRANSIENT_RETRY_COUNT = 2

# Retriever used as the primary evidence source for generation, with automatic
# fallback to the other if the primary backend is unavailable.
ANSWER_PRIMARY_RETRIEVER = "semantic"
ANSWER_FALLBACK_RETRIEVER = "tfidf"

# --- Phase 5: process-local conversational context ---
CONVERSATION_MEMORY_MAX_TURNS = 6

# --- Final production guardrails ---
MAX_ACTION_AMOUNT_PAISE = 100_000
MAX_ACTIONS_PER_CONVERSATION = 3
AUDIT_LOG_PATH = DATA_DIR / "audit_log.jsonl"
