# Razorpay AI Agent — integrated final submission

This is one runnable Razorpay AI Agent application. It retains the Part 1 Razorpay-like chat experience and connects it directly to the Part 2 FastAPI service. The browser **does not call Gemini** and no Gemini key is required in the frontend.

## Architecture

```text
Part 1 Razorpay-like chat UI (frontend/public)
        -> POST /v1/rag/answer {text, conversation_id}
Part 2 FastAPI application
        -> conversation context -> NLU/confidence routing
        -> semantic RAG (TF-IDF fallback) -> evidence gate
        -> optional grounded Gemini generation -> answer + provenance
        -> chat UI, including expandable source links
```

The backend is the single source of truth for NLU, retrieval, generation, conversation memory, provenance, and action authorization. The former Part 1 Express server, direct Gemini browser flow, and Node dependencies are not part of this submission.

## Included behavior

- Part 1 UI, styling, chat interactions, and Razorpay-like presentation live in `frontend/public` and are served by FastAPI at `http://127.0.0.1:8000/`.
- Domain and intent classification, rule-based entity extraction, and confidence routing return `PROCEED`, `CLARIFY`, or `OUT_OF_DOMAIN` as appropriate.
- Semantic retrieval uses the included local MiniLM model and persisted index; TF-IDF is retained as its retrieval fallback. Returned sources carry real provenance metadata and are displayed in the UI.
- Answers are generated only when the confidence and evidence gates pass. No evidence, ambiguous requests, and out-of-domain requests abstain safely.
- A per-browser-session `conversation_id` is sent with every message. The API stores a bounded, in-memory six-turn context; it is lost on restart and is not a user-data store.
- `POST /v1/actions/authorize` still enforces its existing allowlist, confirmations, amount/count limits, and server-side audit logging. The UI does not bypass it or receive audit files/secrets.

The supplied corpus is intentionally small. Questions beyond `data/knowledge/` are expected to return an insufficient-evidence response.

## Install and run

Python 3.11+ is recommended. The app is self-contained: starting FastAPI also serves the frontend, so a Node installation is not required.

```powershell
cd Razorpay-AI-Agent-FINAL-INTEGRATED
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn razorpay_ai.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, ask a documentation-backed question, then ask a follow-up such as “What about the expiration?” in the same page. The browser keeps the same `conversation_id` automatically.

### Separately hosting the static frontend (optional)

The default `frontend/public/config.js` uses an empty `apiBaseUrl`, meaning the same FastAPI origin. If the static UI is served elsewhere, copy `frontend/public/config.example.js` over `config.js` (or set its `apiBaseUrl`) to the backend origin. Start the backend with the corresponding browser origin:

```powershell
$env:CORS_ALLOW_ORIGINS = "http://127.0.0.1:3000"
.\.venv\Scripts\python -m uvicorn razorpay_ai.api:app --app-dir src --host 127.0.0.1 --port 8000
```

`CORS_ALLOW_ORIGINS` accepts a comma-separated allowlist. It is unnecessary when the supplied UI is served from FastAPI itself.

## API endpoints

- `GET /health` — service health.
- `POST /v1/nlu/analyze` — `{"text":"..."}` for NLU/routing only.
- `POST /v1/rag/retrieve` — `{"text":"...", "retriever":"semantic"}`; use `tfidf` to request the other retriever.
- `POST /v1/rag/answer` — `{"text":"...", "conversation_id":"optional-id"}`; returns an answer status, answer text when available, provenance sources, and `context_applied`.
- `POST /v1/actions/authorize` — `{"action":"CREATE_ORDER", "amount_paise":1000, "confirmed":true, "conversation_id":"optional-id"}`. This only authorizes; it never executes a provider-side payment action.

## Optional Gemini generation

Gemini is an optional, backend-only generation provider. To enable it, set the key in the terminal that launches FastAPI:

```powershell
$env:GEMINI_API_KEY = "your-key"
```

The API key is read by the server only, never sent to the frontend or written to project files. Without it, the system uses its built-in evidence-only local fallback: it renders the highest-ranked retrieved documentation excerpt and returns its real provenance. It does not invent an answer or make a browser-side model call.

## Verification

Run the focused integration test plus the existing answer API coverage:

```powershell
.\.venv\Scripts\python -m pytest -q tests/test_frontend_integration.py tests/test_answer_api.py
```

For a manual integration check, start the server, load the UI, submit a knowledge-backed question, check the Sources disclosure, submit a follow-up in the same tab, and try an unrelated question to confirm its guardrail response.

## Known limitations

- Conversation memory is process-local and bounded; do not use it for durable customer history or across multiple worker processes.
- Semantic-retrieval thresholds and the small corpus are evaluation artifacts, not production coverage claims.
- Live Gemini generation depends on a valid API key and reachable Google API. Retrieval and all non-generation guardrails work without it.
- Audit records are runtime operational data at `data/audit_log.jsonl`; they are excluded from source control and the final archive.
