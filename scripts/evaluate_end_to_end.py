"""Run measurable end-to-end routing, evidence, context, and latency checks.

Generation quality is intentionally not scored when no live Gemini key is
available: mocked output is useful for unit tests, not a real quality metric.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.answer_pipeline import AnswerPipeline
from razorpay_ai.conversation import ConversationMemory
from razorpay_ai.generation import GeminiClient
from razorpay_ai.rag_pipeline import RAGRetrievalPipeline

EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "end_to_end_eval.jsonl"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    pipeline = AnswerPipeline(RAGRetrievalPipeline.load(), GeminiClient())
    memory = ConversationMemory()
    records = []
    for case in load_cases():
        query, context_applied = memory.contextualize(case.get("conversation_id"), case["text"])
        started = time.perf_counter()
        result = pipeline.answer(query)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        result["query"] = case["text"]
        memory.record(case.get("conversation_id"), case["text"], result)
        records.append({
            "id": case["id"], "category": case["category"], "status": result["status"],
            "source_count": len(result["sources"]), "context_applied": context_applied,
            "latency_ms": latency_ms, "expected_status": case.get("expected_status"),
            "status_matches_expectation": result["status"] == case["expected_status"] if case.get("expected_status") else None,
            "context_matches_expectation": context_applied == case["expects_context"] if "expects_context" in case else None,
        })
    checked_statuses = [r["status_matches_expectation"] for r in records if r["status_matches_expectation"] is not None]
    checked_context = [r["context_matches_expectation"] for r in records if r["context_matches_expectation"] is not None]
    report = {
        "case_count": len(records),
        "status_expectation_accuracy": round(sum(checked_statuses) / len(checked_statuses), 3) if checked_statuses else None,
        "context_expectation_accuracy": round(sum(checked_context) / len(checked_context), 3) if checked_context else None,
        "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 2),
        "generation_quality": "not_scored: requires a live Gemini response and human or reference-answer judgement",
        "records": records,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
