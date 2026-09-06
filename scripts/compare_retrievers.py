"""Compare TF-IDF (Phase 2) vs semantic embedding (Phase 3) retrieval.

Uses the SAME data/retrieval_eval.jsonl queries for both systems -- no queries are
added, removed, or reworded to favor either retriever. Reports ranking accuracy
(unfiltered), deployed-threshold accuracy, and abstention behavior on the
out-of-corpus negative queries, for both systems side by side, plus a per-query
breakdown of where each system succeeds or fails.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.config import (
    RETRIEVAL_EVAL_PATH,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from razorpay_ai.retrieval import KnowledgeRetriever
from razorpay_ai.semantic_retrieval import SemanticRetriever


def load_eval_set(path: Path = RETRIEVAL_EVAL_PATH) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_system(retriever, eval_rows: list[dict], deployed_threshold: float, system_name: str) -> dict:
    in_corpus = [row for row in eval_rows if row["expected_source_id"] is not None]
    not_in_corpus = [row for row in eval_rows if row["expected_source_id"] is None]

    top1_hits = top3_hits = top5_hits = 0
    top1_hits_at_threshold = top3_hits_at_threshold = 0
    per_query = []

    query_latencies = []
    for row in in_corpus:
        acceptable = set(row["acceptable_source_ids"])

        start = time.perf_counter()
        ranked = retriever.retrieve(row["query"], top_k=5, similarity_threshold=0.0)
        query_latencies.append(time.perf_counter() - start)

        ranked_ids = [r.source_id for r in ranked]
        hit1 = bool(ranked_ids[:1]) and ranked_ids[0] in acceptable
        hit3 = any(sid in acceptable for sid in ranked_ids[:3])
        hit5 = any(sid in acceptable for sid in ranked_ids[:5])
        top1_hits += int(hit1)
        top3_hits += int(hit3)
        top5_hits += int(hit5)

        ranked_at_threshold = retriever.retrieve(row["query"], top_k=5, similarity_threshold=deployed_threshold)
        ids_at_threshold = [r.source_id for r in ranked_at_threshold]
        hit1_t = bool(ids_at_threshold[:1]) and ids_at_threshold[0] in acceptable
        hit3_t = any(sid in acceptable for sid in ids_at_threshold[:3])
        top1_hits_at_threshold += int(hit1_t)
        top3_hits_at_threshold += int(hit3_t)

        per_query.append(
            {
                "query": row["query"],
                "expected": row["expected_source_id"],
                "top5_unfiltered": [(r.source_id, round(r.score, 3)) for r in ranked],
                "hit@1": hit1,
                "hit@3": hit3,
                "hit@5": hit5,
                "hit@1_at_threshold": hit1_t,
            }
        )

    n = len(in_corpus)
    correct_abstentions = 0
    abstention_detail = []
    for row in not_in_corpus:
        ranked = retriever.retrieve(row["query"], top_k=5, similarity_threshold=deployed_threshold)
        abstained = len(ranked) == 0
        correct_abstentions += int(abstained)
        abstention_detail.append(
            {"query": row["query"], "abstained": abstained, "returned": [(r.source_id, round(r.score, 3)) for r in ranked]}
        )

    return {
        "system": system_name,
        "top1_unfiltered": round(top1_hits / n, 3) if n else 0.0,
        "top3_unfiltered": round(top3_hits / n, 3) if n else 0.0,
        "top5_unfiltered": round(top5_hits / n, 3) if n else 0.0,
        "top1_at_deployed_threshold": round(top1_hits_at_threshold / n, 3) if n else 0.0,
        "top3_at_deployed_threshold": round(top3_hits_at_threshold / n, 3) if n else 0.0,
        "deployed_threshold": deployed_threshold,
        "abstention_rate": round(correct_abstentions / len(not_in_corpus), 3) if not_in_corpus else None,
        "mean_query_latency_ms": round(1000 * sum(query_latencies) / len(query_latencies), 2) if query_latencies else None,
        "per_query": per_query,
        "abstention_detail": abstention_detail,
    }


def main() -> None:
    eval_rows = load_eval_set()

    tfidf_retriever = KnowledgeRetriever.load()
    semantic_retriever = SemanticRetriever.load()

    tfidf_report = evaluate_system(tfidf_retriever, eval_rows, RETRIEVAL_SIMILARITY_THRESHOLD, "tfidf")
    semantic_report = evaluate_system(semantic_retriever, eval_rows, SEMANTIC_SIMILARITY_THRESHOLD, "semantic")

    comparison = {
        "top1_unfiltered_delta": round(semantic_report["top1_unfiltered"] - tfidf_report["top1_unfiltered"], 3),
        "top3_unfiltered_delta": round(semantic_report["top3_unfiltered"] - tfidf_report["top3_unfiltered"], 3),
        "top5_unfiltered_delta": round(semantic_report["top5_unfiltered"] - tfidf_report["top5_unfiltered"], 3),
    }

    both_succeed, both_fail, semantic_only, tfidf_only = [], [], [], []
    for t_row, s_row in zip(tfidf_report["per_query"], semantic_report["per_query"]):
        assert t_row["query"] == s_row["query"]
        t_ok, s_ok = t_row["hit@3"], s_row["hit@3"]
        entry = {"query": t_row["query"], "expected": t_row["expected"]}
        if t_ok and s_ok:
            both_succeed.append(entry)
        elif not t_ok and not s_ok:
            both_fail.append(entry)
        elif s_ok and not t_ok:
            semantic_only.append(entry)
        else:
            tfidf_only.append(entry)

    print(
        json.dumps(
            {
                "tfidf_summary": {k: v for k, v in tfidf_report.items() if k not in ("per_query", "abstention_detail")},
                "semantic_summary": {k: v for k, v in semantic_report.items() if k not in ("per_query", "abstention_detail")},
                "comparison_deltas_semantic_minus_tfidf": comparison,
                "both_succeed_at_top3": both_succeed,
                "both_fail_at_top3": both_fail,
                "semantic_succeeds_tfidf_fails": semantic_only,
                "tfidf_succeeds_semantic_fails": tfidf_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
