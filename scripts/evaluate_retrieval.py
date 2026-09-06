"""Evaluate RAG V1 retrieval against data/retrieval_eval.jsonl.

Reports, separately:
  - Ranking quality (Top-1/3/5 accuracy) for queries that ARE covered by the corpus,
    using unfiltered ranking (similarity_threshold=0.0) so ranking quality is measured
    independently of the operational threshold.
  - Abstention correctness for queries that are NOT covered by the corpus, using the
    actual operational RETRIEVAL_SIMILARITY_THRESHOLD, since abstaining IS the desired
    behavior for those and depends on the real threshold.

Does not fabricate or adjust results after the fact.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from razorpay_ai.config import RETRIEVAL_EVAL_PATH, RETRIEVAL_SIMILARITY_THRESHOLD
from razorpay_ai.retrieval import KnowledgeRetriever


def load_eval_set(path: Path = RETRIEVAL_EVAL_PATH) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    eval_rows = load_eval_set()
    retriever = KnowledgeRetriever.load()

    in_corpus = [row for row in eval_rows if row["expected_source_id"] is not None]
    not_in_corpus = [row for row in eval_rows if row["expected_source_id"] is None]

    top1_hits, top3_hits, top5_hits = 0, 0, 0
    top1_hits_at_threshold, top3_hits_at_threshold = 0, 0
    weak_cases = []
    per_query_report = []

    for row in in_corpus:
        acceptable = set(row["acceptable_source_ids"])

        # Unfiltered ranking quality (threshold disabled): measures whether the
        # TF-IDF ranking itself is good, independent of the deployed threshold.
        ranked = retriever.retrieve(row["query"], top_k=5, similarity_threshold=0.0)
        ranked_source_ids = [r.source_id for r in ranked]

        hit1 = bool(ranked_source_ids[:1]) and ranked_source_ids[0] in acceptable
        hit3 = any(sid in acceptable for sid in ranked_source_ids[:3])
        hit5 = any(sid in acceptable for sid in ranked_source_ids[:5])

        top1_hits += int(hit1)
        top3_hits += int(hit3)
        top5_hits += int(hit5)

        # Deployed behavior: same query, but through the actual operational threshold.
        # A correct chunk that ranks #1 but scores below the threshold will be
        # silently dropped in production -- this catches that gap explicitly.
        ranked_at_threshold = retriever.retrieve(row["query"], top_k=5, similarity_threshold=RETRIEVAL_SIMILARITY_THRESHOLD)
        ranked_ids_at_threshold = [r.source_id for r in ranked_at_threshold]
        hit1_at_threshold = bool(ranked_ids_at_threshold[:1]) and ranked_ids_at_threshold[0] in acceptable
        hit3_at_threshold = any(sid in acceptable for sid in ranked_ids_at_threshold[:3])
        top1_hits_at_threshold += int(hit1_at_threshold)
        top3_hits_at_threshold += int(hit3_at_threshold)

        record = {
            "query": row["query"],
            "expected": row["expected_source_id"],
            "top5_retrieved_unfiltered": [(r.source_id, round(r.score, 3)) for r in ranked],
            "hit@1_unfiltered": hit1,
            "hit@3_unfiltered": hit3,
            "hit@5_unfiltered": hit5,
            "hit@1_at_deployed_threshold": hit1_at_threshold,
            "hit@3_at_deployed_threshold": hit3_at_threshold,
            "suppressed_by_threshold": hit1 and not hit1_at_threshold,
        }
        per_query_report.append(record)
        if not hit3:
            weak_cases.append(record)

    n = len(in_corpus)
    top1_acc = top1_hits / n if n else 0.0
    top3_acc = top3_hits / n if n else 0.0
    top5_acc = top5_hits / n if n else 0.0
    top1_acc_at_threshold = top1_hits_at_threshold / n if n else 0.0
    top3_acc_at_threshold = top3_hits_at_threshold / n if n else 0.0
    suppressed_count = sum(1 for r in per_query_report if r["suppressed_by_threshold"])

    abstention_report = []
    correct_abstentions = 0
    for row in not_in_corpus:
        ranked = retriever.retrieve(row["query"], top_k=5, similarity_threshold=RETRIEVAL_SIMILARITY_THRESHOLD)
        abstained = len(ranked) == 0
        correct_abstentions += int(abstained)
        abstention_report.append(
            {
                "query": row["query"],
                "abstained": abstained,
                "returned": [(r.source_id, round(r.score, 3)) for r in ranked],
            }
        )

    report = {
        "eval_set_size": len(eval_rows),
        "in_corpus_queries": n,
        "not_in_corpus_queries": len(not_in_corpus),
        "top1_accuracy_unfiltered_ranking": round(top1_acc, 3),
        "top3_accuracy_unfiltered_ranking": round(top3_acc, 3),
        "top5_accuracy_unfiltered_ranking": round(top5_acc, 3),
        "top1_accuracy_at_deployed_threshold": round(top1_acc_at_threshold, 3),
        "top3_accuracy_at_deployed_threshold": round(top3_acc_at_threshold, 3),
        "correct_hits_suppressed_by_threshold": suppressed_count,
        "weak_cases_count": len(weak_cases),
        "weak_cases": weak_cases,
        "correct_abstentions": correct_abstentions,
        "abstention_rate": round(correct_abstentions / len(not_in_corpus), 3) if not_in_corpus else None,
        "abstention_detail": abstention_report,
        "operational_similarity_threshold": RETRIEVAL_SIMILARITY_THRESHOLD,
        "per_query_detail": per_query_report,
    }
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
