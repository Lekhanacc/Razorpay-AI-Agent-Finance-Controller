"""Build the semantic retrieval index.

Reuses the exact chunk set already produced by scripts/ingest_knowledge.py
(data/knowledge_index/chunks.jsonl) -- chunking is a shared step in the
architecture (Phase 3 does not re-chunk, only re-represents the same chunks).
Copies that chunk set into data/knowledge_index_semantic/ so the semantic index
is self-contained and reproducible on its own, without overwriting the TF-IDF
index in data/knowledge_index/.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from razorpay_ai.config import (
    CHUNKS_PATH,
    SEMANTIC_CHUNKS_PATH,
    SEMANTIC_EMBEDDING_DIMENSION,
    SEMANTIC_EMBEDDINGS_PATH,
    SEMANTIC_INDEX_DIR,
    SEMANTIC_INDEX_MANIFEST_PATH,
    SEMANTIC_MODEL_DIR,
    SEMANTIC_MODEL_NAME,
)
from razorpay_ai.embeddings import LocalEmbeddingModel


def load_tfidf_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/ingest_knowledge.py first (Phase 2 chunking).")
    chunks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise ValueError(f"{path} contained no chunks.")
    return chunks


def main() -> None:
    chunks = load_tfidf_chunks()

    SEMANTIC_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEMANTIC_CHUNKS_PATH, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")

    print(f"Loading local embedding model from {SEMANTIC_MODEL_DIR} (offline)...")
    model = LocalEmbeddingModel()

    texts = [c["text"] for c in chunks]
    start = time.perf_counter()
    embeddings = model.encode(texts)
    elapsed = time.perf_counter() - start

    if embeddings.shape[1] != SEMANTIC_EMBEDDING_DIMENSION:
        raise ValueError(f"Unexpected embedding dimension {embeddings.shape[1]}, expected {SEMANTIC_EMBEDDING_DIMENSION}")

    np.save(SEMANTIC_EMBEDDINGS_PATH, embeddings.astype(np.float32))

    index_manifest = {
        "embedding_model_name": SEMANTIC_MODEL_NAME,
        "embedding_model_source": "local (offline, user-provided files; huggingface.co is not reachable in this environment)",
        "num_chunks": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "embeddings_normalized": True,
        "embedding_build_time_seconds": round(elapsed, 4),
        "chunks_per_source": {
            source_id: sum(1 for c in chunks if c["source_id"] == source_id) for source_id in {c["source_id"] for c in chunks}
        },
    }
    with open(SEMANTIC_INDEX_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(index_manifest, handle, indent=2, sort_keys=True)

    print(f"Embedded {len(chunks)} chunks into {embeddings.shape[1]}-dimensional vectors in {elapsed:.3f}s.")
    print(f"Index written to {SEMANTIC_INDEX_DIR}")


if __name__ == "__main__":
    main()
