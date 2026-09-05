"""Ingest data/knowledge/ into a chunked, TF-IDF-indexed retrieval artifact.

Reads data/knowledge_manifest.jsonl for provenance, chunks each referenced document
with razorpay_ai.chunking, fits a TF-IDF vectorizer over the chunks, and writes the
resulting artifacts to data/knowledge_index/:

  chunks.jsonl        - one JSON object per chunk, with full provenance metadata
  vectorizer.joblib    - fitted TfidfVectorizer
  matrix.joblib         - TF-IDF matrix, row i corresponds to chunks.jsonl line i
  index_manifest.json  - build configuration and summary counts
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from razorpay_ai.chunking import chunk_document
from razorpay_ai.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    INDEX_MANIFEST_PATH,
    KNOWLEDGE_INDEX_DIR,
    KNOWLEDGE_MANIFEST_PATH,
    MATRIX_PATH,
    MIN_CHUNK_WORDS,
    VECTORIZER_PATH,
)

REQUIRED_MANIFEST_FIELDS = ("source_id", "title", "product", "topic", "provenance_status", "local_path")


def load_manifest(path: Path = KNOWLEDGE_MANIFEST_PATH) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
            if missing:
                raise ValueError(f"manifest row {row.get('source_id', '<unknown>')} missing fields: {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no manifest rows found in {path}")
    return rows


def build_chunk_records(manifest_rows: list[dict], project_root: Path) -> list[dict]:
    records: list[dict] = []
    for source in manifest_rows:
        doc_path = project_root / source["local_path"]
        if not doc_path.exists():
            raise FileNotFoundError(f"knowledge document not found for {source['source_id']}: {doc_path}")
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_document(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, MIN_CHUNK_WORDS)
        if not chunks:
            raise ValueError(f"chunking produced zero chunks for source {source['source_id']}")
        url = source.get("final_url") or source.get("requested_url", "")
        for i, chunk_text in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{source['source_id']}::chunk{i}",
                    "source_id": source["source_id"],
                    "title": source["title"],
                    "url": url,
                    "product": source["product"],
                    "topic": source["topic"],
                    "provenance_status": source["provenance_status"],
                    "text": chunk_text,
                }
            )
    return records


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_rows = load_manifest()
    chunk_records = build_chunk_records(manifest_rows, project_root)

    KNOWLEDGE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as handle:
        for record in chunk_records:
            handle.write(json.dumps(record) + "\n")

    texts = [record["text"] for record in chunk_records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)

    index_manifest = {
        "num_source_documents": len(manifest_rows),
        "num_chunks": len(chunk_records),
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "min_chunk_words": MIN_CHUNK_WORDS,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "chunks_per_source": {
            source_id: sum(1 for r in chunk_records if r["source_id"] == source_id)
            for source_id in {r["source_id"] for r in chunk_records}
        },
    }
    with open(INDEX_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(index_manifest, handle, indent=2, sort_keys=True)

    print(f"Ingested {len(manifest_rows)} documents into {len(chunk_records)} chunks.")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Index written to {KNOWLEDGE_INDEX_DIR}")


if __name__ == "__main__":
    main()
