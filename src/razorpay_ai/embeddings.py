"""Offline-only wrapper around a local sentence-transformers model.

This module NEVER contacts a network. It forces the underlying huggingface_hub /
transformers libraries into offline mode before loading, so a missing local model
fails fast and loudly instead of silently trying (and hanging on) a network call.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import SEMANTIC_EMBEDDING_DIMENSION, SEMANTIC_MODEL_DIR

# Must be set before sentence_transformers/transformers/huggingface_hub are imported
# anywhere in the process, so import is deferred to inside the loader function below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class LocalEmbeddingModel:
    """Loads a local sentence-transformers model directory and encodes text."""

    def __init__(self, model_dir: Path = SEMANTIC_MODEL_DIR):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Local embedding model not found at {self.model_dir}. "
                "This project does not download models at runtime -- place the "
                "sentence-transformers model files there first."
            )
        required_files = ["config.json", "modules.json", "tokenizer.json"]
        missing = [f for f in required_files if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Local embedding model at {self.model_dir} is missing files: {missing}")

        from sentence_transformers import SentenceTransformer  # deferred import

        self._model = SentenceTransformer(str(self.model_dir), device="cpu")
        actual_dim = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        if actual_dim != SEMANTIC_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Loaded model embedding dimension ({actual_dim}) does not match "
                f"configured SEMANTIC_EMBEDDING_DIMENSION ({SEMANTIC_EMBEDDING_DIMENSION})."
            )

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]):
        """Return an (n, dimension) float32 array of L2-normalized embeddings."""
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
