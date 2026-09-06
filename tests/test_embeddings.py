import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from razorpay_ai.config import SEMANTIC_EMBEDDING_DIMENSION, SEMANTIC_MODEL_DIR
from razorpay_ai.embeddings import LocalEmbeddingModel


def test_local_model_loads_offline_from_the_configured_directory():
    model = LocalEmbeddingModel(SEMANTIC_MODEL_DIR)
    assert model.dimension == SEMANTIC_EMBEDDING_DIMENSION


def test_local_model_raises_file_not_found_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalEmbeddingModel(tmp_path / "does_not_exist")


def test_local_model_raises_file_not_found_for_incomplete_model_directory(tmp_path):
    incomplete_dir = tmp_path / "incomplete_model"
    incomplete_dir.mkdir()
    (incomplete_dir / "config.json").write_text("{}")
    # modules.json and tokenizer.json are intentionally missing.
    with pytest.raises(FileNotFoundError):
        LocalEmbeddingModel(incomplete_dir)


def test_encode_returns_expected_shape_and_normalized_vectors():
    model = LocalEmbeddingModel(SEMANTIC_MODEL_DIR)
    embeddings = model.encode(["How do I create a payment link?", "What is the weather today?"])
    assert embeddings.shape == (2, SEMANTIC_EMBEDDING_DIMENSION)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_encode_is_deterministic_for_the_same_input():
    model = LocalEmbeddingModel(SEMANTIC_MODEL_DIR)
    first = model.encode(["How do I create a payment link?"])
    second = model.encode(["How do I create a payment link?"])
    assert np.allclose(first, second)


def test_similar_sentences_score_higher_than_unrelated_sentences():
    model = LocalEmbeddingModel(SEMANTIC_MODEL_DIR)
    query = model.encode(["How do I create a payment link?"])[0]
    similar = model.encode(["How can I generate a link to collect a payment?"])[0]
    unrelated = model.encode(["What is the capital of France?"])[0]
    assert float(query @ similar) > float(query @ unrelated)
