"""Domain classifier: Razorpay, ambiguous, or out of domain."""

from __future__ import annotations

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import DOMAIN_CLASS_WEIGHT_MAX, RANDOM_SEED


def build_domain_model(random_seed: int = RANDOM_SEED, class_weight: dict[str, float] | None = None) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=random_seed, class_weight=class_weight)),
    ])


def capped_balanced_class_weights(labels, maximum: float = DOMAIN_CLASS_WEIGHT_MAX) -> dict[str, float]:
    """Balance sparse classes without allowing a handful of rows to dominate."""
    counts = Counter(labels)
    total = sum(counts.values())
    class_count = len(counts)
    return {
        label: min(max(total / (class_count * count), 1.0), maximum)
        for label, count in counts.items()
    }


def train_domain_model(texts, labels, random_seed: int = RANDOM_SEED) -> Pipeline:
    model = build_domain_model(random_seed, capped_balanced_class_weights(labels))
    model.fit(texts, labels)
    return model
