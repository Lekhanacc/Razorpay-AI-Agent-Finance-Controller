"""Razorpay intent classifier."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED


def build_intent_model(random_seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=random_seed)),
    ])


def train_intent_model(texts, labels, random_seed: int = RANDOM_SEED) -> Pipeline:
    model = build_intent_model(random_seed)
    model.fit(texts, labels)
    return model
