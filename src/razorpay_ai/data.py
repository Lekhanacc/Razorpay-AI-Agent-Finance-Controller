"""Dataset loading, normalization, validation, and reproducible splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_PATH, DOMAIN_LABEL_MAP, RANDOM_SEED, REQUIRED_COLUMNS, TAXONOMY_PATH


@dataclass(frozen=True)
class DatasetBundle:
    data: pd.DataFrame
    taxonomy: dict


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def normalize_domain_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical uppercase domain labels."""
    normalized = data.copy()
    source = normalized["domain"].astype(str).str.strip().str.lower()
    unknown = sorted(set(source) - set(DOMAIN_LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown dataset domain labels: {unknown}")
    normalized["domain"] = source.map(DOMAIN_LABEL_MAP)
    return normalized


def validate_dataset(data: pd.DataFrame, taxonomy: dict) -> None:
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(data) != 770:
        raise ValueError(f"Expected 770 dataset rows, found {len(data)}")
    taxonomy_intents = {item["name"] for item in taxonomy["intents"]}
    dataset_intents = set(data["intent"])
    unknown = dataset_intents - taxonomy_intents
    absent = taxonomy_intents - dataset_intents
    if unknown:
        raise ValueError(f"Dataset contains unknown intent labels: {sorted(unknown)}")
    if absent:
        raise ValueError(f"Taxonomy intents without examples: {sorted(absent)}")
    duplicates = data["text"].astype(str).str.strip().str.lower().duplicated(keep=False)
    if duplicates.any():
        duplicate_texts = data.loc[duplicates, "text"].tolist()
        raise ValueError(f"Duplicate training utterances after normalization: {duplicate_texts}")


def load_dataset(dataset_path: Path = DATASET_PATH, taxonomy_path: Path = TAXONOMY_PATH) -> DatasetBundle:
    data = pd.read_csv(dataset_path)
    data = normalize_domain_labels(data)
    taxonomy = load_taxonomy(taxonomy_path)
    validate_dataset(data, taxonomy)
    return DatasetBundle(data=data, taxonomy=taxonomy)


def split_dataset(data: pd.DataFrame, label_column: str, random_seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make a 60/20/20 split, stratifying labels that have enough examples.

    Labels with fewer than five rows cannot be present in every split. They are
    retained in training so inference can still recognize them.
    """
    counts = data[label_column].value_counts()
    rare_labels = counts[counts < 5].index
    rare = data[data[label_column].isin(rare_labels)]
    eligible = data[~data[label_column].isin(rare_labels)]
    train, held_out = train_test_split(
        eligible, test_size=0.4, random_state=random_seed, stratify=eligible[label_column]
    )
    validation, test = train_test_split(
        held_out,
        test_size=0.5,
        random_state=random_seed,
        stratify=held_out[label_column],
    )
    train = pd.concat([train, rare], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
