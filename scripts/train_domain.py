"""Train and persist the domain classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from razorpay_ai.config import DOMAIN_MODEL_PATH, MODELS_DIR
from razorpay_ai.data import load_dataset, split_dataset
from razorpay_ai.domain import train_domain_model


def main() -> None:
    bundle = load_dataset()
    train, validation, test = split_dataset(bundle.data, "domain")
    model = train_domain_model(train.text, train.domain)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, DOMAIN_MODEL_PATH)
    print(f"Saved domain model to {DOMAIN_MODEL_PATH}")
    print(f"Split sizes: train={len(train)}, validation={len(validation)}, test={len(test)}")


if __name__ == "__main__":
    main()
