"""Train and persist the Razorpay-only intent classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from razorpay_ai.config import INTENT_MODEL_PATH, MODELS_DIR
from razorpay_ai.data import load_dataset, split_dataset
from razorpay_ai.intent import train_intent_model


def main() -> None:
    bundle = load_dataset()
    razorpay_data = bundle.data[bundle.data.domain == "RAZORPAY"].reset_index(drop=True)
    train, validation, test = split_dataset(razorpay_data, "intent")
    model = train_intent_model(train.text, train.intent)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, INTENT_MODEL_PATH)
    print(f"Saved intent model to {INTENT_MODEL_PATH}")
    print(f"Split sizes: train={len(train)}, validation={len(validation)}, test={len(test)}")


if __name__ == "__main__":
    main()
