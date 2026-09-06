"""Evaluate saved baseline models on fixed validation and test splits."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from razorpay_ai.data import load_dataset, split_dataset
from razorpay_ai.pipeline import RazorpayAIPipeline


def metrics(model, split, label_column: str, include_per_class: bool = False) -> dict:
    actual = split[label_column]
    predicted = model.predict(split.text)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
    result = {
        "accuracy": accuracy_score(actual, predicted),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": macro_f1,
        "class_distribution": actual.value_counts().sort_index().to_dict(),
        "confusion_matrix": {
            "labels": sorted(set(actual) | set(predicted)),
            "matrix": confusion_matrix(actual, predicted, labels=sorted(set(actual) | set(predicted))).tolist(),
        },
    }
    if include_per_class:
        result["per_intent_metrics"] = classification_report(actual, predicted, output_dict=True, zero_division=0)
    return result


def main() -> None:
    bundle = load_dataset()
    domain_train, domain_validation, domain_test = split_dataset(bundle.data, "domain")
    razorpay_data = bundle.data[bundle.data.domain == "RAZORPAY"].reset_index(drop=True)
    intent_train, intent_validation, intent_test = split_dataset(razorpay_data, "intent")
    pipeline = RazorpayAIPipeline.load()
    report = {
        "domain": {
            "validation": metrics(pipeline.domain_model, domain_validation, "domain"),
            "test": metrics(pipeline.domain_model, domain_test, "domain"),
        },
        "intent": {
            "validation": metrics(pipeline.intent_model, intent_validation, "intent", include_per_class=True),
            "test": metrics(pipeline.intent_model, intent_test, "intent", include_per_class=True),
        },
        "split_sizes": {
            "domain": {"train": len(domain_train), "validation": len(domain_validation), "test": len(domain_test)},
            "intent": {"train": len(intent_train), "validation": len(intent_validation), "test": len(intent_test)},
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
