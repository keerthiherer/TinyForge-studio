"""
Model Testing Script for IoT ML

Evaluates a saved model artifact against feature and label files.
"""

from __future__ import annotations

import json

from ml_utils import DEFAULT_MODEL_PATH, load_artifact, load_features, load_labels, metric_summary


def optional_path(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def main() -> None:
    model_path = optional_path("Trained model artifact", DEFAULT_MODEL_PATH)
    test_features_path = optional_path("Test features .npy", "test_features.npy")
    test_labels_path = optional_path("Test labels", "y_test.npy")

    artifact = load_artifact(model_path)
    estimator = artifact["estimator"]
    x_test = load_features(test_features_path)
    y_test = load_labels(test_labels_path, expected_len=len(x_test))

    predictions = estimator.predict(x_test)
    metrics = metric_summary(y_test, predictions)

    print("\nEvaluation metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 score:  {metrics['f1']:.4f}")
    print("\nClassification report:")
    print(metrics["report"])

    with open("test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in metrics.items() if k != "report"}, handle, indent=2)
    print("Metrics saved to test_metrics.json")


if __name__ == "__main__":
    main()

