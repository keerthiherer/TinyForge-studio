"""
Detailed Model Diagnostics for IoT ML

Runs a complete test pass on a saved model artifact and writes a detailed report
with model metadata, dataset details, metrics, confusion matrix, and predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from ml_utils import DEFAULT_MODEL_PATH, load_artifact, load_features, load_labels


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def describe_estimator(estimator: Any) -> dict[str, Any]:
    details = {
        "type": type(estimator).__name__,
        "module": type(estimator).__module__,
        "n_features_in": getattr(estimator, "n_features_in_", None),
        "classes": getattr(estimator, "classes_", None),
    }
    if hasattr(estimator, "steps"):
        details["pipeline_steps"] = [name for name, _ in estimator.steps]
        final_estimator = estimator.steps[-1][1]
        details["final_estimator_type"] = type(final_estimator).__name__
        details["final_estimator_module"] = type(final_estimator).__module__
        details["classes"] = getattr(final_estimator, "classes_", details["classes"])
        details["n_features_in"] = getattr(final_estimator, "n_features_in_", details["n_features_in"])
    if hasattr(estimator, "get_params"):
        params = estimator.get_params(deep=False)
        details["parameters"] = {key: json_safe(value) for key, value in params.items()}
    return json_safe(details)


def label_distribution(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None) -> dict[str, Any]:
    labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if probabilities is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, probabilities))
        except ValueError as exc:
            metrics["log_loss_error"] = str(exc)
    return metrics


def save_predictions(
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    probability_classes: list[str] | None,
) -> None:
    frame = pd.DataFrame({"actual": y_true, "predicted": y_pred, "correct": y_true == y_pred})
    if probabilities is not None and probability_classes is not None:
        for index, label in enumerate(probability_classes):
            frame[f"probability_{label}"] = probabilities[:, index]
    frame.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed diagnostics report for an IoT ML model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to saved model artifact.")
    parser.add_argument("--features", default="test_features.npy", help="Path to test features .npy file.")
    parser.add_argument("--labels", default="y_test.npy", help="Path to test labels file.")
    parser.add_argument("--report", default="model_diagnostics_report.json", help="Output JSON report path.")
    parser.add_argument("--predictions", default="model_predictions.csv", help="Output prediction CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.model)
    estimator = artifact["estimator"]
    metadata = artifact.get("metadata", {})

    x_test = load_features(args.features)
    y_test = load_labels(args.labels, expected_len=len(x_test))

    expected_features = metadata.get("feature_count") or getattr(estimator, "n_features_in_", None)
    if expected_features is not None and int(expected_features) != x_test.shape[1]:
        raise ValueError(f"Feature mismatch: model expects {expected_features}, test data has {x_test.shape[1]}.")

    y_pred = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test) if hasattr(estimator, "predict_proba") else None
    probability_classes = [str(label) for label in estimator.classes_] if probabilities is not None and hasattr(estimator, "classes_") else None
    if probabilities is not None and probability_classes is None and hasattr(estimator, "steps"):
        final_estimator = estimator.steps[-1][1]
        probability_classes = [str(label) for label in getattr(final_estimator, "classes_", [])]

    report = {
        "artifact": {
            "path": args.model,
            "artifact_type": artifact.get("artifact_type"),
            "version": artifact.get("version"),
            "metadata": metadata,
        },
        "model": describe_estimator(estimator),
        "test_data": {
            "features_path": args.features,
            "labels_path": args.labels,
            "feature_shape": list(x_test.shape),
            "label_count": int(len(y_test)),
            "label_distribution": label_distribution(y_test),
        },
        "metrics": calculate_metrics(y_test, y_pred, probabilities),
    }

    report_path = Path(args.report)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2)

    predictions_path = Path(args.predictions)
    save_predictions(predictions_path, y_test, y_pred, probabilities, probability_classes)

    print("\nModel diagnostics complete.")
    print(f"Model: {args.model}")
    print(f"Test rows: {x_test.shape[0]}, features: {x_test.shape[1]}")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Balanced accuracy: {report['metrics']['balanced_accuracy']:.4f}")
    print(f"Weighted F1: {report['metrics']['f1_weighted']:.4f}")
    print(f"Report saved to {report_path}")
    print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()

