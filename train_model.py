"""
Training Script for IoT ML Models

Loads generated feature files, trains a local scikit-learn classifier, evaluates it
when test labels are available, and saves a reusable model artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_utils import (
    DEFAULT_MODEL_PATH,
    build_classifier,
    load_features,
    load_labels,
    metric_summary,
    read_choice,
    save_artifact,
)


DATA_TYPES = ["image", "audio", "numerical", "character"]
IMAGE_MODELS = [
    "MobileNetV2-style MLP",
    "FOMO-style MLP",
    "YOLO-style MLP",
    "RandomForest",
]
MODEL_OPTIONS = {
    "image": IMAGE_MODELS,
    "audio": ["MLP", "RandomForest", "SVM"],
    "numerical": ["LogisticRegression", "RandomForest", "SVM", "KNN"],
    "character": ["LogisticRegression", "RandomForest", "MLP"],
}


def choose_data_type() -> str:
    print("Available data types:")
    return read_choice("Select your data type (number): ", DATA_TYPES)


def choose_model(data_type: str) -> str:
    print("Available models:")
    return read_choice("Select model (number): ", MODEL_OPTIONS[data_type])


def optional_path(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def main() -> None:
    data_type = choose_data_type()
    model_name = choose_model(data_type)

    train_features_path = optional_path("Training features .npy", "train_features.npy")
    train_labels_path = optional_path("Training labels", "y_train.npy")
    test_features_path = optional_path("Test features .npy", "test_features.npy")
    test_labels_path = optional_path("Test labels", "y_test.npy")
    output_path = optional_path("Output model artifact", DEFAULT_MODEL_PATH)

    x_train = load_features(train_features_path)
    y_train = load_labels(train_labels_path, expected_len=len(x_train))
    print(f"Loaded training data: features={x_train.shape}, labels={y_train.shape}")

    classifier = build_classifier(model_name)
    classifier.fit(x_train, y_train)
    print(f"Trained {model_name} for {data_type} data.")

    classes = sorted(str(label) for label in np.unique(y_train))
    metadata = {
        "data_type": data_type,
        "model_name": model_name,
        "feature_count": int(x_train.shape[1]),
        "classes": classes,
    }

    test_features = Path(test_features_path)
    test_labels = Path(test_labels_path)
    if test_features.is_file() and test_labels.is_file():
        x_test = load_features(test_features_path)
        y_test = load_labels(test_labels_path, expected_len=len(x_test))
        predictions = classifier.predict(x_test)
        metrics = metric_summary(y_test, predictions)
        metadata["last_test_metrics"] = {k: v for k, v in metrics.items() if k != "report"}
        print("\nEvaluation:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 score:  {metrics['f1']:.4f}")
    else:
        print("Test features or labels were not found, so evaluation was skipped.")

    saved_path = save_artifact(output_path, classifier, metadata)
    with open("training_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nModel saved to {saved_path}")
    print("Training metadata saved to training_metadata.json")


if __name__ == "__main__":
    main()

