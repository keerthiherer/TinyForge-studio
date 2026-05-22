"""
Live Classification Script for IoT ML

Loads a trained artifact and classifies one sample at a time from a file, a CSV
numeric row, or direct text input.
"""

from __future__ import annotations

import numpy as np

from generate_features import extract_features_character_text, process_data
from ml_utils import DEFAULT_MODEL_PATH, ensure_2d, load_artifact, read_choice


def optional_path(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def read_sample(source_type: str, data_type: str) -> np.ndarray:
    if source_type == "file":
        path = input("Enter sample file path: ").strip()
        return ensure_2d(process_data(path, data_type))
    if source_type == "numeric row":
        raw = input("Enter comma-separated numeric values: ").strip()
        return np.fromstring(raw, sep=",", dtype=float).reshape(1, -1)
    text = input("Enter text to classify: ")
    return extract_features_character_text(text).reshape(1, -1)


def print_prediction(estimator, sample: np.ndarray) -> None:
    prediction = estimator.predict(sample)[0]
    print(f"Prediction: {prediction}")
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(sample)[0]
        classes = estimator.classes_ if hasattr(estimator, "classes_") else range(len(probabilities))
        best = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:3]
        print("Top probabilities:")
        for label, probability in best:
            print(f"  {label}: {probability:.4f}")


def main() -> None:
    model_path = optional_path("Trained model artifact", DEFAULT_MODEL_PATH)
    artifact = load_artifact(model_path)
    estimator = artifact["estimator"]
    metadata = artifact.get("metadata", {})
    data_type = metadata.get("data_type") or read_choice(
        "Select model data type (number): ", ["image", "audio", "numerical", "character"]
    )

    print(f"Loaded model for {data_type} data.")
    source_options = ["file", "numeric row", "text"]
    while True:
        print("\nInput sources:")
        source_type = read_choice("Select input source (number): ", source_options)
        sample = read_sample(source_type, data_type)
        expected = metadata.get("feature_count")
        if expected is not None and sample.shape[1] != expected:
            raise ValueError(f"Sample has {sample.shape[1]} features, but the model expects {expected}.")
        print_prediction(estimator, sample)
        if input("Classify another sample? (y/n): ").strip().lower() != "y":
            break

    print("Live classification ended.")


if __name__ == "__main__":
    main()

