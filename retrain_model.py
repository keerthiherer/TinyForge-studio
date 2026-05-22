"""
Retrain Model Script for IoT ML

Loads an existing model artifact, fits it again with new labeled features, and
saves the updated artifact.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ml_utils import DEFAULT_MODEL_PATH, load_artifact, load_features, load_labels, save_artifact


def optional_path(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def main() -> None:
    model_path = optional_path("Existing trained model artifact", DEFAULT_MODEL_PATH)
    new_features_path = optional_path("New training features .npy", "train_features.npy")
    new_labels_path = optional_path("New training labels", "y_train.npy")
    output_path = optional_path("Output retrained model artifact", "iot_model_retrained.pkl")

    artifact = load_artifact(model_path)
    estimator = artifact["estimator"]
    metadata = dict(artifact.get("metadata", {}))

    x_train = load_features(new_features_path)
    y_train = load_labels(new_labels_path, expected_len=len(x_train))

    print(f"Retraining with features={x_train.shape}, labels={y_train.shape}")
    estimator.fit(x_train, y_train)

    metadata.update(
        {
            "retrained_from": model_path,
            "retrained_at": datetime.now().isoformat(timespec="seconds"),
            "feature_count": int(x_train.shape[1]),
            "classes": sorted(str(label) for label in np.unique(y_train)),
        }
    )
    saved_path = save_artifact(output_path, estimator, metadata)
    print(f"Retrained model saved to {saved_path}")


if __name__ == "__main__":
    main()

