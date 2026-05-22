"""Shared helpers for the IoT ML command-line workflow."""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


DEFAULT_MODEL_PATH = "iot_model.pkl"


def read_choice(prompt: str, options: list[str]) -> str:
    """Read a 1-based menu choice and return the selected option."""
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    raw = input(prompt).strip()
    try:
        idx = int(raw) - 1
    except ValueError as exc:
        raise ValueError(f"Expected a number from 1 to {len(options)}, got {raw!r}.") from exc
    if idx < 0 or idx >= len(options):
        raise ValueError(f"Choice must be between 1 and {len(options)}.")
    return options[idx]


def require_file(path: str | os.PathLike[str], description: str = "file") -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"{description.capitalize()} not found: {candidate}")
    return candidate


def load_features(path: str | os.PathLike[str]) -> np.ndarray:
    """Load features from .npy and normalize to a 2D matrix."""
    feature_path = require_file(path, "feature file")
    if feature_path.suffix.lower() != ".npy":
        raise ValueError("Feature files must be NumPy .npy files.")
    features = np.load(feature_path, allow_pickle=False)
    return ensure_2d(features)


def ensure_2d(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim == 0:
        raise ValueError("Feature array is empty or scalar.")
    if features.ndim == 1:
        return features.reshape(1, -1)
    if features.ndim > 2:
        return features.reshape(features.shape[0], -1)
    return features


def load_labels(path: str | os.PathLike[str], expected_len: int | None = None) -> np.ndarray:
    """Load labels from .npy, .csv, .txt, or .json."""
    label_path = require_file(path, "label file")
    suffix = label_path.suffix.lower()
    if suffix == ".npy":
        labels = np.load(label_path, allow_pickle=True)
    elif suffix == ".csv":
        with label_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            labels = np.array([])
        else:
            preferred = "target" if "target" in rows[0] else next(iter(rows[0].keys()))
            labels = np.array([row[preferred] for row in rows])
    elif suffix == ".json":
        with label_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        labels = np.array(payload["labels"] if isinstance(payload, dict) and "labels" in payload else payload)
    else:
        with label_path.open(encoding="utf-8") as handle:
            labels = np.array([line.strip() for line in handle if line.strip()])

    labels = np.asarray(labels).reshape(-1)
    if expected_len is not None and len(labels) != expected_len:
        raise ValueError(f"Label count ({len(labels)}) does not match feature rows ({expected_len}).")
    if len(labels) == 0:
        raise ValueError("No labels were loaded.")
    return labels


def build_classifier(model_name: str, random_state: int = 42) -> Pipeline:
    """Create a practical sklearn classifier from the menu model name."""
    normalized = model_name.lower()
    if "randomforest" in normalized or "random forest" in normalized:
        estimator = RandomForestClassifier(n_estimators=120, random_state=random_state)
        return Pipeline([("model", estimator)])
    if "svm" in normalized:
        estimator = SVC(kernel="rbf", probability=True, random_state=random_state)
    elif "mlp" in normalized or "neural" in normalized or "mobilenet" in normalized or "fomo" in normalized or "yolo" in normalized:
        estimator = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=random_state)
    elif "knn" in normalized:
        estimator = KNeighborsClassifier(n_neighbors=3)
    elif "naive" in normalized or "bayes" in normalized:
        estimator = GaussianNB()
        return Pipeline([("model", estimator)])
    else:
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def save_artifact(path: str | os.PathLike[str], estimator: Any, metadata: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "iotml_sklearn_model",
        "version": 1,
        "estimator": estimator,
        "metadata": metadata,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return output_path


def load_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    model_path = require_file(path, "model artifact")
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "estimator" in payload:
        return payload
    return {
        "artifact_type": "legacy_pickle_model",
        "version": 0,
        "estimator": payload,
        "metadata": {},
    }


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    average = "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
