"""
Feature Generation Script for IoT ML

Extracts simple local features for image, audio, numerical, and character data.
If a dataset directory contains class subfolders, labels are saved alongside the
features as y_train.npy and y_test.npy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import librosa
except ImportError:
    librosa = None

from ml_utils import read_choice


DATA_TYPES = ["image", "audio", "numerical", "character"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_EXTENSIONS = {".txt", ".log", ".md", ".csv"}


def get_data_paths() -> tuple[str, str]:
    train_path = input("Enter path to training data folder or file: ").strip()
    test_path = input("Enter path to test data folder or file: ").strip()
    return train_path, test_path


def choose_data_type() -> str:
    print("Available data types:")
    return read_choice("Select your data type (number): ", DATA_TYPES)


def extract_features_image(file_path: str | os.PathLike[str]) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required for image processing. Install it with: pip install pillow")
    img = Image.open(file_path).convert("L").resize((64, 64))
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def extract_features_audio(file_path: str | os.PathLike[str]) -> np.ndarray:
    if librosa is None:
        raise ImportError("librosa is required for audio processing. Install it with: pip install librosa")
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1).astype(np.float32)


def extract_features_numerical(file_path: str | os.PathLike[str]) -> np.ndarray:
    try:
        arr = np.loadtxt(file_path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(file_path, delimiter=",", skiprows=1)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def load_numerical_table(file_path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray | None]:
    frame = pd.read_csv(file_path)
    label_column = None
    for candidate in ("target", "label", "class"):
        if candidate in frame.columns:
            label_column = candidate
            break
    if label_column is not None:
        labels = frame[label_column].to_numpy()
        features = frame.drop(columns=[label_column]).to_numpy(dtype=np.float32)
    else:
        labels = None
        features = frame.to_numpy(dtype=np.float32)
    return features, labels


def extract_features_character_text(text: str) -> np.ndarray:
    from collections import Counter

    counts = Counter(text)
    return np.array([counts.get(chr(i), 0) for i in range(32, 127)], dtype=np.float32)


def extract_features_character(file_path: str | os.PathLike[str]) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        return extract_features_character_text(handle.read())


def feature_function(data_type: str) -> Callable[[str | os.PathLike[str]], np.ndarray]:
    return {
        "image": extract_features_image,
        "audio": extract_features_audio,
        "numerical": extract_features_numerical,
        "character": extract_features_character,
    }[data_type]


def allowed_extension(data_type: str, path: Path) -> bool:
    suffix = path.suffix.lower()
    if data_type == "image":
        return suffix in IMAGE_EXTENSIONS
    if data_type == "audio":
        return suffix in AUDIO_EXTENSIONS
    if data_type == "character":
        return suffix in TEXT_EXTENSIONS
    return path.is_file()


def pad_and_stack(features: list[np.ndarray]) -> np.ndarray:
    if not features:
        raise ValueError("No features were extracted.")
    flattened = [np.asarray(feature).reshape(-1) for feature in features]
    max_len = max(len(feature) for feature in flattened)
    stacked = np.zeros((len(flattened), max_len), dtype=np.float32)
    for row, feature in enumerate(flattened):
        stacked[row, : len(feature)] = feature
    return stacked


def iter_labeled_files(folder_path: Path, data_type: str) -> list[tuple[Path, str | None]]:
    class_dirs = [path for path in folder_path.iterdir() if path.is_dir() and not path.name.startswith(".")]
    labeled_files: list[tuple[Path, str | None]] = []
    if class_dirs:
        for class_dir in sorted(class_dirs):
            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and allowed_extension(data_type, file_path):
                    labeled_files.append((file_path, class_dir.name))
    else:
        for file_path in sorted(folder_path.rglob("*")):
            if file_path.is_file() and allowed_extension(data_type, file_path):
                labeled_files.append((file_path, None))
    return labeled_files


def process_folder(folder_path: str | os.PathLike[str], data_type: str) -> tuple[np.ndarray, np.ndarray | None]:
    root = Path(folder_path)
    extractor = feature_function(data_type)
    features = []
    labels = []
    for file_path, label in iter_labeled_files(root, data_type):
        try:
            features.append(extractor(file_path))
            if label is not None:
                labels.append(label)
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")
    label_array = np.array(labels) if labels and len(labels) == len(features) else None
    return pad_and_stack(features), label_array


def process_data(path: str | os.PathLike[str], data_type: str) -> np.ndarray:
    candidate = Path(path)
    if candidate.is_dir():
        features, _ = process_folder(candidate, data_type)
        return features
    if not candidate.is_file():
        raise FileNotFoundError(f"Data path not found: {candidate}")
    return feature_function(data_type)(candidate)


def process_dataset(path: str | os.PathLike[str], data_type: str) -> tuple[np.ndarray, np.ndarray | None]:
    candidate = Path(path)
    if candidate.is_dir():
        return process_folder(candidate, data_type)
    if data_type == "numerical" and candidate.suffix.lower() == ".csv":
        return load_numerical_table(candidate)
    return pad_and_stack([process_data(candidate, data_type)]), None


def save_features(prefix: str, features: np.ndarray, labels: np.ndarray | None) -> None:
    np.save(f"{prefix}_features.npy", features)
    if labels is not None:
        np.save(f"y_{prefix}.npy", labels)
        print(f"Saved {prefix} labels as y_{prefix}.npy")


def main() -> None:
    data_type = choose_data_type()
    train_path, test_path = get_data_paths()

    print("\nExtracting features for training data...")
    train_features, train_labels = process_dataset(train_path, data_type)
    print(f"Train features shape: {train_features.shape}")

    print("\nExtracting features for test data...")
    test_features, test_labels = process_dataset(test_path, data_type)
    print(f"Test features shape: {test_features.shape}")

    save_features("train", train_features, train_labels)
    save_features("test", test_features, test_labels)

    metadata = {
        "data_type": data_type,
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape),
        "labels_saved": bool(train_labels is not None and test_labels is not None),
    }
    with open("feature_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Features saved as train_features.npy and test_features.npy")
    print("Feature metadata saved as feature_metadata.json")


if __name__ == "__main__":
    main()
