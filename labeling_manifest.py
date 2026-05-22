"""Utilities for manual labeling inside the local IoT ML Studio (Flask app).

This module is intentionally simple and file-based.

It expects datasets like:
  dataset_split/train/<image files...>
  dataset_split/test/<image files...>

Manual labeling produces:
  y_train.npy and y_test.npy

Labels are provided by the user via the web UI and stored as a JSON
mapping file:
  labeling_train.json / labeling_test.json

Then the y_*.npy arrays are created using a stable label->index mapping.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS


@dataclass
class LabelMapping:
    label_to_idx: dict[str, int]
    idx_to_label: dict[int, str]

    @staticmethod
    def from_labels(labels: Iterable[str]) -> "LabelMapping":
        uniq = sorted({str(x) for x in labels if str(x) != ""})
        label_to_idx = {lab: i for i, lab in enumerate(uniq)}
        idx_to_label = {i: lab for lab, i in label_to_idx.items()}
        return LabelMapping(label_to_idx=label_to_idx, idx_to_label=idx_to_label)


def list_files_recursively(folder: str | os.PathLike[str]) -> list[str]:
    root = Path(folder)
    if not root.exists():
        return []
    files = [p for p in sorted(root.rglob("*")) if is_image_file(p)]
    return [str(p.relative_to(root)) for p in files]


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_manifest(split_dir: str, out_dir: str = ".") -> dict:
    """Creates a manifest for a split if missing.

    Returns:
      {"files": [relative_paths...]}
    """
    split_dir_path = Path(split_dir)
    manifest_path = Path(out_dir) / f"label_manifest_{split_dir_path.name}.json"

    files = list_files_recursively(split_dir)
    data = {"files": files}
    _write_json(manifest_path, data)
    return data


def load_or_create_manifest(split_dir: str, out_dir: str = ".") -> dict:
    split_dir_path = Path(split_dir)
    manifest_path = Path(out_dir) / f"label_manifest_{split_dir_path.name}.json"
    if manifest_path.exists():
        return _read_json(manifest_path)
    return ensure_manifest(split_dir, out_dir=out_dir)


def save_label_mapping(split_name: str, mapping: dict[str, str], out_dir: str = ".") -> Path:
    """Saves {relative_file_path: label_string}."""
    p = Path(out_dir) / f"labeling_{split_name}.json"
    _write_json(p, mapping)
    return p


def load_label_mapping(split_name: str, out_dir: str = ".") -> dict[str, str]:
    p = Path(out_dir) / f"labeling_{split_name}.json"
    obj = _read_json(p)
    return {str(k): str(v) for k, v in obj.items()}


def build_y_from_labels(
    split_name: str,
    manifest_files: list[str],
    labels_by_file: dict[str, str],
) -> tuple[np.ndarray, LabelMapping]:
    missing = [f for f in manifest_files if f not in labels_by_file or str(labels_by_file[f]).strip() == ""]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} files in split '{split_name}'.")

    label_values = [str(labels_by_file[f]) for f in manifest_files]
    mapping = LabelMapping.from_labels(label_values)
    y = np.array([mapping.label_to_idx[v] for v in label_values], dtype=np.int64)
    return y, mapping


def write_y_npy(
    split_name: str,
    manifest_files: list[str],
    labels_by_file: dict[str, str],
    out_dir: str = ".",
) -> dict:
    y, mapping = build_y_from_labels(split_name, manifest_files, labels_by_file)

    y_path = Path(out_dir) / ("y_train.npy" if split_name == "train" else "y_test.npy")
    np.save(y_path, y)

    mapping_path = Path(out_dir) / f"label_mapping_{split_name}.json"
    _write_json(
        mapping_path,
        {"label_to_idx": mapping.label_to_idx, "idx_to_label": mapping.idx_to_label},
    )

    return {
        "y_path": str(y_path),
        "mapping_path": str(mapping_path),
        "num_classes": len(mapping.label_to_idx),
    }

