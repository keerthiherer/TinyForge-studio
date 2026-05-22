"""Object detection workflow helpers.

This module provides best-effort dataset preparation utilities for detection.
For full correctness you must ensure the labeling JSON/yolo format matches.

Current repo contains classification-style labeling.
To integrate YOLO/FasterRCNN/SSD we primarily wire backend routes and keep
training/inference as placeholders unless the user supplies YOLO-style
`dataset.yaml` + labels.
"""

from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class DetectionWorkflowResult:
    ok: bool
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "artifacts": self.artifacts or {},
            "error": self.error,
        }


def guess_yolo_dataset_yaml(dataset_dir: str | Path) -> Optional[Path]:
    dataset_dir = Path(dataset_dir)
    for file_name in ("dataset.yaml", "data.yaml"):
        yaml_path = dataset_dir / file_name
        if yaml_path.exists():
            return yaml_path

    return None


def extract_class_names_from_dataset_yaml(yaml_path: str | Path) -> list[str]:
    """Best-effort class-name extraction without requiring PyYAML."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return []

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    names: list[str] = []
    in_names_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            raw = stripped.split(":", 1)[1].strip()
            if raw:
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, dict):
                        return [str(parsed[key]) for key in sorted(parsed)]
                    if isinstance(parsed, (list, tuple)):
                        return [str(item) for item in parsed]
                except (SyntaxError, ValueError):
                    return [item.strip().strip("'\"") for item in raw.strip("[]").split(",") if item.strip()]
            in_names_block = True
            continue

        if in_names_block:
            if stripped.startswith("-"):
                names.append(stripped[1:].strip().strip("'\""))
                continue
            if ":" in stripped:
                _, value = stripped.split(":", 1)
                names.append(value.strip().strip("'\""))
                continue
            break

    if names:
        return names

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("nc:"):
            try:
                return [f"class_{idx}" for idx in range(int(stripped.split(":", 1)[1].strip()))]
            except ValueError:
                return []

    return []


def detection_dataset_summary(dataset_dir: str | Path) -> DetectionWorkflowResult:
    dataset_dir = Path(dataset_dir)
    yaml_path = guess_yolo_dataset_yaml(dataset_dir)
    artifacts: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "dataset_yaml": str(yaml_path) if yaml_path else None,
        "class_names": extract_class_names_from_dataset_yaml(yaml_path) if yaml_path else [],
    }
    return DetectionWorkflowResult(ok=dataset_dir.exists() and dataset_dir.is_dir(), artifacts=artifacts)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _image_files(split_dir: Path, exclude_dir: Path | None = None) -> list[Path]:
    files: list[Path] = []
    exclude_resolved = exclude_dir.resolve() if exclude_dir and exclude_dir.exists() else None
    for file_path in sorted(split_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if exclude_resolved and exclude_resolved in file_path.resolve().parents:
            continue
        files.append(file_path)
    return files


def _split_sources(dataset_dir: Path) -> dict[str, Path]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"
    if train_dir.exists() and (val_dir.exists() or test_dir.exists()):
        return {"train": train_dir, "val": val_dir if val_dir.exists() else test_dir}
    if train_dir.exists():
        return {"train": train_dir, "val": train_dir}
    return {"train": dataset_dir, "val": dataset_dir}


def _label_for_image(
    file_path: Path,
    split_dir: Path,
    split_name: str,
    label_maps: dict[str, dict[str, str]],
) -> str:
    rel = file_path.relative_to(split_dir).as_posix()
    labels = label_maps.get(split_name, {})
    mapped = str(labels.get(rel, "")).strip()
    if mapped:
        return mapped

    # Class-folder datasets keep the class as the first relative path segment.
    rel_path = Path(rel)
    if len(rel_path.parts) > 1:
        return rel_path.parts[0]

    return "object"


def prepare_yolo_dataset(dataset_dir: str | Path, output_dir: str | Path | None = None) -> DetectionWorkflowResult:
    """Build a YOLO dataset from stored COCO bbox annotations.

    Primary behavior:
      - Read COCO JSON from `coco_store/<project>/coco_train.json` and `coco_store/<project>/coco_test.json`.
      - Write YOLO TXT labels with real COCO bbox coordinates (no fake full-image boxes).

    Fallback behavior (legacy): if COCO annotations are missing for the dataset,
    create one full-image box per image-level label from `labeling_train.json` / `labeling_test.json`.
    """


    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return DetectionWorkflowResult(
            ok=False,
            error=f"Detection dataset directory not found: {dataset_dir}",
            artifacts={"dataset_dir": str(dataset_dir)},
        )

    yolo_dir = Path(output_dir) if output_dir else dataset_dir / "yolo_auto"
    # ----------------------------
    # COCO-first export (no fake boxes)
    # ----------------------------
    repo_root = dataset_dir.resolve().parent
    project_guess = dataset_dir.parent.name if dataset_dir.parent.name else "default"
    coco_store_dir = repo_root / "coco_store" / project_guess

    def load_coco(split: str) -> dict[str, Any]:
        p = coco_store_dir / f"coco_{split}.json"
        if not p.is_file():
            return {}
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("images", [])
                payload.setdefault("annotations", [])
                payload.setdefault("categories", [])
                return payload
        except Exception:
            pass
        return {}

    coco_train = load_coco("train")
    coco_test = load_coco("test")
    coco_has_ann = bool(coco_train.get("images") or coco_train.get("annotations"))

    def source_dir_for_yolo_split(split_yolo: str) -> Path:
        # yolo splits: train -> dataset_split/train, val -> dataset_split/test
        if split_yolo == "train":
            return dataset_dir / "train" if (dataset_dir / "train").exists() else dataset_dir
        return dataset_dir / "test" if (dataset_dir / "test").exists() else dataset_dir

    def get_image_size_from_entry(img_entry: dict[str, Any], source_images_dir: Path) -> tuple[int, int] | None:
        try:
            w = int(img_entry.get("width"))
            h = int(img_entry.get("height"))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass

        file_name = str(img_entry.get("file_name"))
        img_path = source_images_dir / Path(file_name)
        if not img_path.is_file():
            return None
        try:
            from PIL import Image

            with Image.open(img_path) as im:
                w, h = im.size
            return int(w), int(h)
        except Exception:
            return None

    if coco_has_ann:
        cat_by_id: dict[int, str] = {}
        for cat in (coco_train.get("categories") or []):
            try:
                cat_by_id[int(cat.get("id"))] = str(cat.get("name"))
            except Exception:
                continue
        if not cat_by_id:
            for cat in (coco_test.get("categories") or []):
                try:
                    cat_by_id[int(cat.get("id"))] = str(cat.get("name"))
                except Exception:
                    continue
        if not cat_by_id:
            cat_by_id = {0: "object"}

        names = [cat_by_id[k] for k in sorted(cat_by_id.keys())]
        class_to_idx = {name: idx for idx, name in enumerate(names)}
        idx_by_cat_id = {cat_id: class_to_idx[name] for cat_id, name in cat_by_id.items() if name in class_to_idx}

        def write_split(coco: dict[str, Any], split_yolo: str) -> tuple[int, int]:
            source_images_dir = source_dir_for_yolo_split(split_yolo)
            images = coco.get("images", []) or []
            anns = coco.get("annotations", []) or []

            img_by_id: dict[int, dict[str, Any]] = {int(img["id"]): img for img in images if "id" in img}
            anns_by_img: dict[int, list[dict[str, Any]]] = {}
            for ann in anns:
                try:
                    image_id = int(ann.get("image_id"))
                except Exception:
                    continue
                anns_by_img.setdefault(image_id, []).append(ann)

            out_img_dir = yolo_dir / "images" / split_yolo
            out_lbl_dir = yolo_dir / "labels" / split_yolo
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            written_images = 0
            written_labels = 0

            for image_id, img_entry in img_by_id.items():
                file_name = str(img_entry.get("file_name"))
                size = get_image_size_from_entry(img_entry, source_images_dir)
                if not size:
                    continue
                img_w, img_h = size
                if img_w <= 0 or img_h <= 0:
                    continue

                out_img_path = out_img_dir / Path(file_name)
                out_lbl_path = (out_lbl_dir / Path(file_name)).with_suffix(".txt")
                out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

                src_img_path = source_images_dir / Path(file_name)
                if src_img_path.is_file():
                    if (not out_img_path.exists()) or src_img_path.stat().st_mtime > out_img_path.stat().st_mtime:
                        shutil.copy2(src_img_path, out_img_path)

                yolo_lines: list[str] = []
                for ann in anns_by_img.get(image_id, []):
                    bbox = ann.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    try:
                        x, y, bw, bh = map(float, bbox)
                        cat_id = int(ann.get("category_id"))
                    except Exception:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue

                    cls_idx = idx_by_cat_id.get(cat_id)
                    if cls_idx is None:
                        continue

                    cx = (x + bw / 2.0) / img_w
                    cy = (y + bh / 2.0) / img_h
                    nw = bw / img_w
                    nh = bh / img_h

                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))

                    yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                out_lbl_path.write_text("".join(yolo_lines), encoding="utf-8")
                written_images += 1
                if yolo_lines:
                    written_labels += 1

            return written_images, written_labels

        train_images, train_labels = write_split(coco_train, "train")
        val_images, val_labels = write_split(coco_test, "val")

        yaml_path = yolo_dir / "data.yaml"
        names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {yolo_dir.resolve().as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    f"nc: {len(names)}",
                    "names:",
                    names_yaml,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return DetectionWorkflowResult(
            ok=True,
            artifacts={
                "dataset_dir": str(dataset_dir),
                "dataset_yaml": str(yaml_path),
                "yolo_dir": str(yolo_dir),
                "class_names": names,
                "train_images": train_images,
                "val_images": val_images,
                "annotation_mode": "coco_bbox",
                "train_images_with_labels": train_labels,
                "val_images_with_labels": val_labels,
            },
        )

    # Legacy fallback: image-level labels -> one full-image box.
    sources = _split_sources(dataset_dir)
    root_for_labels = dataset_dir.parent
    label_maps = {
        "train": _read_json_if_exists(root_for_labels / "labeling_train.json"),
        "val": _read_json_if_exists(root_for_labels / "labeling_test.json"),
    }


    entries: dict[str, list[tuple[Path, str, str]]] = {"train": [], "val": []}
    class_names: set[str] = set()

    for split_name, source_dir in sources.items():
        for image_path in _image_files(source_dir, exclude_dir=yolo_dir):
            label = _label_for_image(image_path, source_dir, split_name, label_maps)
            rel = image_path.relative_to(source_dir).as_posix()
            entries[split_name].append((image_path, rel, label))
            class_names.add(label)

    if not entries["train"]:
        return DetectionWorkflowResult(
            ok=False,
            error=f"No training images found in {sources['train']}",
            artifacts={"dataset_dir": str(dataset_dir), "yolo_dir": str(yolo_dir)},
        )

    names = sorted(class_names or {"object"})
    class_to_idx = {name: idx for idx, name in enumerate(names)}

    for split_name, split_entries in entries.items():
        for image_path, rel, label in split_entries:
            target_image = yolo_dir / "images" / split_name / rel
            target_label = (yolo_dir / "labels" / split_name / rel).with_suffix(".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists() or image_path.stat().st_mtime > target_image.stat().st_mtime:
                shutil.copy2(image_path, target_image)
            target_label.write_text(f"{class_to_idx[label]} 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    yaml_path = yolo_dir / "data.yaml"
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {yolo_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names:",
                names_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )

    return DetectionWorkflowResult(
        ok=True,
        artifacts={
            "dataset_dir": str(dataset_dir),
            "dataset_yaml": str(yaml_path),
            "yolo_dir": str(yolo_dir),
            "class_names": names,
            "train_images": len(entries["train"]),
            "val_images": len(entries["val"]),
            "annotation_mode": "full_image_boxes",
        },
    )


def ensure_detection_dataset(dataset_dir: str | Path) -> DetectionWorkflowResult:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return DetectionWorkflowResult(
            ok=False,
            error=f"Detection dataset directory not found: {dataset_dir}",
            artifacts={"dataset_dir": str(dataset_dir)},
        )

    yaml_path = guess_yolo_dataset_yaml(dataset_dir)

    if yaml_path is None:
        return DetectionWorkflowResult(
            ok=False,
            error=(
                "dataset.yaml or data.yaml not found. For YOLO training, place a YOLO/Ultralytics "
                "dataset config under the dataset directory and point it at train/val images and labels."
            ),
            artifacts={"dataset_dir": str(dataset_dir)},
        )

    return DetectionWorkflowResult(
        ok=True,
        artifacts={
            "dataset_dir": str(dataset_dir),
            "dataset_yaml": str(yaml_path),
            "class_names": extract_class_names_from_dataset_yaml(yaml_path),
        },
    )


def load_class_names(labels_json_path: str | Path) -> list[str]:
    labels_json_path = Path(labels_json_path)
    if not labels_json_path.exists():
        return []
    with labels_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        # common: {"imgname": "label"}
        names = sorted({str(v) for v in payload.values()})
        return names
    if isinstance(payload, list):
        return [str(x) for x in payload]
    return []

