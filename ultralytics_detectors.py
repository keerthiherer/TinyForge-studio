"""Ultralytics-based object detection backends.

This module is used by the Flask app to train/detect using:
- YOLO (YOLOv8/YOLOv11 via Ultralytics)
- Faster R-CNN
- SSD

Notes:
- Training/inference requires the `ultralytics` package for YOLO.
- For Faster R-CNN and SSD we fall back to torchvision if available.
- The integration points in `web_app.py` use `train_yolo`, `train_faster_rcnn`,
  `train_ssd`, and share a common return structure.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainResult:
    ok: bool
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "logs": self.logs,
            "error": self.error,
        }


def _require_dir(p: str | Path, name: str) -> Path:
    path = Path(p)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} not found or not a directory: {path}")
    return path


def _require_file(p: str | Path, name: str) -> Path:
    path = Path(p)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} not found or not a file: {path}")
    return path


def _label_name(names: Any, label_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(label_id, names.get(str(label_id), label_id)))
    if isinstance(names, (list, tuple)) and 0 <= label_id < len(names):
        return str(names[label_id])
    return str(label_id)


def _draw_detections(image_path: str | Path, detections: list[dict[str, Any]], output_path: str | Path) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError(f"Pillow is required to draw detection results. Details: {e}") from e

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = ["#5265f6", "#20c788", "#f69d22", "#fa7d9b", "#13aee4", "#7138d2"]

    for index, det in enumerate(detections):
        box = det["box"]
        x1, y1, x2, y2 = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        color = palette[index % len(palette)]
        label = f"{det['label']} {det['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label_box = draw.textbbox((x1, y1), label, font=font)
        label_height = label_box[3] - label_box[1] + 8
        label_width = label_box[2] - label_box[0] + 10
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
        draw.text((x1 + 5, label_y + 4), label, fill="#ffffff", font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def detect_yolo(
    model_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    confidence: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run YOLO inference and save an annotated preview image."""
    model_path = _require_file(model_path, "model_path")
    image_path = _require_file(image_path, "image_path")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Ultralytics is required for YOLO detection. Install it with `pip install ultralytics`. Details: {e}") from e

    yolo = YOLO(str(model_path))
    results = yolo.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    result = results[0]
    names = getattr(result, "names", None) or getattr(yolo, "names", {})
    detections: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        for coords, score, class_id in zip(xyxy, confs, classes):
            if float(score) < confidence:
                continue
            x1, y1, x2, y2 = [float(value) for value in coords]
            label_id = int(class_id)
            detections.append(
                {
                    "label": _label_name(names, label_id),
                    "class_id": label_id,
                    "confidence": float(score),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    annotated_path = _draw_detections(image_path, detections, output_path)
    return {
        "ok": True,
        "detector": "YOLO",
        "model_path": str(model_path),
        "image_path": str(image_path),
        "annotated_path": annotated_path,
        "detections": detections,
        "count": len(detections),
    }


def detect_torchvision_detector(
    detector: str,
    model_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    confidence: float = 0.25,
    num_classes: int = 2,
    device: str = "cpu",
    class_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Run Faster R-CNN or SSD inference from a Torchvision state_dict."""
    model_path = _require_file(model_path, "model_path")
    image_path = _require_file(image_path, "image_path")

    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
    except Exception as e:
        raise RuntimeError(f"Torch, Torchvision, and Pillow are required for {detector} detection. Details: {e}") from e

    normalized = detector.lower().replace("_", "-").replace(" ", "-")
    if "faster" in normalized or "rcnn" in normalized:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        detector_name = "Faster R-CNN"
    elif normalized == "ssd" or "ssdlite" in normalized:
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
        detector_name = "SSD"
    else:
        raise ValueError(f"Unsupported Torchvision detector: {detector}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    names = class_names or ["background"] + [str(index) for index in range(1, num_classes)]
    detections: list[dict[str, Any]] = []
    boxes = output.get("boxes", []).detach().cpu().tolist()
    scores = output.get("scores", []).detach().cpu().tolist()
    labels = output.get("labels", []).detach().cpu().tolist()
    for coords, score, label_id in zip(boxes, scores, labels):
        if float(score) < confidence:
            continue
        x1, y1, x2, y2 = [float(value) for value in coords]
        class_id = int(label_id)
        detections.append(
            {
                "label": _label_name(names, class_id),
                "class_id": class_id,
                "confidence": float(score),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    annotated_path = _draw_detections(image_path, detections, output_path)
    return {
        "ok": True,
        "detector": detector_name,
        "model_path": str(model_path),
        "image_path": str(image_path),
        "annotated_path": annotated_path,
        "detections": detections,
        "count": len(detections),
    }


def _safe_dataset_yaml(dataset_dir: str | Path) -> Path:
    """Return dataset.yaml/data.yaml if present.

    Expected structure for YOLO:
    dataset_dir/
      dataset.yaml
      images/train, images/val
    This is a best-effort helper.
    """
    dataset_dir = Path(dataset_dir)
    for file_name in ("dataset.yaml", "data.yaml"):
        yaml_path = dataset_dir / file_name
        if yaml_path.exists():
            return yaml_path
    raise FileNotFoundError(
        f"dataset.yaml or data.yaml not found in {dataset_dir}. "
        "Create a YOLO/Ultralytics dataset config before training YOLO."
    )


def train_yolo(
    dataset_dir: str | Path,
    model: str = "yolo11n.pt",
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    output_dir: str | Path = "runs/detect",
    run_name: str = "iotml-yolo",
) -> Dict[str, Any]:
    """Train YOLO via Ultralytics."""
    dataset_dir = _require_dir(dataset_dir, "dataset_dir")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        return TrainResult(
            ok=False,
            error=f"Ultralytics is required for YOLO training. Install it with `pip install ultralytics`. Details: {e}",
        ).to_dict()

    try:
        data_yaml = _safe_dataset_yaml(dataset_dir)
        yolo = YOLO(model)
        results = yolo.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )

        # best-effort path discovery
        best = getattr(results, "save_dir", None)
        model_path = None
        if best is not None:
            best_dir = Path(best)
            cand = best_dir / "weights" / "best.pt"
            if cand.exists():
                model_path = str(cand)

        return TrainResult(
            ok=True,
            model_path=model_path,
            metrics={"epochs": epochs, "imgsz": imgsz, "batch": batch},
            logs={"dataset_yaml": str(data_yaml), "base_model": model, "run_name": run_name},
        ).to_dict()
    except Exception as e:
        return TrainResult(ok=False, error=str(e)).to_dict()


def _load_coco_detections_for_dataset(dataset_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load coco_train/coco_test for the selected project.

    The Flask UI passes dataset_dir like:
      dataset_split/  (legacy) OR dataset_split/<project>/ (if used)

    In this repo, COCO JSONs are stored at:
      coco_store/<project_guess>/coco_train.json
      coco_store/<project_guess>/coco_test.json

    The COCO JSONs must be loaded from an explicit project folder.
    web_app.py passes dataset_dir like dataset_split/ (legacy) or dataset_split/train (legacy),
    so we derive the repo root from dataset_dir and default to 'default' only if the
    caller does not supply a project name.
    """
    dataset_dir = _require_dir(dataset_dir, "dataset_dir")
    repo_root = dataset_dir.resolve().parent
    project_guess = "default"

    coco_store_dir = repo_root / "coco_store" / project_guess
    coco_train_path = coco_store_dir / "coco_train.json"
    coco_test_path = coco_store_dir / "coco_test.json"


    if not coco_train_path.is_file() or not coco_test_path.is_file():
        raise FileNotFoundError(
            f"Missing COCO files for project '{project_guess}'. Expected: {coco_train_path} and {coco_test_path}"
        )

    coco_train = json.loads(coco_train_path.read_text(encoding="utf-8"))
    coco_test = json.loads(coco_test_path.read_text(encoding="utf-8"))
    if not isinstance(coco_train, dict) or not isinstance(coco_test, dict):
        raise ValueError("COCO JSONs must be dict root objects")

    return coco_train, coco_test, str(project_guess)


class _CocoDetectionDataset(__import__("torch").utils.data.Dataset):
    """Minimal COCO bbox dataset for torchvision detection models."""

    def __init__(self, coco: dict[str, Any], images_root: Path, transforms=None) -> None:
        super().__init__()
        self.coco = coco
        self.images_root = images_root
        self.transforms = transforms

        self.images = coco.get("images", []) or []
        self.id_to_image = {int(img["id"]): img for img in self.images if "id" in img}

        # index anns by image_id
        anns = coco.get("annotations", []) or []
        self.anns_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for ann in anns:
            try:
                img_id = int(ann.get("image_id"))
            except Exception:
                continue
            self.anns_by_image_id.setdefault(img_id, []).append(ann)

        # Normalize COCO category_id -> contiguous label ids for torchvision.
        # torchvision detection models expect labels in [0, num_classes-1]
        # where labels are foreground classes only (background is implicit).
        cats = coco.get("categories", []) or []
        cat_ids = []
        for c in cats:
            try:
                cat_ids.append(int(c.get("id")))
            except Exception:
                pass
        # Fallback: infer unique ids from annotations.
        if not cat_ids:
            for ann in anns:
                try:
                    cat_ids.append(int(ann.get("category_id")))
                except Exception:
                    continue
            cat_ids = sorted(set(cat_ids))

        self.category_id_to_label: dict[int, int] = {cid: i for i, cid in enumerate(sorted(set(cat_ids)))}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_entry = self.images[idx]
        img_id = int(img_entry["id"])
        file_name = str(img_entry.get("file_name"))
        img_path = self.images_root / Path(file_name)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        from PIL import Image as PILImage
        image = PILImage.open(img_path).convert("RGB")

        # Prepare target dict
        anns = self.anns_by_image_id.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            bbox = ann.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            x = float(x); y = float(y); w = float(w); h = float(h)
            if w <= 0 or h <= 0:
                continue
            # torchvision expects boxes in xyxy
            boxes.append([x, y, x + w, y + h])
            cid = int(ann.get("category_id"))
            # Map COCO category_id -> contiguous torchvision label id
            labels.append(int(self.category_id_to_label.get(cid, 0)))
            areas.append(float(w * h))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        t = __import__("torch")
        boxes_t = t.tensor(boxes, dtype=t.float32)
        labels_t = t.tensor(labels, dtype=t.int64)
        image_id_t = t.tensor([img_id], dtype=t.int64)
        areas_t = t.tensor(areas, dtype=t.float32)
        iscrowd_t = t.tensor(iscrowd, dtype=t.int64)

        target: dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": image_id_t,
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            from torchvision.transforms import functional as F
            image = F.to_tensor(image)

        return image, target


def _collate_fn(batch):
    return tuple(zip(*batch))


def train_faster_rcnn(
    dataset_dir: str | Path,
    epochs: int = 10,
    num_classes: int = 2,
    device: str = "cpu",
    output_model: str | Path = "fasterrcnn_detector.pth",
) -> Dict[str, Any]:
    """Train Faster R-CNN via torchvision using COCO bbox annotations."""
    dataset_dir = _require_dir(dataset_dir, "dataset_dir")

    try:
        torch = __import__("torch")
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torch.utils.data import DataLoader

        device_str = str(device)
        try:
            device_actual = "cuda" if torch.cuda.is_available() and ("cuda" in device_str or device_str == "gpu") else "cpu"
        except Exception:
            device_actual = device_str

        coco_train, coco_test, project_name = _load_coco_detections_for_dataset(dataset_dir)


        # dataset_split contains both train/test images; COCO images point into that.
        # We'll try dataset_split/<split>/file_name resolution first.
        # For COCO import used by this repo, file_name is typically basename.
        repo_root = dataset_dir.resolve().parent
        images_train_root = repo_root / "dataset_split" / "train"
        images_test_root = repo_root / "dataset_split" / "test"

        ds_train = _CocoDetectionDataset(coco_train, images_train_root)
        ds_test = _CocoDetectionDataset(coco_test, images_test_root)

        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
        model.to(device_actual)


        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # a lightweight scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # torchvision detection models require custom collate
        # Faster R-CNN is memory-hungry; batch_size=1 is much more stable on smaller GPUs.
        batch_size = 1
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate_fn)


        model.train()
        for epoch in range(1, epochs + 1):
            lr_scheduler.step()
            epoch_loss = 0.0

            for images, targets in dl_train:

                images = [img.to(device_actual) for img in images]
                targets = [{k: v.to(device_actual) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += float(losses.detach().cpu())

            # best-effort: report average loss
            avg_loss = epoch_loss / max(1, len(dl_train))

        out_path = Path(output_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)

        return TrainResult(
            ok=True,
            model_path=str(out_path),
            metrics={"epochs": epochs, "num_classes_including_background": num_classes},
            logs={
                "project": project_name,
                "training_status": "real",
                "dataset_images_train": len(ds_train),
                "dataset_images_test": len(ds_test),
                "loss_scheduler": "StepLR",
            },
        ).to_dict()

    except Exception as e:
        return TrainResult(ok=False, error=str(e)).to_dict()


def train_ssd(
    dataset_dir: str | Path,
    epochs: int = 10,
    num_classes: int = 2,
    device: str = "cpu",
    output_model: str | Path = "ssd_detector.pth",
) -> Dict[str, Any]:
    """Train SSD via torchvision using COCO bbox annotations."""
    dataset_dir = _require_dir(dataset_dir, "dataset_dir")

    try:
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from torch.utils.data import DataLoader

        coco_train, coco_test, project_name = _load_coco_detections_for_dataset(dataset_dir)

        repo_root = dataset_dir.resolve().parent
        images_train_root = repo_root / "dataset_split" / "train"
        images_test_root = repo_root / "dataset_split" / "test"

        ds_train = _CocoDetectionDataset(coco_train, images_train_root)
        ds_test = _CocoDetectionDataset(coco_test, images_test_root)

        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
        device_str = str(device)
        try:
            device_actual = "cuda" if torch.cuda.is_available() and ("cuda" in device_str or device_str == "gpu") else "cpu"
        except Exception:
            device_actual = device_str
        model.to(device_actual)


        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        batch_size = 2
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate_fn)

        model.train()
        for epoch in range(1, epochs + 1):
            lr_scheduler.step()
            epoch_loss = 0.0
            for images, targets in dl_train:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += float(losses.detach().cpu())

            _avg_loss = epoch_loss / max(1, len(dl_train))

        out_path = Path(output_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)

        return TrainResult(
            ok=True,
            model_path=str(out_path),
            metrics={"epochs": epochs, "num_classes_including_background": num_classes},
            logs={
                "project": project_name,
                "training_status": "real",
                "dataset_images_train": len(ds_train),
                "dataset_images_test": len(ds_test),
                "loss_scheduler": "StepLR",
            },
        ).to_dict()

    except Exception as e:
        return TrainResult(ok=False, error=str(e)).to_dict()


