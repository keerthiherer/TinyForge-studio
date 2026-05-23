"""Local web UI for the IoT ML workflow."""

from __future__ import annotations

import contextlib
import csv
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, send_file
from werkzeug.utils import secure_filename

from sklearn.metrics import classification_report, confusion_matrix

from deploy_model import FORMATS, copy_or_package
from eon_tuner_like import IOT_DEVICES, estimate_ram_mb, suggest_hyperparams
from generate_features import extract_features_character_text, process_data, process_dataset, save_features

from ml_utils import (
    DEFAULT_MODEL_PATH,
    build_classifier,
    ensure_2d,
    load_artifact,
    load_features,
    load_labels,
    metric_summary,
    save_artifact,
)
from object_detection_workflow import (
    detection_dataset_summary,
    ensure_detection_dataset,
    load_class_names,
    prepare_yolo_dataset,
)

from post_processing import map_labels, nms, safe_nms, threshold_outputs
from split_and_store_data import detect_and_split
from labeling_manifest import load_or_create_manifest, load_label_mapping, save_label_mapping, write_y_npy

# Object detection (COCO annotation support for bbox labeling)
from object_detection_workflow import prepare_yolo_dataset

from PIL import Image

from ultralytics_detectors import detect_torchvision_detector, detect_yolo, train_faster_rcnn, train_ssd, train_yolo


def _project_dir(project: str) -> Path:
    safe_project = "".join(ch for ch in str(project).strip() if ch.isalnum() or ch in {"-", "_"}) or "default"
    return COCO_STORE_DIR / safe_project



def _coco_split_path(project: str, split: str) -> Path:
    split = split.strip().lower()
    if split not in {"train", "test"}:
        split = "train"
    return _project_dir(project) / f"coco_{split}.json"


def _load_coco_file(project: str, split: str) -> dict[str, Any]:
    path = _coco_split_path(project, split)
    if not path.is_file():
        return {
            "info": {"description": "iotml coco annotations"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # minimal shape validation
            payload.setdefault("images", [])
            payload.setdefault("annotations", [])
            payload.setdefault("categories", [])
            return payload
    except Exception:
        pass
    return {
        "info": {"description": "iotml coco annotations"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }


def _save_coco_file(project: str, split: str, coco: dict[str, Any]) -> None:
    path = _coco_split_path(project, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=2), encoding="utf-8")


def _next_int_id(items: list[dict[str, Any]], key: str, fallback: int = 1) -> int:
    max_id = fallback - 1
    for it in items:
        try:
            v = int(it.get(key))
            if v > max_id:
                max_id = v
        except Exception:
            continue
    return max_id + 1


def _ensure_category(coco: dict[str, Any], category_name: str) -> tuple[int, str]:
    name = str(category_name).strip() or "object"
    for cat in coco.get("categories", []):
        if str(cat.get("name")) == name:
            return int(cat.get("id")), name

    cat_id = _next_int_id(coco.get("categories", []), "id", fallback=1)
    coco.setdefault("categories", []).append({"id": cat_id, "name": name})
    return cat_id, name


def _find_image_entry(coco: dict[str, Any], image_filename: str) -> dict[str, Any] | None:
    for img in coco.get("images", []):
        if str(img.get("file_name")) == str(image_filename):
            return img
    return None



def _get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        w, h = im.size
    return int(w), int(h)


app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
COCO_STORE_DIR = ROOT / "coco_store"

JOB_STORE_DIR = ROOT / ".iotml_jobs"
UPLOAD_DIR = ROOT / ".iotml_uploads"
DETECTION_OUTPUT_DIR = ROOT / "static" / "detection_results"
JOBS: dict[str, dict[str, Any]] = {}
# In-memory store for auto-label proposals: {(project, split, image_filename): [proposals]}
AUTO_LABEL_PROPOSALS: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
from werkzeug.exceptions import HTTPException
def _image_size_for_coco(project: str, split: str, image_filename: str) -> tuple[int, int]:
    """Get image size from COCO or disk."""
    coco = _load_coco_file(project, split)
    img_entry = _find_image_entry(coco, image_filename)
    if img_entry and "width" in img_entry and "height" in img_entry:
        return int(img_entry["width"]), int(img_entry["height"])
    img_path = _split_dir(split) / Path(image_filename)
    if img_path.exists():
        return _get_image_size(img_path)
    raise FileNotFoundError(f"Image not found: {image_filename}")


# --- AUTO-LABEL ENDPOINTS ---
@app.post("/api/auto-label/predict")
def auto_label_predict():
    payload = request.get_json(force=True)
    project = str(payload.get("project", "default")).strip() or "default"
    split = str(payload.get("split", "train")).strip() or "train"
    image_filename = str(payload.get("image_filename", "")).strip()
    model_path = payload.get("model_path") or "runs/detect/iotml-yolo/weights/best.pt"
    confidence = float(payload.get("confidence", 0.25))
    imgsz = int(payload.get("imgsz", 640))
    device = payload.get("device", "cpu")
    if not image_filename:
        return jsonify({"ok": False, "error": "image_filename is required"}), 400
    img_path = _split_dir(split) / Path(image_filename)
    if not img_path.exists():
        return jsonify({"ok": False, "error": f"Image not found: {image_filename}"}), 400
    # Run YOLO detection (can extend to other detectors)
    try:
        result = detect_yolo(
            model_path=model_path,
            image_path=img_path,
            output_path=DETECTION_OUTPUT_DIR / f"auto_{uuid.uuid4().hex}.jpg",
            confidence=confidence,
            imgsz=imgsz,
            device=device,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    # Convert detections to proposal format: {id,label,confidence,bbox_px:[x,y,w,h]}
    proposals = []
    for i, det in enumerate(result.get("detections", [])):
        box = det["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        w, h = x2 - x1, y2 - y1
        proposals.append({
            "id": f"auto_{i}",
            "label": det["label"],
            "confidence": det["confidence"],
            "bbox_px": [x1, y1, w, h],
        })
    # Store in-memory
    AUTO_LABEL_PROPOSALS[(project, split, image_filename)] = proposals
    return jsonify({"ok": True, "proposals": proposals})


@app.post("/api/auto-label/approve")
def auto_label_approve():
    payload = request.get_json(force=True)
    project = str(payload.get("project", "default")).strip() or "default"
    split = str(payload.get("split", "train")).strip() or "train"
    image_filename = str(payload.get("image_filename", "")).strip()
    if not image_filename:
        return jsonify({"ok": False, "error": "image_filename is required"}), 400
    proposals = AUTO_LABEL_PROPOSALS.get((project, split, image_filename))
    if not proposals:
        return jsonify({"ok": False, "error": "No proposals to approve for this image."}), 400
    # Get image size
    try:
        img_w, img_h = _image_size_for_coco(project, split, image_filename)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    # Prepare annotation payload for /api/annotations/save
    ann_payload = {
        "project": project,
        "split": split,
        "image_filename": image_filename,
        "annotations": [],
    }
    for prop in proposals:
        x, y, w, h = prop["bbox_px"]
        # Convert to normalized xywh center
        xc = (x + w / 2.0) / img_w
        yc = (y + h / 2.0) / img_h
        norm = [xc, yc, w / img_w, h / img_h]
        ann_payload["annotations"].append({
            "annotation_id": None,
            "category_name": prop["label"],
            "bbox_norm_xywh": norm,
        })
    # Call the save logic directly
    with app.test_request_context(json=ann_payload):
        resp = annotations_save()
    # Remove proposals after approval
    AUTO_LABEL_PROPOSALS.pop((project, split, image_filename), None)
    return resp
JOBS_LOCK = threading.Lock()
MAX_JOB_LOG_LINES = 2000


def now_for_log() -> str:
    return datetime.now().strftime("%H:%M:%S")


def append_job_log(job_id: str, message: str, stream: str = "job") -> None:
    if not message:
        return
    clean = str(message).rstrip()
    if not clean:
        return
    line = f"[{now_for_log()}] {stream}> {clean}"
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        logs.append(line)
        if len(logs) > MAX_JOB_LOG_LINES:
            del logs[: len(logs) - MAX_JOB_LOG_LINES]
        persist_job_unlocked(job)


def job_file_path(job_id: str) -> Path:
    safe_id = "".join(ch for ch in job_id if ch.isalnum() or ch in {"-", "_"})
    return JOB_STORE_DIR / f"{safe_id}.json"


def persist_job_unlocked(job: dict[str, Any]) -> None:
    """Best-effort disk snapshot so a dev-server restart does not look like an unknown job."""
    try:
        JOB_STORE_DIR.mkdir(exist_ok=True)
        path = job_file_path(str(job.get("id", "")))
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(json_safe(job), indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        pass


def load_persisted_job(job_id: str) -> dict[str, Any] | None:
    path = job_file_path(job_id)
    if not path.is_file():
        return None
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(job, dict):
        return None
    if job.get("id") != job_id:
        return None
    return job


def orphan_persisted_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") in {"queued", "running"}:
        job["status"] = "error"
        job["ok"] = False
        job["error"] = (
            "The Flask development server restarted while this job was running, "
            "so the in-memory worker was lost. Restart the app with the built-in "
            "`python web_app.py` command and retry."
        )
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        logs = job.setdefault("logs", [])
        logs.append(f"[{now_for_log()}] error> {job['error']}")
    return job


class JobLogWriter:
    def __init__(self, job_id: str, stream: str) -> None:
        self.job_id = job_id
        self.stream = stream
        self.buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.buffer += text.replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            append_job_log(self.job_id, line, self.stream)
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            append_job_log(self.job_id, self.buffer, self.stream)
        self.buffer = ""


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def form_float(name: str, default: float) -> float:
    raw = request.form.get(name, "").strip()
    return default if not raw else float(raw)


def form_int(name: str, default: int) -> int:
    raw = request.form.get(name, "").strip()
    return default if not raw else int(raw)


def form_path(name: str, default: str) -> str:
    return request.form.get(name, "").strip() or default


def resolve_workspace_path(path_text: str, default: str = "") -> Path:
    raw = (path_text or default).strip()
    if not raw:
        raise ValueError("Path is required.")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def comma_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def artifact_status() -> dict[str, Any]:
    files = [
        "train_features.npy",
        "test_features.npy",
        "y_train.npy",
        "y_test.npy",
        DEFAULT_MODEL_PATH,
        "model_diagnostics_report.json",
        "model_predictions.csv",
        "deployment_package",
        "runs",
        "fasterrcnn_detector.pth",
        "ssd_detector.pth",
    ]
    return {
        file_name: {
            "exists": (ROOT / file_name).exists(),
            "path": str(ROOT / file_name),
        }
        for file_name in files
    }


def run_split_data() -> dict[str, Any]:
    input_path = form_path("input_path", "")
    output_dir = form_path("output_dir", "dataset_split")
    target = form_path("target_column", "target")
    test_size = form_float("test_size", 0.2)
    detect_and_split(input_path, output_dir, target, test_size)
    return {"message": "Data split complete.", "output_dir": output_dir}


def run_create_impulse() -> dict[str, Any]:
    data_type = form_path("data_type", "image")
    classifier = form_path("classifier", "YOLO" if data_type == "image" else "RandomForest")
    input_shape = form_path("input_shape", "64x64x1")
    impulse = {
        "data_type": data_type,
        "classifier": classifier,
        "input_shape": input_shape,
        "preprocessing": {
            "image": ["resize", "normalize"],
            "audio": ["resample", "normalize"],
            "numerical": ["normalize", "impute_missing"],
            "character": ["tokenize", "pad_sequences"],
        }.get(data_type, []),
        "feature_extraction": {
            "image": ["object detection labels", "bounding boxes"],
            "audio": ["MFCC"],
            "numerical": ["raw table features"],
            "character": ["character count vector"],
        }.get(data_type, []),
    }
    output_file = Path(form_path("output_file", "impulse_config.json"))
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(impulse, handle, indent=2)
    return {"message": "Impulse configuration saved.", "output_file": output_file, "impulse": impulse}


def run_generate_features() -> dict[str, Any]:
    data_type = form_path("data_type", "image")
    derived = ACTIVE_CONSTRAINTS.get("derived") or {}
    warnings: list[str] = []
    # Hardware-aware caps (best-effort for legacy feature extraction)
    if data_type == "image" and derived:
        cap_w, cap_h = derived.get("input_resolution_cap", (64, 64))
        # generate_features_image() is hardcoded to 64x64 in generate_features.py,
        # so we only warn until generate_features is made constraint-driven.
        if min(cap_w, cap_h) < 64:
            warnings.append("Device input resolution cap is below 64x64; current feature extractor is fixed at 64x64." )


    train_path = form_path("train_path", "dataset_split/train")
    test_path = form_path("test_path", "dataset_split/test")
    train_features, train_labels = process_dataset(train_path, data_type)
    test_features, test_labels = process_dataset(test_path, data_type)
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
    return {"message": "Features generated.", "metadata": metadata}


def run_eon_tuner() -> dict[str, Any]:
    device_name = form_path("device_name", "esp32")
    profile = IOT_DEVICES[device_name]
    suggestions = suggest_hyperparams(profile)
    estimates = {
        model_type: {
            "params": params,
            "estimated_ram_mb": estimate_ram_mb(model_type, params),
            "fits_ram": estimate_ram_mb(model_type, params) <= profile["max_ram_mb"],
        }
        for model_type, params in suggestions.items()
    }
    return {"message": "Tuner suggestions ready.", "device": device_name, "profile": profile, "suggestions": estimates}


def run_train_model() -> dict[str, Any]:
    data_type = form_path("data_type", "numerical")
    model_name = form_path("model_name", "RandomForest")
    normalized_model = model_name.lower().replace("_", "-").replace(" ", "-")

    if "yolo" in normalized_model:
        return run_train_object_detection_yolo()
    if "faster" in normalized_model or "rcnn" in normalized_model:
        return run_train_object_detection_faster_rcnn()
    if normalized_model == "ssd" or "ssdlite" in normalized_model:
        return run_train_object_detection_ssd()

    if data_type == "image":
        raise ValueError(
            "Image workflows now use object detection models. "
            "Select YOLO, Faster R-CNN, or SSD from the model dropdown."
        )

    x_train = load_features(form_path("train_features", "train_features.npy"))
    y_train = load_labels(form_path("train_labels", "y_train.npy"), expected_len=len(x_train))
    classifier = build_classifier(model_name)
    classifier.fit(x_train, y_train)

    metadata = {
        "data_type": data_type,
        "model_name": model_name,
        "feature_count": int(x_train.shape[1]),
        "classes": sorted(str(label) for label in np.unique(y_train)),
    }

    test_features_path = Path(form_path("test_features", "test_features.npy"))
    test_labels_path = Path(form_path("test_labels", "y_test.npy"))
    metrics = None
    if test_features_path.is_file() and test_labels_path.is_file():
        x_test = load_features(test_features_path)
        y_test = load_labels(test_labels_path, expected_len=len(x_test))
        predictions = classifier.predict(x_test)
        metrics = metric_summary(y_test, predictions)
        metadata["last_test_metrics"] = {key: value for key, value in metrics.items() if key != "report"}

    output_model = save_artifact(form_path("output_model", DEFAULT_MODEL_PATH), classifier, metadata)
    return {"message": "Model trained and saved.", "model_path": output_model, "metadata": metadata, "metrics": metrics}


def _object_detection_class_count(dataset_dir: str, labels_path: str = "", default: int = 2) -> tuple[int, dict[str, Any]]:
    summary = detection_dataset_summary(dataset_dir).to_dict()
    class_names = list(summary.get("artifacts", {}).get("class_names", []))

    if labels_path:
        class_names = load_class_names(labels_path) or class_names

    # Torchvision detectors include background as class 0.
    return max(default, len(class_names) + 1), {"summary": summary, "class_names": class_names}


def run_train_object_detection_yolo() -> dict[str, Any]:
    dataset_dir = form_path("dataset_dir", "dataset_split")
    dataset_check = ensure_detection_dataset(dataset_dir)
    if not dataset_check.ok:
        dataset_check = prepare_yolo_dataset(dataset_dir)
    if not dataset_check.ok:
        raise ValueError(dataset_check.error or "Detection dataset is not ready.")

    training_dataset_dir = Path(dataset_check.artifacts.get("dataset_yaml", dataset_dir)).parent

    result = train_yolo(
        dataset_dir=training_dataset_dir,
        model=form_path("model", "yolo11n.yaml"),
        epochs=form_int("epochs", 10),
        imgsz=form_int("imgsz", 640),
        batch=form_int("batch", 16),
        device=form_path("device", "cpu"),
        output_dir=form_path("output_dir", "runs/detect"),
        run_name=form_path("run_name", "iotml-yolo"),
    )
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "YOLO training failed.")

    return {
        "message": "YOLO object detector training complete.",
        "detector": "YOLO",
        "model_path": result.get("model_path"),
        "dataset": dataset_check.to_dict(),
        "training": result,
    }


def run_train_object_detection_faster_rcnn() -> dict[str, Any]:
    dataset_dir = form_path("dataset_dir", "dataset_split")
    labels_path = request.form.get("labels_path", "").strip()
    inferred_num_classes, dataset_info = _object_detection_class_count(dataset_dir, labels_path)
    output_model = form_path("output_model", "fasterrcnn_detector.pth")
    if output_model == DEFAULT_MODEL_PATH:
        output_model = "fasterrcnn_detector.pth"
    result = train_faster_rcnn(
        dataset_dir=dataset_dir,
        epochs=form_int("epochs", 10),
        num_classes=form_int("num_classes", inferred_num_classes),
        device=form_path("device", "cpu"),
        output_model=output_model,
    )
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "Faster R-CNN setup failed.")

    return {
        "message": "Faster R-CNN detector artifact created.",
        "detector": "Faster R-CNN",
        "model_path": result.get("model_path"),
        "dataset": dataset_info,
        "training": result,
    }


def run_train_object_detection_ssd() -> dict[str, Any]:
    dataset_dir = form_path("dataset_dir", "dataset_split")
    labels_path = request.form.get("labels_path", "").strip()
    inferred_num_classes, dataset_info = _object_detection_class_count(dataset_dir, labels_path)
    output_model = form_path("output_model", "ssd_detector.pth")
    if output_model == DEFAULT_MODEL_PATH:
        output_model = "ssd_detector.pth"
    result = train_ssd(
        dataset_dir=dataset_dir,
        epochs=form_int("epochs", 10),
        num_classes=form_int("num_classes", inferred_num_classes),
        device=form_path("device", "cpu"),
        output_model=output_model,
    )
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "SSD setup failed.")

    return {
        "message": "SSD detector artifact created.",
        "detector": "SSD",
        "model_path": result.get("model_path"),
        "dataset": dataset_info,
        "training": result,
    }


def run_retrain_model() -> dict[str, Any]:
    artifact = load_artifact(form_path("model_path", DEFAULT_MODEL_PATH))
    estimator = artifact["estimator"]
    metadata = dict(artifact.get("metadata", {}))
    x_train = load_features(form_path("train_features", "train_features.npy"))
    y_train = load_labels(form_path("train_labels", "y_train.npy"), expected_len=len(x_train))
    estimator.fit(x_train, y_train)
    metadata.update(
        {
            "retrained": True,
            "feature_count": int(x_train.shape[1]),
            "classes": sorted(str(label) for label in np.unique(y_train)),
        }
    )
    output_model = save_artifact(form_path("output_model", "iot_model_retrained.pkl"), estimator, metadata)
    return {"message": "Model retrained.", "model_path": output_model, "metadata": metadata}


def run_model_testing() -> dict[str, Any]:
    artifact = load_artifact(form_path("model_path", DEFAULT_MODEL_PATH))
    estimator = artifact["estimator"]
    x_test = load_features(form_path("test_features", "test_features.npy"))
    y_test = load_labels(form_path("test_labels", "y_test.npy"), expected_len=len(x_test))
    predictions = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test) if hasattr(estimator, "predict_proba") else None
    metrics = metric_summary(y_test, predictions)
    labels = np.unique(np.concatenate([y_test.reshape(-1), predictions.reshape(-1)]))
    report = {
        "artifact": artifact.get("metadata", {}),
        "test_shape": list(x_test.shape),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }
    report_path = Path(form_path("report_path", "model_diagnostics_report.json"))
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2)
    predictions_path = Path("model_predictions.csv")
    probability_classes = [str(label) for label in getattr(estimator, "classes_", [])]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["actual", "predicted", "correct"] + [f"probability_{label}" for label in probability_classes]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (actual, predicted) in enumerate(zip(y_test, predictions)):
            row = {"actual": actual, "predicted": predicted, "correct": actual == predicted}
            if probabilities is not None:
                for class_index, label in enumerate(probability_classes):
                    row[f"probability_{label}"] = float(probabilities[row_index, class_index])
            writer.writerow(row)
    return {
        "message": "Model testing complete.",
        "report_path": report_path,
        "predictions_path": predictions_path,
        "report": report,
    }


def run_live_classification() -> dict[str, Any]:
    artifact = load_artifact(form_path("model_path", DEFAULT_MODEL_PATH))
    estimator = artifact["estimator"]
    metadata = artifact.get("metadata", {})
    source_type = form_path("source_type", "numeric")
    data_type = form_path("data_type", metadata.get("data_type", "numerical"))
    if source_type == "file":
        sample = ensure_2d(process_data(form_path("sample_path", ""), data_type))
    elif source_type == "text":
        sample = extract_features_character_text(form_path("sample_text", "")).reshape(1, -1)
    else:
        sample = np.fromstring(form_path("numeric_row", ""), sep=",", dtype=float).reshape(1, -1)

    expected_features = metadata.get("feature_count")
    if expected_features is not None and sample.shape[1] != int(expected_features):
        raise ValueError(f"Sample has {sample.shape[1]} features, but the model expects {expected_features}.")

    prediction = estimator.predict(sample)[0]
    probabilities = None
    if hasattr(estimator, "predict_proba"):
        classes = [str(label) for label in estimator.classes_]
        probabilities = dict(zip(classes, [float(value) for value in estimator.predict_proba(sample)[0]]))
    return {"message": "Classification complete.", "prediction": str(prediction), "probabilities": probabilities}


def run_post_processing() -> dict[str, Any]:
    outputs = np.load(form_path("output_path", "model_output.npy"), allow_pickle=True)
    mode = form_path("mode", "threshold")
    if mode == "nms":
        processed = safe_nms(outputs, form_float("iou_threshold", 0.5))
    elif mode == "label_mapping":
        processed = map_labels(outputs, form_path("mapping_path", "label_mapping.json"))
    else:
        processed = threshold_outputs(outputs, form_float("threshold", 0.5))
    output_file = Path(form_path("save_path", "post_processed_output.npy"))
    np.save(output_file, processed)
    return {"message": "Post-processing complete.", "output_file": output_file, "shape": list(np.shape(processed))}


def run_deployment() -> dict[str, Any]:
    model_path = Path(form_path("model_path", DEFAULT_MODEL_PATH))
    target_format = form_path("target_format", FORMATS[0])
    output_dir = Path(form_path("output_dir", "deployment_package"))
    metadata = {"target_format": target_format, "source_model": str(model_path)}
    if model_path.suffix.lower() == ".pkl":
        metadata["model_metadata"] = load_artifact(model_path).get("metadata", {})
    output_path = copy_or_package(model_path, output_dir, target_format, metadata)
    return {"message": "Deployment package created.", "output_path": output_path, "metadata": metadata}


def run_model_outputs() -> dict[str, Any]:
    """Create model_output.npy required by post-processing.

    For sklearn classifiers this stores per-sample outputs as either:
    - predict_proba(x_test) if available
    - else predictions repeated as a 2D array
    """
    artifact = load_artifact(form_path("model_path", DEFAULT_MODEL_PATH))
    estimator = artifact["estimator"]
    x_test = load_features(form_path("test_features", "test_features.npy"))
    y_test = load_labels(form_path("test_labels", "y_test.npy"), expected_len=len(x_test))

    # Prefer probability outputs when available.
    outputs = None
    if hasattr(estimator, "predict_proba"):
        outputs = estimator.predict_proba(x_test)
    else:
        preds = estimator.predict(x_test)
        outputs = np.asarray(preds).reshape(-1, 1)

    output_file = Path(form_path("output_path", "model_output.npy"))
    np.save(output_file, outputs, allow_pickle=True)
    return {
        "message": "Model outputs saved.",
        "output_file": output_file,
        "output_shape": list(np.shape(outputs)),
        "y_test_shape": list(np.shape(y_test)),
    }


from microcontroller_export import export_for_mcu_pipeline





def run_export_for_mcu() -> dict[str, Any]:
    model_path = form_path("model_path", DEFAULT_MODEL_PATH)
    mcu = form_path("mcu", "arduino").lower().strip()
    output_dir = form_path("output_dir", "deployment_package")
    quantize_int8 = form_path("quantize_int8", "true").lower().strip() in {"1", "true", "yes", "y"}

    result = export_for_mcu_pipeline(
        model_path_text=model_path,
        mcu=mcu,
        output_dir_text=output_dir,
        quantize_int8=quantize_int8,
    )
    return result


# ---- Hardware-aware constraint engine (active device + budget) ----

ACTIVE_CONSTRAINTS: dict[str, Any] = {
    "device_key": None,
    "device_profile": None,
    "budget": {},
    "derived": {},
    "set_at": None,
}


def _clamp(name: str, value: float, min_v: float, max_v: float) -> float:
    return float(max(min_v, min(max_v, value)))


def _compute_derived_constraints(profile: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    # Cap user budget to device max. If user does not set, use device max.
    max_ram_mb = float(profile.get("max_ram_mb", 0.0))
    max_flash_mb = float(profile.get("max_flash_mb", 0.0))

    ram_mb = float(budget.get("ram_mb", max_ram_mb))
    flash_mb = float(budget.get("flash_mb", max_flash_mb))
    max_latency_ms = float(budget.get("max_latency_ms", 500.0))

    # Respect hard caps
    ram_mb = _clamp("ram_mb", ram_mb, 0.000001, max_ram_mb if max_ram_mb > 0 else ram_mb)
    flash_mb = _clamp("flash_mb", flash_mb, 0.000001, max_flash_mb if max_flash_mb > 0 else flash_mb)

    max_tensor_arena_kb = float(profile.get("max_tensor_arena_kb", max_ram_mb * 1024.0))
    # The user can request less, but not more than device capability.
    tensor_arena_kb = float(budget.get("tensor_arena_kb", max_tensor_arena_kb))
    tensor_arena_kb = _clamp("tensor_arena_kb", tensor_arena_kb, 1.0, max_tensor_arena_kb if max_tensor_arena_kb > 0 else tensor_arena_kb)

    max_input_res = profile.get("max_input_resolution", [64, 64])
    input_w, input_h = map(int, max_input_res)

    device_edge_only = bool(profile.get("edge_only", False))
    edge_only = bool(budget.get("edge_only", device_edge_only)) if "edge_only" in budget else device_edge_only

    supported_runtimes = list(profile.get("preferred_runtimes", [])) or list(profile.get("preferred_runtimes", []))
    supported_frameworks = list(profile.get("preferred_model_types", []))

    return {
        "max_ram_mb": max_ram_mb,
        "max_flash_mb": max_flash_mb,
        "ram_mb": ram_mb,
        "flash_mb": flash_mb,
        "max_latency_ms": max_latency_ms,
        "max_tensor_arena_kb": max_tensor_arena_kb,
        "tensor_arena_kb": tensor_arena_kb,
        "max_input_resolution": [input_w, input_h],
        "input_resolution_cap": [input_w, input_h],
        "edge_only": edge_only,
        "device_edge_only": device_edge_only,
        "supported_runtimes": supported_runtimes,
        "supported_frameworks": supported_frameworks,
        "accelerators": profile.get("accelerators", []),
        "opencv_expected": bool(profile.get("opencv_expected", False)),
        "quantization_preferred": bool(profile.get("quantization", False)),
    }


def _get_active_device_profile() -> dict[str, Any]:
    if ACTIVE_CONSTRAINTS.get("device_profile") is None:
        # Fall back to first device for stability.
        k = next(iter(IOT_DEVICES.keys()))
        return IOT_DEVICES[k]
    return ACTIVE_CONSTRAINTS["device_profile"]


def _apply_constraint_validation(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate and clamp known fields in payload based on active constraints."""
    warnings: list[str] = []
    derived = ACTIVE_CONSTRAINTS.get("derived") or {}

    out = dict(payload)
    # RAM/Flash/model-size caps only exist if profile was selected.
    if derived:
        ram_cap = derived.get("max_ram_mb")
        flash_cap = derived.get("max_flash_mb")
        if ram_cap is not None and "ram_mb" in out:
            if float(out["ram_mb"]) > float(ram_cap):
                warnings.append("Requested RAM exceeds device capability; clamped.")
                out["ram_mb"] = float(ram_cap)
        if flash_cap is not None and "flash_mb" in out:
            if float(out["flash_mb"]) > float(flash_cap):
                warnings.append("Requested Flash exceeds device capability; clamped.")
                out["flash_mb"] = float(flash_cap)

        # input resolution cap
        if "imgsz" in out:
            cap_w, cap_h = derived.get("input_resolution_cap", (64, 64))
            imgsz = int(out["imgsz"])
            cap_side = min(int(cap_w), int(cap_h))
            if imgsz > cap_side:
                warnings.append("Input size exceeds device capability; clamped.")
                out["imgsz"] = cap_side

        # quantization logic: prefer/force quantize for edge-only devices
        if derived.get("quantization_preferred") and "quantize_int8" in out:
            # if user said false but device prefers quantization, we do not force hard by spec;
            # we only warn and let user reduce further. But never exceed capability.
            if str(out.get("quantize_int8")).lower() in {"false", "0", "no"}:
                warnings.append("Device prefers quantization; consider enabling INT8 for size/latency.")

    return out, warnings


def run_set_target_device() -> dict[str, Any]:
    device_key = form_path("target_device", "").strip().lower()
    if not device_key:
        # if UI posted nothing, do nothing
        return {"ok": False, "error": "Missing target_device."}
    if device_key not in IOT_DEVICES:
        return {"ok": False, "error": f"Unknown target_device: {device_key}"}

    profile = IOT_DEVICES[device_key]
    ACTIVE_CONSTRAINTS["device_key"] = device_key
    ACTIVE_CONSTRAINTS["device_profile"] = profile
    ACTIVE_CONSTRAINTS["derived"] = _compute_derived_constraints(profile, ACTIVE_CONSTRAINTS.get("budget") or {})
    ACTIVE_CONSTRAINTS["set_at"] = datetime.now().isoformat(timespec="seconds")
    return {"ok": True, "message": "Target device set.", "device_key": device_key, "profile": profile, "constraints": ACTIVE_CONSTRAINTS.get("derived")}


def run_set_application_budget() -> dict[str, Any]:
    profile = _get_active_device_profile()

    budget: dict[str, Any] = {
        "ram_mb": form_float("ram_mb", float(profile.get("max_ram_mb", 128.0))),
        "flash_mb": form_float("flash_mb", float(profile.get("max_flash_mb", 1024.0))),
        "max_latency_ms": form_float("max_latency_ms", 200.0),
        "edge_only": (form_path("edge_only", "false").lower() in {"1", "true", "yes", "on"}),
    }


    # tensor arena is optional in current UI; derive from device default if missing.
    if request.form.get("tensor_arena_kb"):
        budget["tensor_arena_kb"] = form_float(
            "tensor_arena_kb", float(profile.get("max_tensor_arena_kb", 0.0))
        )



    ACTIVE_CONSTRAINTS["budget"] = budget
    ACTIVE_CONSTRAINTS["derived"] = _compute_derived_constraints(profile, budget)
    return {"ok": True, "message": "Application budget set.", "constraints": ACTIVE_CONSTRAINTS.get("derived")}


@app.get("/api/constraints")
def api_get_constraints():
    return jsonify({"ok": True, "constraints": ACTIVE_CONSTRAINTS.get("derived"), "device": ACTIVE_CONSTRAINTS.get("device_profile")})


# --- ImageLabeler import/open endpoints ---
@app.post("/api/imagelabeler/open")
def imagelabeler_open():
    """Open ImageLabeler on the local machine.

    NOTE: The exact executable path may differ on your system.
    If this fails, check the server console output and update IMAGE_LABELER_EXE.
    """
    project = str(request.args.get("project", "default")).strip() or "default"
    split = str(request.args.get("split", "train")).strip() or "train"

    if split not in {"train", "test"}:
        split = "train"

    image_labeler_dir = ROOT / "ImageLabeler"

    # Try to locate a windows binary under ImageLabeler/bin or ImageLabeler root.
    # You can adjust this if your clone includes a packaged .exe elsewhere.
    candidates = [
        image_labeler_dir / "bin" / "ImageLabeler.exe",
        image_labeler_dir / "ImageLabeler.exe",
    ]
    exe = next((c for c in candidates if c.exists()), None)

    if exe is None:
        # Fallback: tell frontend how to run manually.
        return jsonify({
            "ok": False,
            "error": "ImageLabeler executable not found. Look under ImageLabeler/bin or ImageLabeler root."
        }), 404

    import subprocess
    img_dir = _split_dir(split)

    # Best-effort: open app without file args (app is manual).
    # If you later wire argv support, pass img_dir as a starting folder.
    subprocess.Popen([str(exe)])

    return jsonify({"ok": True})


@app.post("/api/imagelabeler/import")
def imagelabeler_import():
    """Import ImageLabeler detection JSONs into our COCO store.

    ImageLabeler outputs per-image JSON files:
      *_detect_annotations.json
    and a labels.json file per directory (labels in that JSON include label ids etc).

    We import all *_detect_annotations.json found under dataset_split/<split>.
    """
    payload_project = str(request.args.get("project", "default")).strip() or "default"
    payload_split = str(request.args.get("split", "train")).strip() or "train"
    if payload_split not in {"train", "test"}:
        payload_split = "train"

    # ImageLabeler JSON output is stored next to images inside dataset_split/<train|test>.
    # Some of your earlier imports may have generated files only under the test split.
    # To be robust, if the selected split has no *_detect_annotations.json, we also
    # scan the other split and import from there.
    split_root = _split_dir(payload_split)
    other_split = "test" if payload_split == "train" else "train"
    other_root = _split_dir(other_split)

    selected_has_json = split_root.exists() and any(split_root.rglob("*_detect_annotations.json"))
    if not selected_has_json and other_root.exists():
        scan_root = other_root
    else:
        scan_root = split_root

    if not scan_root.exists():
        return jsonify({"ok": False, "error": "Split directory not found."}), 404

    # Load authoritative class definitions from dataset_split/<split>/labels.json
    # ImageLabeler generates a global labels.json like: {"labels": [{"id": 1, "label": "sun", ...}, ...]}
    labels_json_path = _split_dir(payload_split) / "labels.json"
    if not labels_json_path.is_file():
        # If labels.json is missing for the selected split, try the other split.
        other_labels_json_path = _split_dir(other_split) / "labels.json"
        if other_labels_json_path.is_file():
            labels_json_path = other_labels_json_path
        else:
            return jsonify({"ok": False, "error": f"labels.json not found in dataset_split/{payload_split}/ or dataset_split/{other_split}/"}), 404

    labels_payload = json.loads(labels_json_path.read_text(encoding="utf-8"))
    labels_list = labels_payload.get("labels", []) if isinstance(labels_payload, dict) else []

    # Build stable mapping: class name -> stable COCO category id.
    # Also ensures coco has stable categories (by clearing existing categories).
    class_name_to_cat_id: dict[str, int] = {}
    stable_categories: list[dict[str, Any]] = []
    for item in labels_list:
        try:
            cid = int(item.get("id"))
            cname = str(item.get("label")).strip()
        except Exception:
            continue
        if not cname:
            continue
        class_name_to_cat_id[cname] = cid
        stable_categories.append({"id": cid, "name": cname})

    if not stable_categories:
        return jsonify({"ok": False, "error": "labels.json contained no usable labels."}), 400

    # Collect coco and ensure basic structure (write into the selected split)
    coco = _load_coco_file(payload_project, payload_split)

    # Force stable categories based on labels.json.
    coco["categories"] = list(stable_categories)

    # Prepare lookup: existing image entries by file_name
    img_by_file = {str(img.get("file_name")): img for img in coco.get("images", []) if "file_name" in img}

    updated = 0

    from PIL import Image as PILImage

    def _ensure_image(image_filename: str) -> dict[str, Any]:
        nonlocal coco, img_by_file, updated
        if image_filename in img_by_file:
            return img_by_file[image_filename]
        img_path = split_root / Path(image_filename)
        if not img_path.exists():
            # try basename-only
            img_path2 = split_root / Path(image_filename).name
            if img_path2.exists():
                img_path = img_path2
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found on disk: {image_filename}")
        w, h = _get_image_size(img_path)
        img_id = _next_int_id(coco.get("images", []), "id", fallback=1)
        entry = {"id": img_id, "file_name": image_filename, "width": w, "height": h}
        coco.setdefault("images", []).append(entry)
        img_by_file[image_filename] = entry
        return entry

    # Find all ImageLabeler per-image JSONs.
    # Newer workflow saves them directly next to each image, with names like:
    #   <image_filename>_<any_suffix>_<m|n>_detect_annotations.json
    # We pair each detect-JSON with its corresponding image by removing the
    # trailing "_detect_annotations" part.
    json_paths = list(scan_root.rglob("*_detect_annotations.json"))

    for jp in json_paths:
        try:
            rel_img = jp.relative_to(split_root)
            # ImageLabeler uses the original image filename + suffix.
            # Example: vid_01.jpg -> vid_01_detect_annotations.json
            stem = jp.stem  # includes _detect_annotations
            if stem.endswith("_detect_annotations"):
                # Our detect json filename already contains the image filename
                # and extra suffix parts from ImageLabeler.
                # Example:
                #   14646279002_9cdf97be97_n_detect_annotations.json
                #   -> image file is 14646279002_9cdf97be97_n.jpg (or .png etc)
                base = stem[: -len("_detect_annotations")]

                # Prefer common extensions.
                exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                found = None
                for ext in exts:
                    # Check in the same directory as the json
                    cand = jp.with_name(base + ext)
                    if cand.exists():
                        found = ext
                        break
                if found is None:
                    # Fallback: try basename match without extension probing.
                    found = ".jpg"

                image_filename = str(rel_img.parent / (base + found)).replace("\\", "/")
            else:
                # unknown pattern: skip
                continue

            doc = json.loads(jp.read_text(encoding="utf-8"))
            annotations = doc.get("annotations", [])

            img_entry = _ensure_image(image_filename)
            img_id = img_entry.get("id")
            img_w = int(img_entry.get("width"))
            img_h = int(img_entry.get("height"))

            # Keep an index of existing ann by (image_id, instance id) so edits replace.
            anns = coco.get("annotations", [])
            existing_by_key = {}
            for a in anns:
                if a.get("image_id") == img_id:
                    # ImageLabeler uses "id" as instance id
                    existing_by_key[(a.get("image_id"), a.get("category_id"), a.get("id"))] = a

            for anno in annotations:
                label = str(anno.get("label", "object")).strip() or "object"
                instance_id = anno.get("id", None)

                # points: [ [x1,y1], [x2,y2] ]
                pts = anno.get("points")
                if not isinstance(pts, list) or len(pts) != 2:
                    continue
                p1, p2 = pts[0], pts[1]
                x1 = float(p1[0]); y1 = float(p1[1])
                x2 = float(p2[0]); y2 = float(p2[1])
                left = min(x1, x2); top = min(y1, y2)
                w = abs(x2 - x1); h = abs(y2 - y1)

                if w <= 1 or h <= 1:
                    continue

                # Map annotation label to stable category id using labels.json.
                # If label not found in labels.json, skip this annotation.
                if label not in class_name_to_cat_id:
                    continue
                category_id = int(class_name_to_cat_id[label])

                # Build bbox in COCO top-left xywh
                bbox = [float(left), float(top), float(w), float(h)]

                # Create a deterministic COCO annotation id by using instance_id when possible.
                # If instance_id is numeric and unique, we try to map it.
                new_id = None
                if instance_id is not None:
                    try:
                        new_id = int(instance_id)
                    except Exception:
                        new_id = None

                # Determine COCO ann id
                if new_id is not None:
                    # Ensure id uniqueness; if conflict, allocate next
                    id_taken = any(int(a.get("id")) == new_id for a in coco.get("annotations", []))
                    if id_taken:
                        new_id = _next_int_id(coco.get("annotations", []), "id", fallback=1)
                else:
                    new_id = _next_int_id(coco.get("annotations", []), "id", fallback=1)

                coco.setdefault("annotations", []).append({
                    "id": int(new_id),
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "iscrowd": 0,
                })
                updated += 1

        except Exception:
            continue

    # Save coco store
    _save_coco_file(payload_project, payload_split, coco)
    return jsonify({"ok": True, "imported": updated, "images": len(coco.get("images", []))})



@app.get("/api/compatible-runtimes")
def api_compatible_runtimes():
    profile = ACTIVE_CONSTRAINTS.get("device_profile") or _get_active_device_profile()
    derived = ACTIVE_CONSTRAINTS.get("derived") or _compute_derived_constraints(profile, ACTIVE_CONSTRAINTS.get("budget") or {})
    supported_runtimes = profile.get("preferred_runtimes", [])
    # Also reflect edge_only preference into runtime list
    if derived.get("edge_only"):
        supported_runtimes = [r for r in supported_runtimes if "micro" in r or r in {"tflite_micro", "cmsis_nn", "esp_dl"} or True]
    return jsonify({"ok": True, "runtimes": supported_runtimes, "constraints": derived})


@app.get("/api/recommended-models")
def api_recommended_models():
    profile = ACTIVE_CONSTRAINTS.get("device_profile") or _get_active_device_profile()
    derived = ACTIVE_CONSTRAINTS.get("derived") or _compute_derived_constraints(profile, ACTIVE_CONSTRAINTS.get("budget") or {})
    preferred_model_types = profile.get("preferred_model_types", [])
    # Return them as high-level categories the frontend can map to UI choices
    return jsonify({"ok": True, "model_types": preferred_model_types, "constraints": derived})


@app.post("/api/estimate-memory")
def api_estimate_memory():
    payload = request.get_json(force=True, silent=True) or {}
    # best-effort estimation based on model_type + params from EON tuner-like
    model_type = str(payload.get("model_type", "cnn")).lower()
    params = payload.get("params") or {}
    estimated = estimate_ram_mb(model_type, params) if params else estimate_ram_mb(model_type, {"num_layers": 1, "filters": 8, "kernel_size": 3, "units": 16})
    cap = (ACTIVE_CONSTRAINTS.get("derived") or {}).get("ram_mb")
    ok = cap is None or estimated <= float(cap)
    return jsonify({"ok": True, "estimated_ram_mb": estimated, "fits_ram": ok, "ram_cap_mb": cap})


@app.post("/api/estimate-latency")
def api_estimate_latency():
    payload = request.get_json(force=True, silent=True) or {}
    # Heuristic latency estimator: base on input resolution + model_type weights
    model_type = str(payload.get("model_type", "cnn")).lower()
    imgsz = int(payload.get("imgsz", 64))
    derived = ACTIVE_CONSTRAINTS.get("derived") or {}
    cap = derived.get("max_latency_ms")
    # Rough: bigger input and cnn -> higher latency
    base = 30.0 if model_type in {"tinyml", "mlp", "cnn"} else 60.0
    est = base * (imgsz / 64.0) * (2.0 if model_type == "cnn" else 1.0)
    ok = cap is None or est <= float(cap)
    return jsonify({"ok": True, "estimated_latency_ms": est, "fits_latency": ok, "latency_cap_ms": cap})


@app.post("/api/set-constraints")
def api_set_constraints():
    payload = request.get_json(force=True, silent=True) or {}

    device_key = payload.get("target_device")
    if device_key:
        if device_key not in IOT_DEVICES:
            return jsonify({"ok": False, "error": f"Unknown target device: {device_key}"}), 400
        ACTIVE_CONSTRAINTS["device_key"] = device_key
        ACTIVE_CONSTRAINTS["device_profile"] = IOT_DEVICES[device_key]
    profile = _get_active_device_profile()
    budget = payload.get("budget") or {}
    ACTIVE_CONSTRAINTS["budget"] = budget
    ACTIVE_CONSTRAINTS["derived"] = _compute_derived_constraints(profile, budget)
    return jsonify({"ok": True, "constraints": ACTIVE_CONSTRAINTS.get("derived")})



ACTION_HANDLERS = {
    "split-data": run_split_data,
    "set-target-device": run_set_target_device,
    "set-application-budget": run_set_application_budget,


    "create-impulse": run_create_impulse,
    "generate-features": run_generate_features,
    "eon-tuner": run_eon_tuner,
    "train-model": run_train_model,
    "train-object-detection-yolo": run_train_object_detection_yolo,
    "train-object-detection-faster-rcnn": run_train_object_detection_faster_rcnn,
    "train-object-detection-ssd": run_train_object_detection_ssd,
    "retrain-model": run_retrain_model,
    "model-testing": run_model_testing,
    "model-outputs": run_model_outputs,
    "live-classification": run_live_classification,
    "post-processing": run_post_processing,
    "deployment": run_deployment,
    "export-mcu": run_export_for_mcu,
}




def _split_dir(split_name: str) -> Path:
    # dataset_split/train or dataset_split/test
    return ROOT / "dataset_split" / split_name


@app.get("/datasets/<project>/images/<path:rel_image_path>")
def datasets_project_image(project: str, rel_image_path: str):
    """Serve images for the COCO labeling UI.

    Frontend uses the COCO `images[].file_name` value as `rel_image_path`.
    In this repo those file_names are typically just the basename
    (e.g. `1038...jpg`) or may include a subpath.

    We resolve against both:
      dataset_split/train/<rel_image_path>
      dataset_split/test/<rel_image_path>

    and return a 404 if not found.
    """
    rel_image_path = str(Path(rel_image_path))
    rel_image_path_posix = Path(rel_image_path).as_posix()
    if ".." in Path(rel_image_path_posix).parts:
        return jsonify({"error": "invalid path"}), 404

    train_candidate = ROOT / "dataset_split" / "train" / Path(rel_image_path_posix)
    if train_candidate.is_file():
        return send_file(str(train_candidate))

    test_candidate = ROOT / "dataset_split" / "test" / Path(rel_image_path_posix)
    if test_candidate.is_file():
        return send_file(str(test_candidate))

    return jsonify({"error": "image not found"}), 404



@app.get("/static/<split>/<path:filename>")
def labeling_image_static(split: str, filename: str):
    # Serve dataset_split/<split>/<filename> as /static/<split>/<filename>
    # so templates/labeling.html can load it reliably.
    split = split.strip()
    if split not in {"train", "test"}:
        return jsonify({"error": "invalid split"}), 404

    dir_path = str(ROOT / "dataset_split" / split)
    # ensure we do not allow path traversal
    safe_name = Path(filename).name
    return send_from_directory(dir_path, safe_name)


@app.get("/labeling")
def labeling_page():
    """Launch a manual labeling workflow.

    Current approach: use the local ImageLabeler Qt application.
    The web UI only provides a "Label with ImageLabeler" button and a
    status/import action. The heavy bbox interaction is done in the desktop app.
    """
    project_name = request.args.get("project", "default").strip() or "default"
    split_name = request.args.get("split", "train").strip() or "train"
    if split_name not in {"train", "test"}:
        split_name = "train"

    return render_template("labeling_bbox.html", split_name=split_name, project_name=project_name)




@app.get("/api/annotations/classes")
def annotations_classes():
    project = request.args.get("project", "default").strip() or "default"
    split = request.args.get("split", "train").strip() or "train"
    coco = _load_coco_file(project, split)
    names = sorted({str(cat.get("name")) for cat in coco.get("categories", []) if str(cat.get("name")).strip()})
    if not names:
        names = ["object"]
    return jsonify({"ok": True, "classes": names})


@app.get("/api/annotations/queue")
def annotations_queue():
    project = request.args.get("project", "default").strip() or "default"
    split = request.args.get("split", "train").strip() or "train"
    if split not in {"train", "test"}:
        split = "train"

    # queue images from dataset_split/<split>
    split_path = _split_dir(split)
    if not split_path.exists():
        return jsonify({"ok": True, "files": [], "next_index": 0})

    files = []
    for p in sorted(split_path.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
            files.append(p.name)
        elif p.is_dir():
            for img in sorted(p.rglob("*")):
                if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
                    files.append(img.relative_to(split_path).as_posix())

    coco = _load_coco_file(project, split)
    annotated_by_image = set()
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        for img in coco.get("images", []):
            if img.get("id") == img_id:
                annotated_by_image.add(str(img.get("file_name")))

    next_index = 0
    for i, f in enumerate(files):
        if str(f) not in annotated_by_image:
            next_index = i
            break
    else:
        next_index = max(0, len(files) - 1) if files else 0

    return jsonify({"ok": True, "files": files, "next_index": next_index})


@app.get("/api/annotations/list")
def annotations_list():
    payload_project = request.args.get("project", "default").strip() or "default"
    split = request.args.get("split", "train").strip() or "train"
    image_filename = request.args.get("image", "").strip()

    if split not in {"train", "test"}:
        split = "train"
    if not image_filename:
        return jsonify({"ok": False, "error": "image query param is required"}), 400

    coco = _load_coco_file(payload_project, split)
    img_entry = _find_image_entry(coco, image_filename)

    img_w = img_h = None
    if img_entry is not None:
        try:
            img_w = int(img_entry.get("width"))
            img_h = int(img_entry.get("height"))
        except Exception:
            img_w = img_h = None

    if img_w is None or img_h is None:
        # compute from dataset_split image file path
        img_path = _split_dir(split) / Path(image_filename)
        if img_path.exists():
            img_w, img_h = _get_image_size(img_path)

    annotations = []
    # category map
    cat_by_id = {int(cat.get("id")): str(cat.get("name")) for cat in coco.get("categories", []) if "id" in cat and "name" in cat}

    if img_entry is not None:
        image_id = img_entry.get("id")
        for ann in coco.get("annotations", []):
            if ann.get("image_id") != image_id:
                continue
            bbox = ann.get("bbox")  # COCO bbox: [x,y,w,h] in pixels
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            if not img_w or not img_h:
                continue
            # convert to normalized xywh center
            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            norm = [xc, yc, w / img_w, h / img_h]
            annotations.append(
                {
                    "annotation_id": ann.get("id"),
                    "category_id": ann.get("category_id"),
                    "class_name": cat_by_id.get(int(ann.get("category_id")), "object"),
                    "bbox_norm": norm,
                    # frontend expects bbox_px and bbox_norm
                    "bbox_px": [x, y, w, h],
                }
            )

    return jsonify(
        {
            "ok": True,
            "image_width": img_w or 0,
            "image_height": img_h or 0,
            "annotations": annotations,
        }
    )


@app.post("/api/annotations/save")
def annotations_save():
    payload = request.get_json(force=True)
    project = str(payload.get("project", "default")).strip() or "default"
    split = str(payload.get("split", "train")).strip() or "train"
    image_filename = str(payload.get("image_filename", "")).strip() or ""

    if split not in {"train", "test"}:
        split = "train"
    if not image_filename:
        return jsonify({"ok": False, "error": "image_filename is required"}), 400

    coco = _load_coco_file(project, split)
    img_entry = _find_image_entry(coco, image_filename)

    if img_entry is None:
        img_id = _next_int_id(coco.get("images", []), "id", fallback=1)
        img_path = _split_dir(split) / Path(image_filename)
        if not img_path.exists():
            return jsonify({"ok": False, "error": f"Image not found in dataset_split: {image_filename}"}), 400
        img_w, img_h = _get_image_size(img_path)
        img_entry = {"id": img_id, "file_name": image_filename, "width": img_w, "height": img_h}
        coco.setdefault("images", []).append(img_entry)
    else:
        img_id = img_entry.get("id")

    img_w = int(img_entry.get("width"))
    img_h = int(img_entry.get("height"))

    annotations_in = payload.get("annotations", [])
    if not isinstance(annotations_in, list):
        annotations_in = []

    # Build fast indexes
    anns_by_id = {int(a.get("id")): a for a in coco.get("annotations", []) if "id" in a}

    updated_ids = set()
    for ann in annotations_in:
        ann_id_raw = ann.get("annotation_id")
        ann_id = None
        if ann_id_raw is not None and str(ann_id_raw).strip() != "":
            try:
                ann_id = int(ann_id_raw)
            except Exception:
                ann_id = None

        category_name = ann.get("category_name", "object")
        category_id, _ = _ensure_category(coco, category_name)
        bbox_norm_xywh = ann.get("bbox_norm_xywh") or ann.get("bbox_norm")

        if not isinstance(bbox_norm_xywh, (list, tuple)) or len(bbox_norm_xywh) != 4:
            return jsonify({"ok": False, "error": "bbox_norm_xywh must be [xc,yc,w,h]"}), 400
        xc, yc, nw, nh = map(float, bbox_norm_xywh)
        # convert to COCO pixel xywh (top-left)
        w = nw * img_w
        h = nh * img_h
        x = (xc * img_w) - w / 2.0
        y = (yc * img_h) - h / 2.0
        x, y, w, h = float(x), float(y), float(w), float(h)

        if ann_id is not None and ann_id in anns_by_id:
            coco_ann = anns_by_id[ann_id]
            coco_ann["category_id"] = int(category_id)
            coco_ann["image_id"] = int(img_id)
            coco_ann["bbox"] = [x, y, w, h]
            updated_ids.add(ann_id)
        else:
            new_id = _next_int_id(coco.get("annotations", []), "id", fallback=1)
            new_ann = {
                "id": new_id,
                "image_id": int(img_id),
                "category_id": int(category_id),
                "bbox": [x, y, w, h],
                "iscrowd": 0,
            }
            coco.setdefault("annotations", []).append(new_ann)
            anns_by_id[new_id] = new_ann
            updated_ids.add(new_id)

    # Optional cleanup: remove any annotations for this image not present in incoming payload
    new_annotations = []
    for a in coco.get("annotations", []):
        if a.get("image_id") == img_id:
            if int(a.get("id")) in updated_ids:
                new_annotations.append(a)
        else:
            new_annotations.append(a)
    coco["annotations"] = new_annotations

    _save_coco_file(project, split, coco)
    return jsonify({"ok": True, "saved": len(updated_ids)})


@app.post("/api/annotations/delete")
def annotations_delete():
    payload = request.get_json(force=True)
    project = str(payload.get("project", "default")).strip() or "default"
    split = str(payload.get("split", "train")).strip() or "train"
    image_filename = str(payload.get("image_filename", "")).strip() or ""
    ann_id_raw = payload.get("annotation_id")

    if split not in {"train", "test"}:
        split = "train"
    if not image_filename:
        return jsonify({"ok": False, "error": "image_filename is required"}), 400
    try:
        ann_id = int(ann_id_raw)
    except Exception:
        return jsonify({"ok": False, "error": "annotation_id must be int"}), 400

    coco = _load_coco_file(project, split)
    img_entry = _find_image_entry(coco, image_filename)
    if img_entry is None:
        return jsonify({"ok": True, "deleted": False})

    image_id = img_entry.get("id")
    before = len(coco.get("annotations", []))
    coco["annotations"] = [
        a
        for a in coco.get("annotations", [])
        if not (a.get("image_id") == image_id and int(a.get("id")) == ann_id)
    ]
    after = len(coco.get("annotations", []))
    _save_coco_file(project, split, coco)

    return jsonify({"ok": True, "deleted": before != after})


@app.get("/api/labeling/state")
def labeling_state():

    split_name = request.args.get("split", "train").strip() or "train"
    if split_name not in {"train", "test"}:
        raise HTTPException(400, "split must be 'train' or 'test'")

    split_path = _split_dir(split_name)
    manifest_path = ROOT / f"label_manifest_{split_path.name}.json"
    manifest = load_or_create_manifest(str(split_path), out_dir=str(ROOT))
    files = manifest.get("files", [])

    # Labels are stored per split: labeling_train.json / labeling_test.json
    labels = load_label_mapping(split_name, out_dir=str(ROOT))


    # next index = first unlabeled item
    next_index = 0
    for i, f in enumerate(files):
        if str(labels.get(f, '')).strip() == '':
            next_index = i
            break
    else:
        next_index = max(0, len(files) - 1)

    return {
        "files": files,
        "labels": labels,
        "next_index": next_index,
    }


@app.post("/api/labeling/set")
def labeling_set():
    payload = request.get_json(force=True)
    split_name = str(payload.get("split", "train")).strip()
    file_rel = str(payload.get("file", '')).strip()
    label = str(payload.get("label", '')).strip()

    if split_name not in {"train", "test"}:
        return jsonify({"ok": False, "error": "split must be train or test"}), 400
    if not file_rel:
        return jsonify({"ok": False, "error": "file is required"}), 400
    if not label:
        return jsonify({"ok": False, "error": "label is required"}), 400

    split_path = _split_dir(split_name)
    manifest = load_or_create_manifest(str(split_path), out_dir=str(ROOT))
    files = manifest.get("files", [])

    labels = load_label_mapping(split_name, out_dir=str(ROOT))
    labels[file_rel] = label
    save_label_mapping(split_name, labels, out_dir=str(ROOT))

    # compute next unlabeled index
    next_index = 0
    for i, f in enumerate(files):
        if str(labels.get(f, '')).strip() == '':
            next_index = i
            break
    else:
        next_index = max(0, len(files) - 1)

    return jsonify({"ok": True, "next_index": next_index})


@app.post("/api/labeling/build-y")
def labeling_build_y():

    payload = request.get_json(force=True)
    split_name = str(payload.get("split", "train")).strip()
    if split_name not in {"train", "test"}:
        return jsonify({"ok": False, "error": "split must be train or test"}), 400

    split_path = _split_dir(split_name)
    manifest = load_or_create_manifest(str(split_path), out_dir=str(ROOT))
    files = manifest.get("files", [])

    labels = load_label_mapping(split_name, out_dir=str(ROOT))
    try:
        result = write_y_npy(split_name, files, labels, out_dir=str(ROOT))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "result": result})



@app.route("/")
def index():
    return render_template("index.html", devices=IOT_DEVICES, deployment_formats=FORMATS)


@app.get("/api/status")
def status():
    return jsonify({"ok": True, "artifacts": artifact_status(), "devices": IOT_DEVICES, "deployment_formats": FORMATS})


def detection_image_from_request() -> Path:
    upload = request.files.get("image")
    if upload and upload.filename:
        original_name = secure_filename(upload.filename) or "capture.jpg"
        suffix = Path(original_name).suffix.lower() or ".jpg"
        UPLOAD_DIR.mkdir(exist_ok=True)
        image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        upload.save(image_path)
        return image_path

    image_path_text = request.form.get("image_path", "").strip()
    if not image_path_text:
        raise ValueError("Upload an image, capture a frame, or enter an image path.")
    image_path = resolve_workspace_path(image_path_text)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return image_path


@app.post("/api/detection/live")
def live_detection():
    try:
        model_type = form_path("model_type", "yolo").lower().replace("_", "-").replace(" ", "-")
        model_path = resolve_workspace_path(form_path("model_path", "runs/detect/iotml-yolo/weights/best.pt"))
        image_path = detection_image_from_request()
        confidence = form_float("confidence", 0.25)
        device = form_path("device", "cpu")
        output_name = f"{uuid.uuid4().hex}.jpg"
        output_path = DETECTION_OUTPUT_DIR / output_name

        if "yolo" in model_type:
            result = detect_yolo(
                model_path=model_path,
                image_path=image_path,
                output_path=output_path,
                confidence=confidence,
                imgsz=form_int("imgsz", 640),
                device=device,
            )
        elif "faster" in model_type or "rcnn" in model_type:
            result = detect_torchvision_detector(
                detector="faster-rcnn",
                model_path=model_path,
                image_path=image_path,
                output_path=output_path,
                confidence=confidence,
                num_classes=form_int("num_classes", 2),
                device=device,
                class_names=comma_list(request.form.get("class_names", "")),
            )
        elif model_type == "ssd" or "ssdlite" in model_type:
            result = detect_torchvision_detector(
                detector="ssd",
                model_path=model_path,
                image_path=image_path,
                output_path=output_path,
                confidence=confidence,
                num_classes=form_int("num_classes", 2),
                device=device,
                class_names=comma_list(request.form.get("class_names", "")),
            )
        else:
            raise ValueError(f"Unsupported advanced model type: {model_type}")

        result["annotated_url"] = f"/static/detection_results/{output_name}"
        result["source"] = image_path.name
        return jsonify({"ok": True, "result": json_safe(result), "artifacts": artifact_status()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "artifacts": artifact_status()}), 400


def _cleanup_artifacts_after_run(action: str | None = None):
    """Kill-switch cleanup.

    Per user request: delete meta/report files + all *.npy artifacts after a successful run.
    This helps avoid stale artifacts if a new model is trained/deployed.

    Note: `action` is accepted because the function name is (incorrectly) wired to the
    `/api/run/<action>` route in this codebase. We ignore it for cleanup.
    """

    # meta/report files

    meta_files = [
        "feature_metadata.json",
        "model_diagnostics_report.json",
        "training_metadata.json",
        "model_diagnostics.json",
        "deployment_package/deployment_metadata.json",
        "model_predictions.csv",
        "model_output.npy",  # included here for safety, but also matched by *.npy
    ]

    for rel in meta_files:
        p = ROOT / rel
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass

    # all npy artifacts (repo-root)
    try:
        for p in ROOT.glob("*.npy"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
    except Exception:
        pass

    # some workflows generate npy files inside deployment dirs or other folders; keep it conservative.

    # NOTE: This helper is intentionally used both by the cleanup route and by
    # the post-success cleanup inside `run_action`. It should NOT itself be a
    # Flask response.
    return




@app.post("/api/kill")
def kill_action():
    """Kill switch: cleanup artifacts immediately."""
    try:
        _cleanup_artifacts_after_run()
        return jsonify({"ok": True, "message": "Cleanup completed.", "artifacts": artifact_status()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "artifacts": artifact_status()}), 400


def execute_action(action: str) -> dict[str, Any]:
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        raise KeyError(f"Unknown action: {action}")
    result = handler()
    _cleanup_artifacts_after_run()
    return {"ok": True, "result": json_safe(result), "artifacts": artifact_status()}


@app.post("/api/run/<action>")
def run_action(action: str):
    if action not in ACTION_HANDLERS:
        return jsonify({"ok": False, "error": f"Unknown action: {action}"}), 404
    try:
        return jsonify(execute_action(action))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "artifacts": artifact_status()}), 400


def run_background_job(job_id: str, action: str, form_data: dict[str, list[str]]) -> None:
    append_job_log(job_id, f"Starting action: {action}")
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["started_at"] = datetime.now().isoformat(timespec="seconds")
            persist_job_unlocked(JOBS[job_id])

    stdout = JobLogWriter(job_id, "stdout")
    stderr = JobLogWriter(job_id, "stderr")
    try:
        with app.test_request_context(method="POST", data=form_data):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                payload = execute_action(action)
        stdout.flush()
        stderr.flush()
        append_job_log(job_id, f"Finished action: {action}")
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "complete"
            job["ok"] = True
            job["result"] = payload
            job["artifacts"] = payload.get("artifacts", {})
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            persist_job_unlocked(job)
    except Exception as exc:
        stdout.flush()
        stderr.flush()
        append_job_log(job_id, f"Error: {exc}", "error")
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "error"
            job["ok"] = False
            job["error"] = str(exc)
            job["artifacts"] = artifact_status()
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            persist_job_unlocked(job)


@app.post("/api/run-async/<action>")
def run_action_async(action: str):
    if action not in ACTION_HANDLERS:
        return jsonify({"ok": False, "error": f"Unknown action: {action}"}), 404

    job_id = uuid.uuid4().hex
    form_data = {key: request.form.getlist(key) for key in request.form.keys()}
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "action": action,
            "status": "queued",
            "ok": None,
            "logs": [f"[{now_for_log()}] job> Queued action: {action}"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": artifact_status(),
        }
        persist_job_unlocked(JOBS[job_id])

    thread = threading.Thread(target=run_background_job, args=(job_id, action, form_data), daemon=True)
    thread.start()
    return jsonify({"ok": True, "job_id": job_id, "job": JOBS[job_id]})


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            job = load_persisted_job(job_id)
            if job is None:
                return jsonify({"ok": False, "error": f"Unknown job: {job_id}"}), 404
            job = orphan_persisted_job(job)
            JOBS[job_id] = job
            persist_job_unlocked(job)
        return jsonify({"ok": True, "job": json_safe(job)})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
