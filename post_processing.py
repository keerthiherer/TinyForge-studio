"""
Post-Processing Script for IoT ML

Applies common post-processing steps to saved NumPy model outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_utils import read_choice


OPTIONS = ["Thresholding", "Non-Maximum Suppression (NMS)", "Label Mapping"]


def threshold_outputs(outputs: np.ndarray, threshold: float) -> np.ndarray:
    return (outputs >= threshold).astype(np.int32)


def nms(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Apply NMS to boxes shaped [x1, y1, x2, y2, score, ...].

    If inputs don't look like bounding boxes, raise a clear error.
    """
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] < 5:
        raise ValueError(
            "NMS expects an array shaped [N, 5+] with x1,y1,x2,y2,score columns. "
            f"Got shape {boxes.shape}."
        )

    order = boxes[:, 4].argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]

        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        current_area = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        rest_area = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = current_area + rest_area - intersection
        iou = intersection / np.maximum(union, 1e-12)
        order = rest[iou <= iou_threshold]
    return boxes[keep]


def safe_nms(outputs: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Run NMS only when outputs look like bounding boxes.

    For classification-style outputs, this prevents the UI pipeline from failing.
    """
    arr = np.asarray(outputs)
    if arr.ndim == 2 and arr.shape[1] >= 5:
        return nms(arr.astype(float), iou_threshold)

    # Classification outputs: fall back to thresholding as a reasonable default.
    # If users want real NMS, they should provide a detection model output.
    threshold = 0.5
    if arr.ndim >= 2:
        threshold = float(np.median(arr)) if np.isfinite(arr).all() else 0.5
    return threshold_outputs(arr, threshold)




def map_labels(outputs: np.ndarray, mapping_path: str) -> np.ndarray:
    with open(mapping_path, encoding="utf-8") as handle:
        mapping = json.load(handle)
    flat = outputs.reshape(-1)
    mapped = [mapping.get(str(int(value)), mapping.get(str(value), str(value))) for value in flat]
    return np.array(mapped, dtype=object).reshape(outputs.shape)


def main() -> None:
    output_path = input("Enter path to model output file (.npy): ").strip()
    outputs = np.load(output_path, allow_pickle=True)
    print("Available post-processing options:")
    option = read_choice("Select post-processing option (number): ", OPTIONS)

    if option == "Thresholding":
        threshold = float(input("Threshold value [0.5]: ").strip() or "0.5")
        processed = threshold_outputs(outputs, threshold)
    elif option == "Non-Maximum Suppression (NMS)":
        iou_threshold = float(input("IoU threshold [0.5]: ").strip() or "0.5")
        processed = nms(np.asarray(outputs, dtype=float), iou_threshold)
    else:
        mapping_path = input("Enter label mapping JSON path: ").strip()
        processed = map_labels(outputs, mapping_path)

    output_file = Path(input("Output file [post_processed_output.npy]: ").strip() or "post_processed_output.npy")
    np.save(output_file, processed)
    print(f"Post-processing complete. Results saved as {output_file}")


if __name__ == "__main__":
    main()

