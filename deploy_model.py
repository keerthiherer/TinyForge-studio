"""
Deployment Script for IoT ML Models

Packages a trained model artifact and metadata for device deployment. Native
conversion to TFLite, ONNX, or EIM requires converter-specific tooling, so this
script validates the input and creates an honest deployable package.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from ml_utils import DEFAULT_MODEL_PATH, load_artifact, read_choice


FORMATS = [
    "Edge Impulse package (.zip)",
    "TensorFlow Lite file/package (if .tflite exists)",
    "ONNX file/package (if .onnx exists)",
    "Keras H5 file/package (if .keras/.h5 exists)",
]



def optional_path(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def copy_or_package(model_path: Path, output_dir: Path, target_format: str, metadata: dict) -> Path:
    """Create deployment artifacts.

    This app primarily trains a scikit-learn artifact (.pkl). Converters to ONNX/TFLite
    are only possible if we can reconstruct a compatible model graph.

    For the current project, we still attempt to CREATE target artifacts by:
    - if sklearn->ONNX conversion is available (skl2onnx), export an ONNX graph
    - if ONNX->TFLite is available (tf + onnx-tf), export a TFLite model

    If tooling is missing or conversion fails, we fall back to an Edge Impulse zip,
    but we do not silently skip ONNX/TFLite when conversion dependencies exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "deployment_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    suffix = model_path.suffix.lower()

    # If a converted artifact already exists, just copy it.
    if "TensorFlow Lite" in target_format and suffix == ".tflite":
        target = output_dir / model_path.name
        shutil.copy2(model_path, target)
        return target
    if "ONNX" in target_format and suffix == ".onnx":
        target = output_dir / model_path.name
        shutil.copy2(model_path, target)
        return target
    if "Keras H5" in target_format and suffix in {".h5", ".keras"}:
        target = output_dir / model_path.name
        shutil.copy2(model_path, target)
        return target

    # Attempt conversions from the sklearn artifact.
    # We intentionally try these only when user requested ONNX/TFLite/Keras.
    if model_path.suffix.lower() == ".pkl":
        estimator = load_artifact(model_path).get("estimator")
        model_info = load_artifact(model_path).get("metadata", {})
        feature_count = model_info.get("feature_count")

        # 1) Create ONNX when requested.
        if "ONNX" in target_format:
            try:
                import numpy as _np
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                if feature_count is None:
                    # best-effort fallback: attempt to infer from metadata or estimator
                    # sklearn estimators typically have n_features_in_.
                    feature_count = int(getattr(estimator, "n_features_in_", 0)) or 1

                initial_type = [("input", FloatTensorType([None, int(feature_count)]))]
                onnx_model = convert_sklearn(estimator, initial_types=initial_type)
                onnx_path = output_dir / f"{model_path.stem}.onnx"
                import onnx as _onnx
                _onnx.save_model(onnx_model, str(onnx_path))

                metadata["conversion"] = {
                    "from": "sklearn",
                    "to": "onnx",
                    "tooling": "skl2onnx",
                    "ok": True,
                }
                with metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

                # Package metadata alongside the exported file.
                return onnx_path
            except Exception as e:
                metadata["conversion"] = {
                    "from": "sklearn",
                    "to": "onnx",
                    "tooling": "skl2onnx",
                    "ok": False,
                    "error": str(e),
                }
                with metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

        # 2) Create TFLite when requested.
        if "TensorFlow Lite" in target_format:
            # First ensure ONNX exists; then attempt ONNX->TF->TFLite.
            try:
                import numpy as _np
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                if feature_count is None:
                    feature_count = int(getattr(estimator, "n_features_in_", 0)) or 1

                initial_type = [("input", FloatTensorType([None, int(feature_count)]))]
                onnx_model = convert_sklearn(estimator, initial_types=initial_type)

                import onnx as _onnx
                tmp_onnx = output_dir / f"{model_path.stem}__tmp.onnx"
                _onnx.save_model(onnx_model, str(tmp_onnx))

                # Try ONNX -> TensorFlow -> SavedModel -> TFLite.
                # NOTE: for the current environment, onnx-tf can fail due to
                # TensorFlow/Keras version mismatches.
                # If conversion fails, we must NOT report success.
                from onnx_tf.backend import prepare as _onnx_prepare

                # Create TF graph representation.
                tf_rep = _onnx_prepare(onnx_model, strict=False)

                saved_dir = output_dir / f"{model_path.stem}__tf_savedmodel"

                if saved_dir.exists():
                    shutil.rmtree(saved_dir)
                saved_dir.mkdir(parents=True, exist_ok=True)

                # Some onnx-tf versions write into <saved_dir>/saved_model.pb
                # Ensure we export into the created directory.
                tf_rep.export_graph(str(saved_dir))

                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
                tflite_model = converter.convert()

                tflite_path = output_dir / f"{model_path.stem}.tflite"
                with open(tflite_path, "wb") as f:
                    f.write(tflite_model)

                if not tflite_path.exists() or tflite_path.stat().st_size == 0:
                    raise RuntimeError(f"TFLite conversion reported success but file was not created: {tflite_path}")

                metadata["conversion"] = {
                    "from": "sklearn",
                    "to": "tflite",
                    "tooling": "skl2onnx + onnx-tf",
                    "ok": True,
                    "tflite_path": str(tflite_path),
                    "tflite_size_bytes": int(tflite_path.stat().st_size),
                }
                with metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

                # Clean tmp onnx if possible
                try:
                    tmp_onnx.unlink(missing_ok=True)
                except Exception:
                    pass

                return tflite_path
            except Exception as e:
                metadata["conversion"] = {
                    "from": "sklearn",
                    "to": "tflite",
                    "tooling": "skl2onnx + onnx-tf",
                    "ok": False,
                    "error": str(e),
                }
                with metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)

    # Default: honest Edge Impulse deployable package containing the source model (.pkl).
    package_path = output_dir / "iotml_deployment_package.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.write(model_path, arcname=model_path.name)
        package.write(metadata_path, arcname=metadata_path.name)
    return package_path



def main() -> None:
    model_path = Path(optional_path("Trained model artifact", DEFAULT_MODEL_PATH))
    print("Available deployment targets:")
    target_format = read_choice("Select target format (number): ", FORMATS)
    output_dir = Path(optional_path("Output deployment directory", "deployment_package"))

    artifact_metadata = {}
    if model_path.suffix.lower() == ".pkl":
        artifact_metadata = load_artifact(model_path).get("metadata", {})
    elif not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metadata = {
        "target_format": target_format,
        "source_model": str(model_path),
        "source_suffix": model_path.suffix,
        "model_metadata": artifact_metadata,
        "note": "Converter-specific tooling is required for native binary conversion.",
    }
    output_path = copy_or_package(model_path, output_dir, target_format, metadata)
    print(f"Deployment output created at {output_path}")


if __name__ == "__main__":
    main()

