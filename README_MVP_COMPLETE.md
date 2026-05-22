# No-Code TinyML Studio for Hardware-Aware IoT Deployment (MVP)

> **Purpose of this document**: This README is written at an IEEE-friendly level of detail. It explains the architecture, data flow, constraints/compatibility system, APIs, and MVP workflow design, enabling conversion into a technical paper draft.

---

## 1. Title and Abstract

### Title
**A Modular No-Code TinyML Studio with Hardware-Aware Constraints and Manual Optimization Workflow for Edge IoT Deployment**

### Abstract
This project provides a full-stack no-code platform to help users build, train, and export TinyML models targeted to constrained IoT devices. Unlike fully automated pipelines that tightly couple all stages, the MVP implements a **modular, user-driven workflow** supported by a **hardware-aware constraint and compatibility subsystem**. The platform models device capability limits (memory, flash, latency, and input resolution), validates configurations, estimates memory/latency, and exposes runtime/framework compatibility—while still allowing users to apply optimizations step-by-step.

The MVP includes:
- A **device capability database**.
- A session-level **active constraint engine**.
- Reusable **validation, compatibility, and estimation APIs**.
- A React/Vanilla JavaScript frontend that fetches constraints, displays recommendation cards, and applies **capped input UI** to prevent invalid settings.
- A backend that avoids deeply coupled automatic enforcement, reducing architectural complexity and improving maintainability.

---

## 2. Keywords
TinyML, No-Code, IoT, Edge AI, Hardware-Aware, Constraints, Memory Estimation, Latency Estimation, Runtime Compatibility, FastAPI, React.

---

## 3. System Overview

### 3.1 High-Level Architecture
The system follows a classic full-stack pattern:
- **Frontend (UI)**: A web UI (React/JS + Tailwind styling in the repository) that guides users through steps.
- **Backend (API)**: A FastAPI server that manages datasets/projects, constraint state, and estimation/compatibility logic.
- **Artifacts**: Dataset files stored under `datasets/`, model checkpoints under `models/`, and exported files under `exports/`.

### 3.2 MVP Design Philosophy: Modular Manual Workflow
Fully automatic orchestration can become brittle because preprocessing, training architecture, quantization, and runtime/export stages interact in complex ways. The MVP instead:
- Provides hardware-aware logic through backend APIs.
- Presents interactive, stepwise controls and capped inputs in the frontend.
- Lets users choose optimizations manually while the system validates and recommends.

This approach improves:
- **Scalability**: New model families and devices can be added as independent modules.
- **Debuggability**: Users can isolate where an invalid configuration was chosen.
- **Maintainability**: Constraint logic is centralized but does not rewrite the entire pipeline.

---

## 4. Core Concepts

### 4.1 Device Capability Database
A capability database defines per-device characteristics and constraints. In the current MVP, it exists in `web_app.py` (and related modules) and includes:
- **Max RAM** (MB)
- **Max Flash** (MB)
- **Max latency target** (ms)
- **Preferred runtimes/frameworks**
- **Edge-only / micro-controller orientation**
- **Input resolution cap** (derived or configured)
- Additional metadata such as accelerators and expected preprocessing stack

This database is used to derive session constraints.

### 4.2 Active Constraint Engine (Session State)
The backend maintains an **ACTIVE_CONSTRAINTS** object representing the currently selected device profile and user budget. From these inputs, the system derives hard limits used for validation and recommendations.

Derived constraints include (examples):
- `ram_mb`, `flash_mb`
- `max_latency_ms`
- `input_resolution_cap = (width, height)`
- `edge_only` and runtime-related hints

The constraint engine enables:
- **Compatibility validation**
- **Runtime filtering**
- **Memory/latency estimation**
- **Frontend capping and disabled options** (via capped UI behavior)

### 4.3 Manual Optimization Workflow
In the MVP, the user controls:
- preprocessing parameters (e.g., input resolution)
- DSP/DSP-like settings (where applicable)
- model type and architecture selection
- quantization settings (e.g., INT8 options)
- runtime selection and export target
- optimization level

Backend logic supports the user by:
- returning warnings and recommendations
- preventing unsupported choices via validation APIs and capped UI

---

## 5. MVP Workflow (End-to-End)

### 5.1 Step 0: Device Selection and Budgeting
1. User selects an IoT target device from UI.
2. User sets an application budget and/or relevant constraints.
3. The frontend calls constraint APIs to store the **active constraints**.
4. Frontend displays **capability indicators** and a derived **memory/latency impact** panel.

**IEEE-relevant claim**: By deriving caps at the session level, the platform ensures consistent guidance across subsequent steps.

### 5.2 Step 1: Data Upload and Labeling (MVP-compatible)
- Users upload datasets and optionally provide labels.
- Label metadata is stored as `labels.json`.

The MVP focuses on the hardware-aware subsystem integration rather than fully completing training for all modalities.

### 5.3 Step 2: EON Tuner / Model Search (Manual Control with Recommendations)
- The backend can suggest recommended model types based on the device.
- The user manually selects a model type/parameters.

**Constraint-aware behavior**:
- UI clamps input resolution related values.
- Runtime/framework recommendations are fetched from backend.

### 5.4 Step 3: Memory and Latency Estimation (Pre-Training / Pre-Export)
The user manually configures model + preprocessing settings.
The frontend requests:
- estimated memory footprint
- estimated latency

If `fits_ram` or `fits_latency` is false, frontend warns and user can adjust.

### 5.5 Step 4: Runtime Compatibility and Export Recommendations
- Backend provides compatible runtimes.
- Backend suggests model categories consistent with the device.
- User chooses quantization and export format.

### 5.6 Step 5: Deploy Output (Model Artifacts)
- Backend supports exporting and packaging outputs.

In MVP, the generation logic may include placeholder code paths for certain formats, but the constraint and validation layer is intended to be authoritative.

---

## 6. Backend API Specification (IEEE-Level)

> Note: Endpoint names reflect the current MVP implementation. All constraints and recommendations are served through the `/api/*` namespace.

### 6.1 `POST /api/set-constraints`
**Purpose**: Sets the active device constraints and budget for the current session.

**Request body (example)**:
```json
{
  "target_device": "Arduino Nano 33 BLE Sense",
  "application_budget": {
    "ram_mb": 256,
    "flash_mb": 1024,
    "max_latency_ms": 100
  },
  "user_prefs": {
    "edge_only": true
  }
}
```

**Behavior**:
- Validates input keys.
- Looks up device profile from capability database.
- Computes derived constraints (`ACTIVE_CONSTRAINTS['derived']`).
- Clamps budgets to device maxima.

**Response**:
```json
{
  "ok": true,
  "device": {"name": "..."},
  "constraints": {"ram_mb": ..., "flash_mb": ..., "max_latency_ms": ...}
}
```

### 6.2 `GET /api/constraints`
**Purpose**: Fetches current derived constraints.

**Response** (shape):
```json
{
  "ok": true,
  "constraints": { ... derived constraints ... },
  "device": { ... device profile ... }
}
```

Frontend uses it to:
- update capability indicators
- clamp input UI

### 6.3 `GET /api/compatible-runtimes`
**Purpose**: Returns a list of runtimes compatible with the active constraints.

**Response**:
```json
{
  "ok": true,
  "runtimes": ["tflite_micro", "cmsis_nn", "esp_dl"],
  "constraints": { ... }
}
```

Rules (MVP approximation):
- `preferred_runtimes` from the device profile are returned.
- Edge-only devices may be filtered to micro-friendly runtimes.

### 6.4 `GET /api/recommended-models`
**Purpose**: Returns recommended model categories/types for the active device.

**Response**:
```json
{
  "ok": true,
  "model_types": ["tinyml", "mlp"],
  "constraints": { ... }
}
```

### 6.5 `POST /api/estimate-memory`
**Purpose**: Estimate memory footprint (RAM proxy) based on model type and hyperparameters.

**Request body (example)**:
```json
{
  "model_type": "cnn",
  "params": {"num_layers": 2, "filters": 16, "kernel_size": 3}
}
```

**Response**:
```json
{
  "ok": true,
  "estimated_ram_mb": 12.3,
  "fits_ram": true,
  "ram_cap_mb": 32
}
```

**Validation**:
- estimates computed heuristically
- `fits_ram` compares estimate vs derived cap

### 6.6 `POST /api/estimate-latency`
**Purpose**: Estimate latency based on input resolution and model type.

**Request body (example)**:
```json
{
  "model_type": "cnn",
  "imgsz": 64
}
```

**Response**:
```json
{
  "ok": true,
  "estimated_latency_ms": 58.0,
  "fits_latency": false,
  "latency_cap_ms": 40
}
```

**Note**: MVP uses heuristic estimation. Extending the estimator later to incorporate operator-level profiling is recommended.

---

## 7. Frontend Implementation and Hardware-Aware UX

### 7.1 Constraints Panel
The frontend fetches `/api/constraints` and updates the capability/impact UI panel.

### 7.2 Capped Sliders and Clamping
The MVP clamps numeric controls to derived maxima to prevent invalid configurations:
- RAM input max clamped to `derived.max_ram_mb`
- Flash input max clamped to `derived.max_flash_mb`
- YOLO-like `imgsz` fields clamped to `derived.input_resolution_cap`

### 7.3 Manual Control with Recommendation Cards (Planned/Supported)
The backend exposes recommended model types and compatible runtimes. The frontend can render:
- recommendation cards
- runtime compatibility badges
- warning banners when `fits_ram` or `fits_latency` is false

---

## 8. Memory/Latency Modeling (MVP Heuristics)

### 8.1 Memory Estimate Strategy
The MVP uses an approximate formula for weight/parameter memory:
- for CNN-like structures, weight counts are derived from layer counts and filter sizes
- memory is normalized into MB and compared against device RAM cap

### 8.2 Latency Estimate Strategy
Latency uses a heuristic scaling law:
- latency grows with input resolution
- latency is amplified by CNN factor vs MLP/tinyml baseline
- compared against derived max latency cap

---

## 9. Dataset and Artifact Layout

The project follows a standard filesystem-based storage convention:

### 9.1 Dataset Directory
```
/datasets/<project_name>/
  raw/
    <uploaded files>
  labels.json
  device.json
```

### 9.2 Model Directory
```
/models/
  <project>.pt
  <project>_log.json
  <project>_status.json
```

### 9.3 Export Directory
```
/exports/
  <project>.tflite
  <project>.cpp
```

---

## 10. Reproducibility and Development Notes

### 10.1 Building and Running
This repository includes:
- backend: FastAPI
- frontend: React/Vite/CRA-like structure (as present in repository)

To run locally, execute:
- backend server startup command
- frontend dev server

(Use the repository’s documented run steps if present in the root `README.md`.)

### 10.2 Dependencies
Backend dependencies include:
- FastAPI, Uvicorn
- Torch, TensorFlow (for conversion paths)
- PIL, NumPy

> Some parts of the ML and TFLite export logic are MVP-level and may require environment setup.

---

## 11. Evaluation Plan (For Paper Draft)

### 11.1 Quantitative Metrics
For each selected device/model configuration:
- classification accuracy (or other task metrics)
- measured latency on target hardware (future work)
- measured memory usage (future work)
- TFLite/firmware artifact size

### 11.2 Constraint Violation Rate
Compute:
- fraction of user-selected configurations that violate RAM/latency caps
- fraction of violations prevented by UI clamping

### 11.3 Usability/Workflow Metrics
User study metrics:
- number of steps to reach a working deployment
- time-to-first successful export
- number of user corrections after warnings

---

## 12. Limitations (MVP)

- Memory/latency estimation is heuristic.
- Some export/deploy generation logic may be placeholder-level.
- Constraint enforcement is modular/manual; the system does not automatically rewrite all pipeline stages.

These limitations are acceptable for MVP but should be addressed in future versions.

---

## 13. Future Work

### 13.1 Replace Heuristics with Profiling
- incorporate operator-level profiling
- use quantization calibration estimates
- use hardware-specific performance models

### 13.2 Expand Runtime Filtering and Quantization Validation
- validate quantization compatibility with chosen runtime
- include operator support tables per runtime

### 13.3 Extend Multimodal Support
- add audio/tabular dataset pipelines and corresponding estimators

### 13.4 Improve Frontend Interactivity
- full dynamic disabling/hiding of unsupported controls
- recommendation cards based on estimation results

---

## 14. Conclusion

The MVP implements the key ingredients of a professional TinyML studio:
- device capability modeling
- active constraint state
- compatibility + estimation APIs
- hardware-aware frontend UX via capped controls
- a modular manual workflow that keeps the architecture maintainable

This forms a strong foundation for evolving into a more automated system later, while preserving debuggability and extendability.

