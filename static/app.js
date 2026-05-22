/*
  NOTE:
  This file is JS and should NOT be compiled with python.
  Kept unchanged logically; formatting-only safety.
*/

const titleMap = {
  dashboard: "Dashboard",
  devices: "Devices",
  data: "Data acquisition",
  experiments: "Experiments",
  eon: "EON Tuner",
  impulse: "Impulse design",
  "create-impulse": "Create impulse",
  image: "Image",
  "object-detection": "Object detection",
  "advanced-live": "Advanced live testing",
  retrain: "Retrain model",
  live: "Live classification",
  testing: "Model testing",
  post: "Post-processing",
  deploy: "Deployment",
};


const output = document.querySelector("#result-output");
const runState = document.querySelector("#run-state");
const terminalOutput = document.querySelector("#terminal-output");
const terminalState = document.querySelector("#terminal-state");

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function readJsonResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  if (contentType.includes("application/json")) {
    return JSON.parse(text || "{}");
  }
  return {
    ok: false,
    error: `Server returned ${response.status} ${response.statusText || ""} instead of JSON.`,
    details: text.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, 1200),
  };
}

function showPage(pageId) {
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("visible", page.id === pageId);
  });
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === pageId);
  });
  document.querySelector("#page-title").textContent = titleMap[pageId] || "IoT ML Studio";
}

async function refreshStatus() {
  runState.textContent = "Refreshing";
  const response = await fetch("/api/status");
  const data = await response.json();
  renderArtifacts(data.artifacts);
  output.textContent = pretty(data);
  runState.textContent = "Ready";
}

function renderArtifacts(artifacts) {
  const container = document.querySelector("#artifact-status");
  if (!container) return;
  container.innerHTML = "";
  Object.entries(artifacts).forEach(([name, info]) => {
    const item = document.createElement("div");
    item.className = "status-pill";
    item.title = info.path;
    item.innerHTML = `<span>${name}</span><span class="${info.exists ? "ok" : "missing"}">${info.exists ? "Ready" : "Missing"}</span>`;
    container.appendChild(item);
  });
}

function renderTerminal(job) {
  if (!terminalOutput || !terminalState) return;
  terminalState.textContent = job.status || "Running";
  terminalOutput.textContent = (job.logs && job.logs.length ? job.logs.join("\n") : "Waiting for logs...");
  terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function submitAction(form) {
  const action = form.dataset.action;
  const formData = new FormData(form);
  runState.textContent = "Running";
  output.textContent = "Job starting...";
  if (terminalOutput) terminalOutput.textContent = "Starting job...";
  if (terminalState) terminalState.textContent = "Starting";

  const startResponse = await fetch(`/api/run-async/${action}`, {
    method: "POST",
    body: formData,
  });
  const startData = await startResponse.json();
  if (!startData.ok) {
    output.textContent = pretty(startData);
    runState.textContent = "Error";
    if (terminalState) terminalState.textContent = "Error";
    return;
  }

  const jobId = startData.job_id;
  output.textContent = pretty({ ok: true, message: "Job started.", job_id: jobId });
  if (startData.job) renderTerminal(startData.job);

  let data = startData;
  while (true) {
    await delay(1000);
    const response = await fetch(`/api/job/${jobId}`);
    data = await response.json();
    if (!data.ok) {
      output.textContent = pretty(data);
      runState.textContent = "Error";
      if (terminalState) terminalState.textContent = "Error";
      return;
    }

    const job = data.job;
    renderTerminal(job);
    if (job.artifacts) {
      renderArtifacts(job.artifacts);
    }
    if (job.status === "complete" || job.status === "error") {
      data = job.result || { ok: false, error: job.error, artifacts: job.artifacts };
      break;
    }
  }

  if (data.artifacts) {
    renderArtifacts(data.artifacts);
  }
  // Make sure deployment output is visible even when a zip path is returned.
  if (data.ok && data.result && data.result.output_path) {
    output.textContent = pretty({
      ...data,
      result: {
        ...data.result,
        note:
          "If you selected ONNX/TFLite/Keras but conversion tooling isn't available, the backend still creates the Edge-Impulse deployable zip (deployment_metadata.json is included).",
      },
    });
  } else {
    output.textContent = pretty(data);
  }
  runState.textContent = data.ok ? "Complete" : "Error";
  if (terminalState) terminalState.textContent = data.ok ? "Complete" : "Error";

}

let capturedDetectionBlob = null;

function renderDetectionResult(data) {
  const preview = document.querySelector("#detection-result-image");
  const empty = document.querySelector("#detection-empty-state");
  const list = document.querySelector("#detection-list");
  const count = document.querySelector("#detection-count");
  const result = data.result || {};

  if (preview && result.annotated_url) {
    preview.src = `${result.annotated_url}?t=${Date.now()}`;
    preview.hidden = false;
  }
  if (empty) empty.hidden = Boolean(result.annotated_url);
  if (count) count.textContent = `${result.count || 0} detections`;
  if (list) {
    list.innerHTML = "";
    const detections = result.detections || [];
    if (!detections.length) {
      const item = document.createElement("li");
      item.textContent = "No detections above the selected confidence.";
      list.appendChild(item);
    } else {
      detections.forEach((det) => {
        const item = document.createElement("li");
        const box = det.box || {};
        item.innerHTML = `<strong>${det.label}</strong><span>${Number(det.confidence || 0).toFixed(3)} confidence</span><small>x:${Math.round(box.x1 || 0)} y:${Math.round(box.y1 || 0)} w:${Math.round((box.x2 || 0) - (box.x1 || 0))} h:${Math.round((box.y2 || 0) - (box.y1 || 0))}</small>`;
        list.appendChild(item);
      });
    }
  }
}

async function submitDetection(form) {
  const formData = new FormData(form);
  const imageInput = form.querySelector('input[name="image"]');
  if ((!imageInput || imageInput.files.length === 0) && capturedDetectionBlob) {
    formData.set("image", capturedDetectionBlob, "camera-frame.jpg");
  }

  runState.textContent = "Running";
  output.textContent = "Running advanced detector...";
  if (terminalOutput) terminalOutput.textContent = "Running advanced detector...";
  if (terminalState) terminalState.textContent = "Running";

  try {
    const response = await fetch("/api/detection/live", {
      method: "POST",
      body: formData,
    });
    const data = await readJsonResponse(response);
    output.textContent = pretty(data);
    runState.textContent = data.ok ? "Complete" : "Error";
    if (terminalState) terminalState.textContent = data.ok ? "Complete" : "Error";
    if (terminalOutput) {
      terminalOutput.textContent = data.ok ? `Advanced detection complete: ${data.result.count} detections.` : data.error;
    }
    if (data.ok) renderDetectionResult(data);
  } catch (error) {
    runState.textContent = "Error";
    output.textContent = error?.message || String(error);
    if (terminalState) terminalState.textContent = "Error";
    if (terminalOutput) terminalOutput.textContent = error?.message || String(error);
  }
}

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => showPage(item.dataset.page));
});

// ---- Hardware-aware UI wiring ----
async function refreshConstraintsUI() {
  try {
    const res = await fetch("/api/constraints");
    const data = await res.json();
    const derived = data?.constraints || {};

    // Device capability indicators / warnings
    const budgetEl = document.querySelector("#budget-estimates");
    if (budgetEl) {
      const ramMb = derived.ram_mb ?? "—";
      const flashMb = derived.flash_mb ?? "—";
      const latencyMs = derived.max_latency_ms ?? "—";
      const inputCap = derived.input_resolution_cap ? `${derived.input_resolution_cap[0]}x${derived.input_resolution_cap[1]}` : "—";
      const runtimes = (derived.supported_runtimes || []).join(", ");
      budgetEl.innerHTML =
        `• RAM usage estimate: ${ramMb} MB\n<br/>` +
        `• Flash usage estimate: ${flashMb} MB\n<br/>` +
        `• Latency target: ${latencyMs} ms\n<br/>` +
        `• Input resolution cap: ${inputCap}\n<br/>` +
        `• Supported runtimes: ${runtimes || "—"}`;
    }

    // Clamp/lock numeric inputs that exist on the page.
    const ramInput = document.querySelector("#ram_mb");
    const flashInput = document.querySelector("#flash_mb");
    const latencyInput = document.querySelector("#max_latency_ms");
    if (ramInput && derived.max_ram_mb != null) {
      ramInput.max = Number(derived.max_ram_mb);
      if (Number(ramInput.value) > Number(derived.max_ram_mb)) ramInput.value = String(derived.max_ram_mb);
    }
    if (flashInput && derived.max_flash_mb != null) {
      flashInput.max = Number(derived.max_flash_mb);
      if (Number(flashInput.value) > Number(derived.max_flash_mb)) flashInput.value = String(derived.max_flash_mb);
    }
    if (latencyInput && derived.max_latency_ms != null) {
      latencyInput.max = Number(derived.max_latency_ms);
    }

    // Clamp YOLO/imgsz fields if present
    const imgszInputs = document.querySelectorAll("input[name='imgsz'], input#imgsz");
    if (imgszInputs && derived.input_resolution_cap) {
      const capSide = Math.min(Number(derived.input_resolution_cap[0]), Number(derived.input_resolution_cap[1]));
      imgszInputs.forEach((el) => {
        if (!el) return;
        el.max = capSide;
        if (Number(el.value) > capSide) el.value = String(capSide);
      });
    }

    // Update EON tuner device selector options to reflect backend supported devices already handled by dropdown.
  } catch (e) {
    // ignore
  }
}

function wireConstraintApplyForms() {
  const targetForm = document.querySelector("#target-device-form");
  const budgetForm = document.querySelector("#app-budget-form");

  [targetForm, budgetForm].forEach((form) => {
    if (!form) return;
    form.addEventListener("submit", async (ev) => {
      // Let existing submitAction run, but then refresh constraints UI.
      setTimeout(() => refreshConstraintsUI(), 50);
    });
  });
}

wireConstraintApplyForms();
refreshConstraintsUI();


document.querySelectorAll(".action-form").forEach((form) => {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (form.id === "advanced-detection-form") {
      submitDetection(form);
      return;
    }
    submitAction(form).then(() => refreshConstraintsUI());
  });
});


const detectionImageInput = document.querySelector("#detection-image");
const detectionSourcePreview = document.querySelector("#detection-source-image");
if (detectionImageInput && detectionSourcePreview) {
  detectionImageInput.addEventListener("change", () => {
    capturedDetectionBlob = null;
    const file = detectionImageInput.files && detectionImageInput.files[0];
    if (!file) return;
    detectionSourcePreview.src = URL.createObjectURL(file);
    detectionSourcePreview.hidden = false;
  });
}

const cameraButton = document.querySelector("#start-detector-camera");
const captureButton = document.querySelector("#capture-detector-frame");
const cameraVideo = document.querySelector("#detector-camera");
const cameraCanvas = document.querySelector("#detector-canvas");
if (cameraButton && captureButton && cameraVideo && cameraCanvas) {
  cameraButton.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      cameraVideo.srcObject = stream;
      cameraVideo.hidden = false;
      captureButton.disabled = false;
    } catch (error) {
      output.textContent = error?.message || String(error);
      runState.textContent = "Error";
    }
  });

  captureButton.addEventListener("click", () => {
    cameraCanvas.width = cameraVideo.videoWidth || 640;
    cameraCanvas.height = cameraVideo.videoHeight || 480;
    const context = cameraCanvas.getContext("2d");
    context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    cameraCanvas.toBlob((blob) => {
      capturedDetectionBlob = blob;
      if (detectionImageInput) detectionImageInput.value = "";
      if (detectionSourcePreview) {
        detectionSourcePreview.src = cameraCanvas.toDataURL("image/jpeg", 0.92);
        detectionSourcePreview.hidden = false;
      }
    }, "image/jpeg", 0.92);
  });
}

document.querySelector("#refresh-status").addEventListener("click", refreshStatus);

// Kill switch: cleanup artifacts immediately
const killButton = document.querySelector("#kill-switch");
if (killButton) {
  killButton.addEventListener("click", async () => {
    runState.textContent = "Cleaning...";
    output.textContent = "Working...";
    try {
      const response = await fetch("/api/kill", { method: "POST" });
      const data = await response.json();
      if (data.artifacts) {
        renderArtifacts(data.artifacts);
      }
      output.textContent = pretty(data);
      runState.textContent = data.ok ? "Complete" : "Error";
      if (terminalOutput) terminalOutput.textContent = data.message || "Cleanup completed.";
      if (terminalState) terminalState.textContent = data.ok ? "Complete" : "Error";
    } catch (e) {
      runState.textContent = "Error";
      output.textContent = e?.message || String(e);
      if (terminalState) terminalState.textContent = "Error";
    }
  });
}

refreshStatus().catch((error) => {
  runState.textContent = "Error";
  output.textContent = error.message;
});
