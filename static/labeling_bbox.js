/*
 * COCO bounding-box labeling UI (canvas overlay)
 *
 * This is intentionally framework-free so it can be used from Flask templates.
 */

(function () {
  function $(id) {
    return document.getElementById(id);
  }

  const state = {
    displayedImg: null,

    // auto-label proposals
    proposals: [], // {id,label,confidence,bbox_px:[x,y,w,h]}
    hasProposals: false,
    yoloModel: 'yolov8n.pt',

    split: null,
    project: null,
    files: [],
    index: 0,
    imageListIndexByKey: {},
    activeImageKey: null,
    coco: null,
    annotations: [],
    classes: [],
    nextAnnotationId: 1,
    selectedAnnotationId: null,
    selectedHandle: null,
    mode: "draw", // draw | edit

    // view transform
    zoom: 1,
    panX: 0,
    panY: 0,

    // interaction
    dragging: false,
    resizing: false,
    drawStart: null,
    lastMouse: null,
  };

  const CANVAS_MIN_SIZE = 20;
  const HANDLE_SIZE = 8;
  const COLORS = [
    "#22c55e",
    "#3b82f6",
    "#f97316",
    "#ef4444",
    "#a855f7",
    "#14b8a6",
    "#eab308",
  ];

  function classColor(className) {
    const idx = Math.max(0, state.classes.indexOf(className));
    return COLORS[idx % COLORS.length];
  }

  function loadQueryParams() {
    const params = new URLSearchParams(window.location.search);
    const project = params.get("project") || "default";
    const split = (params.get("split") || "train").toLowerCase();
    return { project, split };
  }

  async function apiGet(path) {
    const res = await fetch(path);
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || `GET ${path} failed`);
    }
    return data;
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || `POST ${path} failed`);
    }
    return data;
  }

  function imageKeyForFile(relPath) {
    // COCO image file_name in our export is relative file_name under images/
    return relPath;
  }

function getImageSrc() {
    // Images are served from unified dataset folders on the backend.
    // In this repo, the Flask route is:
    //   /datasets/<project>/images/<path>
    // where <path> must be relative (usually basename from COCO file_name).
    const rel = state.activeImageKey;
    return `/datasets/${encodeURIComponent(state.project)}/images/${encodeURIComponent(rel)}`;
  }



  function ensureCanvasSizeToImage() {
    const overlay = $("canvas");
    const wrap = overlay?.parentElement;
    if (wrap) {
      // Force overlay wrapper to match image container dimensions.
      const img = $("image");
      if (img && img.getBoundingClientRect) {
        // Canvas and image must always share exact rendered size.
        const rect = img.getBoundingClientRect();
        overlay.style.width = `${rect.width}px`;
        overlay.style.height = `${rect.height}px`;
        overlay.style.left = `${img.offsetLeft}px`;
      }
    }

    const img = $("image");
    const canvas = $("canvas");
    if (!img.complete) return;

    // Always sync canvas size to the actual rendered size of the image container.
    // The canvas is absolutely positioned with inset:0 inside .canvas-wrap,
    // so measuring the canvas element itself gives the real drawable area.
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));

    // Compute displayed-image area within the canvas when using object-fit: contain.
    const imgRect = img.getBoundingClientRect();

    const boxW = imgRect.width;
    const boxH = imgRect.height;

    const imgW0 = img.naturalWidth || 1;
    const imgH0 = img.naturalHeight || 1;

    const imgAspect = imgW0 / imgH0;
    const boxAspect = boxW / boxH;

    let dispW, dispH;
    if (imgAspect > boxAspect) {
      dispW = boxW;
      dispH = boxW / imgAspect;
    } else {
      dispH = boxH;
      dispW = boxH * imgAspect;
    }

    // offset within canvas (since canvas has same size as img container)
    state.displayedImg = {
      offsetX: (rect.width - dispW) / 2,
      offsetY: (rect.height - dispH) / 2,
      dispW,
      dispH,
      imgW: imgW0,
      imgH: imgH0,
    };
  }



  function screenToImageCoords(clientX, clientY) {
    const canvas = $("canvas");
    const rect = canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;

    // Apply inverse pan/zoom in canvas space.
    const ix = (sx - state.panX) / state.zoom;
    const iy = (sy - state.panY) / state.zoom;

    // Map canvas-space -> displayed image-space (inside contain letterboxing).
    const disp = state.displayedImg;
    const offX = disp?.offsetX || 0;
    const offY = disp?.offsetY || 0;
    const dispW = disp?.dispW || canvas.width;
    const dispH = disp?.dispH || canvas.height;

    const xDisp = ix - offX;
    const yDisp = iy - offY;

    // Clamp to displayed image area only.
    const xClamped = Math.max(0, Math.min(dispW, xDisp));
    const yClamped = Math.max(0, Math.min(dispH, yDisp));

    // Convert displayed image coords -> image pixel coords.
    const imgW = disp?.imgW || 1;
    const imgH = disp?.imgH || 1;
    const xPx = (xClamped / dispW) * imgW;
    const yPx = (yClamped / dispH) * imgH;

    return { x: xPx, y: yPx };
  }


  function normalizeBox(x1, y1, x2, y2) {
    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    const w = Math.abs(x2 - x1);
    const h = Math.abs(y2 - y1);
    return { x: left, y: top, w, h };
  }

  function draw() {
    // draw annotations + proposals (dashed)

    const canvas = $("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw in pan/zoom space
    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);

    for (const ann of state.annotations) {
      const isSelected = ann.id === state.selectedAnnotationId;
      const color = classColor(ann.class_name);
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;

      const { x, y, w, h } = ann.bbox_px;

      // Rectangle
      ctx.strokeRect(x, y, w, h);

      // Label background
      const label = `${ann.class_name}`;
      ctx.font = "14px Inter, Segoe UI, Arial";
      const textW = ctx.measureText(label).width;
      const pad = 4;
      const tx = x;
      const ty = Math.max(14, y - 6);
      ctx.fillRect(tx, ty - 14, textW + pad * 2, 18);
      ctx.fillStyle = "#0b1220";
      ctx.fillText(label, tx + pad, ty - 1);
      ctx.fillStyle = color;

      if (isSelected) {
        drawHandles(ctx, x, y, w, h);
      }
    }

    // Draw current draw box if any
    if (state.drawStart && (state.dragging || state.resizing)) {
      const cur = state.lastMouse;
      const b = normalizeBox(state.drawStart.x, state.drawStart.y, cur.x, cur.y);
      if (b.w >= CANVAS_MIN_SIZE && b.h >= CANVAS_MIN_SIZE) {
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = "#111827";
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.setLineDash([]);
      }
    }

    ctx.restore();
  }

  function drawHandles(ctx, x, y, w, h) {
    const handles = getHandles(x, y, w, h);
    ctx.fillStyle = "#0b1220";
    for (const hnd of handles) {
      ctx.fillRect(hnd.x - HANDLE_SIZE / 2, hnd.y - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
    }
  }

  function getHandles(x, y, w, h) {
    return [
      { name: "nw", x: x, y: y },
      { name: "ne", x: x + w, y: y },
      { name: "sw", x: x, y: y + h },
      { name: "se", x: x + w, y: y + h },
    ];
  }

  function hitTestHandle(imgX, imgY) {
    const active = state.annotations.find((a) => a.id === state.selectedAnnotationId);
    if (!active) return null;
    const { x, y, w, h } = active.bbox_px;
    const handles = getHandles(x, y, w, h);
    for (const hnd of handles) {
      if (Math.abs(imgX - hnd.x) <= HANDLE_SIZE && Math.abs(imgY - hnd.y) <= HANDLE_SIZE) {
        return hnd.name;
      }
    }
    return null;
  }

  function hitTestAnnotation(imgX, imgY) {
    // reverse order - later boxes on top
    for (let i = state.annotations.length - 1; i >= 0; i--) {
      const ann = state.annotations[i];
      const { x, y, w, h } = ann.bbox_px;
      if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) {
        return ann.id;
      }
    }
    return null;
  }

  function updateSidebar() {
    const list = $("annotation-list");
    list.innerHTML = "";

    for (const ann of state.annotations) {
      const li = document.createElement("div");
      li.className = "ann-row";
      li.style.display = "flex";
      li.style.gap = "10px";
      li.style.alignItems = "center";
      li.style.borderTop = "1px solid rgba(0,0,0,0.06)";
      li.style.padding = "8px 0";

      const swatch = document.createElement("div");
      swatch.style.width = "12px";
      swatch.style.height = "12px";
      swatch.style.borderRadius = "3px";
      swatch.style.background = classColor(ann.class_name);

      const meta = document.createElement("div");
      meta.style.flex = "1";
      meta.innerHTML = `<div style="font-weight:700;font-size:12px;">${ann.class_name}</div>
        <div style="color:#667085;font-size:12px;">x:${Math.round(ann.bbox_norm[0]*100)/100} y:${Math.round(ann.bbox_norm[1]*100)/100} w:${Math.round(ann.bbox_norm[2]*100)/100} h:${Math.round(ann.bbox_norm[3]*100)/100}</div>`;

      const btn = document.createElement("button");
      btn.textContent = "Select";
      btn.className = "ghost";
      btn.type = "button";
      btn.onclick = () => {
        state.selectedAnnotationId = ann.id;
        $("class-select").value = ann.class_name;
        updateSidebar();
        draw();
      };

      li.appendChild(swatch);
      li.appendChild(meta);
      li.appendChild(btn);
      list.appendChild(li);
    }
  }

  function pushSelectedClassToAnnotation() {
    const className = $("class-select").value;
    const ann = state.annotations.find((a) => a.id === state.selectedAnnotationId);
    if (!ann) return;
    ann.class_name = className;
    // optimistic redraw + enable save
    updateSidebar();
    draw();
  }

  function computeNormFromPx(box, imgW, imgH) {
    const cx = (box.x + box.w / 2) / imgW;
    const cy = (box.y + box.h / 2) / imgH;
    const nw = box.w / imgW;
    const nh = box.h / imgH;
    return [cx, cy, nw, nh];
  }

  function computePxFromNorm(norm, imgW, imgH) {
    const [cx, cy, nw, nh] = norm;
    const w = nw * imgW;
    const h = nh * imgH;
    const x = (cx * imgW) - w / 2;
    const y = (cy * imgH) - h / 2;
    return { x, y, w, h };
  }

  function normalizeBoxToCocoBboxXYWH_px(box) {
    return { x: box.x, y: box.y, w: box.w, h: box.h };
  }

  async function loadClasses() {
    const data = await apiGet(`/api/annotations/classes?project=${encodeURIComponent(state.project)}`);
    state.classes = (data.classes || []).map(String);
    if (!state.classes.length) state.classes = ["object"];

    const sel = $("class-select");
    sel.innerHTML = "";
    for (const c of state.classes) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      sel.appendChild(opt);
    }

    // Keep current selection when possible
    const cur = $("class-select").value;
    if (cur && state.classes.includes(cur)) {
      sel.value = cur;
    } else {
      sel.value = state.classes[0];
    }
  }


  async function loadImageQueue() {
    const data = await apiGet(`/api/annotations/queue?project=${encodeURIComponent(state.project)}&split=${encodeURIComponent(state.split)}`);
    state.files = data.files || [];
    state.index = Math.min(Math.max(0, data.next_index || 0), Math.max(0, state.files.length - 1));
  }

  async function loadAnnotationsForCurrentImage() {
    if (!state.files.length) return;
    const rel = state.files[state.index];
    state.activeImageKey = imageKeyForFile(rel);
    const data = await apiGet(`/api/annotations/list?project=${encodeURIComponent(state.project)}&split=${encodeURIComponent(state.split)}&image=${encodeURIComponent(rel)}`);
    state.annotations = (data.annotations || []).map((a) => {
      // convert normalized XYWH (0..1) to px using img dims from server response
      const imgW = data.image_width;
      const imgH = data.image_height;
      const px = {
        x: a.bbox_px[0],
        y: a.bbox_px[1],
        w: a.bbox_px[2],
        h: a.bbox_px[3],
      };
      const bbox_norm = a.bbox_norm;
      return {
        id: a.annotation_id,
        class_name: a.class_name,
        bbox_px: px,
        bbox_norm: bbox_norm,
        category_id: a.category_id,
      };
    });

    state.selectedAnnotationId = state.annotations.length ? state.annotations[0].id : null;
    const sel = $("class-select");
    if (state.selectedAnnotationId != null) {
      const ann = state.annotations.find((a) => a.id === state.selectedAnnotationId);
      sel.value = ann.class_name;
    }

    $("fileName").textContent = rel;
    $("progress").textContent = `${state.index + 1} / ${state.files.length}`;
    $("msg").textContent = "";
  }

  function renderImage() {
    const img = $("image");
    img.src = getImageSrc() + "?t=" + Date.now();
  }

  async function refreshCurrent() {
    // Reset view transform when switching images.

    if (!state.files.length) {
      state.activeImageKey = null;
      return;
    }
    const rel = state.files[state.index];
    state.activeImageKey = rel;
    renderImage();
    await loadAnnotationsForCurrentImage();

    // Wait for image load then redraw
    await new Promise((resolve) => {
      const img = $("image");
      if (img.complete) return resolve();
      img.onload = () => resolve();
      img.onerror = () => resolve();
    });

    ensureCanvasSizeToImage();
    state.panX = 0;
    state.panY = 0;
    state.zoom = 1;
    draw();
    updateSidebar();
  }

  function nextImage() {
    if (!state.files.length) return;
    state.index = Math.min(state.index + 1, state.files.length - 1);
    refreshCurrent();
  }

  function prevImage() {
    if (!state.files.length) return;
    state.index = Math.max(state.index - 1, 0);
    refreshCurrent();
  }

  async function saveCurrentImageAnnotations() {
    if (!state.activeImageKey) return;

    // Transform current annotations into normalized coco payload
    const image = state.activeImageKey;
    const payload = {
      project: state.project,
      split: state.split,
      image_filename: image,
      annotations: state.annotations.map((ann) => ({
        annotation_id: ann.id,
        category_name: ann.class_name,
        bbox_norm_xywh: ann.bbox_norm, // [x_center,y_center,w,h] normalized
      })),
    };

    const data = await apiPost(`/api/annotations/save`, payload);
    $("msg").textContent = data.ok ? "Saved COCO annotations." : (data.error || "Save failed");

    // reload annotations to reflect ids/category ids
    await refreshCurrent();
  }

  async function deleteSelectedAnnotation() {
    if (state.selectedAnnotationId == null) return;
    const data = await apiPost(`/api/annotations/delete`, {
      project: state.project,
      split: state.split,
      image_filename: state.activeImageKey,
      annotation_id: state.selectedAnnotationId,
    });
    $("msg").textContent = data.ok ? "Deleted annotation." : (data.error || "Delete failed");
    state.selectedAnnotationId = null;
    await refreshCurrent();
  }

  function canvasMouseDown(e) {
    const img = $("image");
    const canvas = $("canvas");
    if (!canvas) return;

    const pos = screenToImageCoords(e.clientX, e.clientY);
    state.lastMouse = pos;

    // Pan with spacebar? (simplified)
    const isSpace = (window._spaceDown === true);
    if (isSpace) {
      state.dragging = true;
      state.drawStart = { x: pos.x, y: pos.y, panX: state.panX, panY: state.panY };
      return;
    }

    if (e.button !== 0) return;

    const selected = hitTestAnnotation(pos.x, pos.y);
    if (selected != null) {
      state.selectedAnnotationId = selected;
      const ann = state.annotations.find((a) => a.id === selected);
      $("class-select").value = ann.class_name;
      draw();

      // handle resizing
      const handle = hitTestHandle(pos.x, pos.y);
      if (handle) {
        state.resizing = true;
        state.selectedHandle = handle;
        state.drawStart = { x: pos.x, y: pos.y };
        state.lastMouse = pos;
      } else {
        state.dragging = true;
        state.drawStart = { x: pos.x, y: pos.y };
      }
      return;
    }

    // start new draw box
    state.dragging = true;
    state.resizing = false;
    state.drawStart = { x: pos.x, y: pos.y };
    state.lastMouse = pos;
    state.selectedAnnotationId = null;
  }

  function canvasMouseMove(e) {
    const pos = screenToImageCoords(e.clientX, e.clientY);
    state.lastMouse = pos;

    if (state.dragging) {
      // pan if space pressed
      if (state.drawStart && state.drawStart.panX != null) {
        const dx = pos.x - state.drawStart.x;
        const dy = pos.y - state.drawStart.y;
        state.panX = state.drawStart.panX + dx * state.zoom;
        state.panY = state.drawStart.panY + dy * state.zoom;
        draw();
        return;
      }

      const ann = state.annotations.find((a) => a.id === state.selectedAnnotationId);
      if (ann && state.resizing && state.selectedHandle) {
        const b = ann.bbox_px;
        const x0 = b.x;
        const y0 = b.y;
        const x1 = b.x + b.w;
        const y1 = b.y + b.h;
        const nx = pos.x;
        const ny = pos.y;

        // update based on handle
        let newBox = { x: x0, y: y0, w: b.w, h: b.h };
        if (state.selectedHandle === "nw") {
          newBox.x = Math.min(nx, x1 - CANVAS_MIN_SIZE);
          newBox.y = Math.min(ny, y1 - CANVAS_MIN_SIZE);
          newBox.w = x1 - newBox.x;
          newBox.h = y1 - newBox.y;
        } else if (state.selectedHandle === "ne") {
          newBox.y = Math.min(ny, y1 - CANVAS_MIN_SIZE);
          newBox.w = Math.max(CANVAS_MIN_SIZE, nx - x0);
          newBox.h = y1 - newBox.y;
        } else if (state.selectedHandle === "sw") {
          newBox.x = Math.min(nx, x1 - CANVAS_MIN_SIZE);
          newBox.w = x1 - newBox.x;
          newBox.h = Math.max(CANVAS_MIN_SIZE, ny - y0);
        } else if (state.selectedHandle === "se") {
          newBox.w = Math.max(CANVAS_MIN_SIZE, nx - x0);
          newBox.h = Math.max(CANVAS_MIN_SIZE, ny - y0);
        }

        ann.bbox_px = newBox;
        // bbox_norm is computed at save time on server for accuracy; keep px.
        draw();
        return;
      }

      if (ann && !state.resizing) {
        // move
        const dx = pos.x - state.drawStart.x;
        const dy = pos.y - state.drawStart.y;
        ann.bbox_px = { ...ann.bbox_px, x: ann.bbox_px.x + dx, y: ann.bbox_px.y + dy };
        state.drawStart = { x: pos.x, y: pos.y };
        draw();
        return;
      }

      // else drawing a new box preview
      draw();
    }
  }

  function canvasMouseUp() {
    if (!state.dragging) return;

    // If drawing a new box (no selected annotation)
    if (state.drawStart && state.selectedAnnotationId == null && !state.resizing) {
      const start = state.drawStart;
      const end = state.lastMouse;
      const box = normalizeBox(start.x, start.y, end.x, end.y);
      if (box.w >= CANVAS_MIN_SIZE && box.h >= CANVAS_MIN_SIZE) {
        const className = $("class-select").value;
        const ann = {
          id: "local_" + Math.random().toString(16).slice(2),
          class_name: className,
          bbox_px: box,
          bbox_norm: [0.5, 0.5, 1, 1],
          category_id: null,
        };
        state.annotations.push(ann);
        state.selectedAnnotationId = ann.id;
        draw();
        updateSidebar();
      }
    }

    state.dragging = false;
    state.resizing = false;
    state.selectedHandle = null;
    state.drawStart = null;
  }

  function installCanvasHandlers() {

    const canvas = $("canvas");
    if (!canvas) return;

    canvas.addEventListener("mousedown", (e) => canvasMouseDown(e));
    window.addEventListener("mousemove", (e) => canvasMouseMove(e));
    window.addEventListener("mouseup", () => canvasMouseUp());

    // wheel zoom
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const delta = Math.sign(e.deltaY);
      const factor = delta > 0 ? 0.9 : 1.1;
      state.zoom = Math.max(0.25, Math.min(6, state.zoom * factor));
      draw();
    }, { passive: false });
  }

  function installKeyboardShortcuts() {
    window._spaceDown = false;
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        window._spaceDown = true;
        e.preventDefault();
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        if ((document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "SELECT")) && e.key !== "Delete") {
          return;
        }
        deleteSelectedAnnotation().catch(() => {});
      }
      if (e.key === "ArrowRight") {
        nextImage();
      }
      if (e.key === "ArrowLeft") {
        prevImage();
      }
      if (e.key === "s" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        saveCurrentImageAnnotations().catch((err) => {
          $("msg").textContent = "Save failed: " + (err?.message || String(err));
        });
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space") {
        window._spaceDown = false;
      }
    });
  }

  async function init() {
    const { project, split } = loadQueryParams();
    state.project = project;
    state.split = split;

    // class selector + keyboard
    await loadClasses();
    await loadImageQueue();

    $("split-select").value = split;

    installKeyboardShortcuts();
    installCanvasHandlers();

    $("class-select").addEventListener("change", () => {
      pushSelectedClassToAnnotation();
    });

    const addBtn = $("class-add-btn");
    const classInput = $("class-input");
    if (addBtn && classInput) {
      addBtn.addEventListener("click", async () => {
        const name = String(classInput.value || "").trim();
        if (!name) return;

        // Optimistically add to UI (backend will create the category on save).
        if (!state.classes.includes(name)) {
          state.classes.push(name);
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          $("class-select").appendChild(opt);
        }
        $("class-select").value = name;
        pushSelectedClassToAnnotation();
        classInput.value = "";
      });
    }


    $("save-btn").addEventListener("click", () => {
      saveCurrentImageAnnotations().catch((err) => {
        $("msg").textContent = "Save failed: " + (err?.message || String(err));
      });
    });

    $("delete-btn").addEventListener("click", () => {
      deleteSelectedAnnotation().catch((err) => {
        $("msg").textContent = "Delete failed: " + (err?.message || String(err));
      });
    });

    await refreshCurrent();

    // If no files
    if (!state.files.length) {
      $("fileName").textContent = "No images found";
      $("msg").textContent = "Upload images into the dataset export images folder before labeling.";
    }
  }

  window.addEventListener("load", init);
})();

