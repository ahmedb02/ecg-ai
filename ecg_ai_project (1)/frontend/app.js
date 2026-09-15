"use strict";

const API_BASE = "";

// ---------- Crop selector (drag rectangle over an image) ----------

class CropSelector {
  constructor(stageEl) {
    this.stageEl = stageEl;
    this.img = null;
    this.rectEl = null;
    this.box = { x0: 0.05, y0: 0.05, x1: 0.95, y1: 0.95 }; // normalized
    this._drag = null;
  }

  async loadFile(file) {
    const url = URL.createObjectURL(file);
    return this._load(url);
  }

  async loadURL(url) {
    return this._load(url);
  }

  _load(src) {
    if (this._resizeHandler) window.removeEventListener("resize", this._resizeHandler);
    this.stageEl.innerHTML = "";
    this.box = { x0: 0.05, y0: 0.05, x1: 0.95, y1: 0.95 };
    const img = document.createElement("img");
    img.src = src;
    img.draggable = false;
    this.img = img;
    this.stageEl.appendChild(img);

    return new Promise((resolve) => {
      img.onload = () => {
        this._buildRect();
        resolve();
      };
    });
  }

  _buildRect() {
    const rect = document.createElement("div");
    rect.className = "crop-rect";
    ["nw", "ne", "sw", "se"].forEach((corner) => {
      const h = document.createElement("div");
      h.className = `crop-handle ${corner}`;
      h.dataset.corner = corner;
      rect.appendChild(h);
    });
    this.stageEl.appendChild(rect);
    this.rectEl = rect;
    // Deferred to the next frame so this measures a laid-out (visible,
    // non-zero-size) stage even if the caller unhides the container in
    // the same tick as loadFile() -- getBoundingClientRect() on a
    // display:none ancestor (or one not yet reflowed) reports 0x0.
    requestAnimationFrame(() => this._syncRectStyle());
    this._resizeHandler = () => this._syncRectStyle();
    window.addEventListener("resize", this._resizeHandler);

    rect.addEventListener("pointerdown", (e) => this._onPointerDown(e, "move"));
    rect.querySelectorAll(".crop-handle").forEach((h) => {
      h.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        this._onPointerDown(e, h.dataset.corner);
      });
    });
    window.addEventListener("pointermove", (e) => this._onPointerMove(e));
    window.addEventListener("pointerup", () => { this._drag = null; });
  }

  _stageRect() {
    return this.stageEl.getBoundingClientRect();
  }

  _syncRectStyle() {
    const stage = this._stageRect();
    const { x0, y0, x1, y1 } = this.box;
    this.rectEl.style.left = `${x0 * stage.width}px`;
    this.rectEl.style.top = `${y0 * stage.height}px`;
    this.rectEl.style.width = `${(x1 - x0) * stage.width}px`;
    this.rectEl.style.height = `${(y1 - y0) * stage.height}px`;
  }

  _onPointerDown(e, mode) {
    e.preventDefault();
    this._drag = {
      mode,
      startX: e.clientX,
      startY: e.clientY,
      startBox: { ...this.box },
    };
  }

  _onPointerMove(e) {
    if (!this._drag) return;
    const stage = this._stageRect();
    const dx = (e.clientX - this._drag.startX) / stage.width;
    const dy = (e.clientY - this._drag.startY) / stage.height;
    const b = { ...this._drag.startBox };
    const { mode } = this._drag;

    if (mode === "move") {
      const w = b.x1 - b.x0, h = b.y1 - b.y0;
      let x0 = clamp(b.x0 + dx, 0, 1 - w);
      let y0 = clamp(b.y0 + dy, 0, 1 - h);
      b.x0 = x0; b.x1 = x0 + w; b.y0 = y0; b.y1 = y0 + h;
    } else {
      if (mode.includes("w")) b.x0 = clamp(b.x0 + dx, 0, b.x1 - 0.03);
      if (mode.includes("e")) b.x1 = clamp(b.x1 + dx, b.x0 + 0.03, 1);
      if (mode.includes("n")) b.y0 = clamp(b.y0 + dy, 0, b.y1 - 0.03);
      if (mode.includes("s")) b.y1 = clamp(b.y1 + dy, b.y0 + 0.03, 1);
    }
    this.box = b;
    this._syncRectStyle();
  }

  getNormalizedBox() {
    const { x0, y0, x1, y1 } = this.box;
    return [x0, y0, x1, y1];
  }

  getFullBoxSelected() {
    const b = this.box;
    return b.x0 < 0.06 && b.y0 < 0.06 && b.x1 > 0.94 && b.y1 > 0.94;
  }
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ---------- Tabs ----------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------- Model status ----------

async function loadHealth() {
  const el = document.getElementById("model-status");
  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    const body = await resp.json();
    const trained = body.model_status.spatial_trained;
    el.textContent = trained ? "Trained model loaded" : "Demo mode (untrained weights)";
    el.className = `status ${trained ? "trained" : "demo"}`;
  } catch (e) {
    el.textContent = "Could not reach backend";
  }
}
loadHealth();

// ---------- Shared rendering helpers ----------

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function renderFindingList(findings) {
  const ul = el("ul", "finding-list");
  if (findings.length === 0) {
    ul.appendChild(el("li", "finding-item", "No findings exceeded the reporting threshold."));
    return ul;
  }
  for (const f of findings) {
    const li = el("li", "finding-item");
    li.appendChild(el("span", null, f.finding));
    li.appendChild(el("span", "finding-prob", `p=${f.probability.toFixed(2)}`));
    ul.appendChild(li);
  }
  return ul;
}

function renderDDx(ddxList) {
  const wrap = el("div", "ddx-list");
  for (const d of ddxList) {
    const card = el("div", "ddx-card");
    const score = el("span", "ddx-score", `score ${d.score.toFixed(2)}`);
    const title = el("div", "ddx-title", d.diagnosis);
    title.appendChild(score);
    card.appendChild(title);
    card.appendChild(el("div", "ddx-rationale", d.rationale));
    const support = d.supporting_findings && d.supporting_findings.length
      ? `Supporting findings: ${d.supporting_findings.join(", ")}`
      : "";
    if (support) card.appendChild(el("div", "ddx-support", support));
    wrap.appendChild(card);
  }
  return wrap;
}

function renderDigitizationInfo(info) {
  const wrap = el("div");
  const methodLabel = {
    grid_autodetect: "Auto-detected from paper grid",
    manual: "Manual calibration",
    fallback_autoscale: "Amplitude auto-scaled (no grid detected)",
  }[info.calibration_method] || info.calibration_method;
  wrap.appendChild(el("div", "hint", `Calibration: ${methodLabel} · ${info.duration_seconds.toFixed(2)}s digitized`));

  if (info.warnings && info.warnings.length) {
    const ul = el("ul", "warning-list");
    info.warnings.forEach((w) => ul.appendChild(el("li", null, w)));
    wrap.appendChild(ul);
  }

  if (info.overlay_image_data_url) {
    const toggle = el("span", "overlay-toggle", "Show extracted-trace overlay (verify digitization)");
    const img = document.createElement("img");
    img.className = "overlay-img";
    img.style.display = "none";
    img.src = info.overlay_image_data_url;
    toggle.addEventListener("click", () => {
      const show = img.style.display === "none";
      img.style.display = show ? "block" : "none";
      toggle.textContent = show
        ? "Hide extracted-trace overlay"
        : "Show extracted-trace overlay (verify digitization)";
    });
    wrap.appendChild(toggle);
    wrap.appendChild(img);
  }
  return wrap;
}

function renderReport(container, report) {
  container.innerHTML = "";

  if (!report.model_status.spatial_trained) {
    const banner = el("div", "error-box", report.model_status.message);
    banner.style.background = "#3a2f10";
    banner.style.borderColor = "#6b551f";
    banner.style.color = "#ffe0a0";
    container.appendChild(banner);
  }

  const urgent = /URGENT FINDING/.test(report.summary);
  const summary = el("div", `summary-card${urgent ? " urgent" : ""}`, report.summary);
  container.appendChild(summary);

  container.appendChild(el("h3", "section-title", "Findings"));
  container.appendChild(renderFindingList(report.findings));

  container.appendChild(el("h3", "section-title", "Differential diagnosis"));
  container.appendChild(renderDDx(report.differential_diagnosis));

  container.appendChild(el("h3", "section-title", "Digitization"));
  container.appendChild(renderDigitizationInfo(report.digitization));

  container.appendChild(el("div", "disclaimer-footer", report.disclaimer));
}

function renderEvolution(container, evo) {
  const wrap = el("div", "panel");
  wrap.appendChild(el("h3", "section-title", "Evolution across studies"));
  wrap.appendChild(el("div", "summary-card", evo.narrative));

  if (evo.most_changed_regions && evo.most_changed_regions.length) {
    const chips = el("div");
    evo.most_changed_regions.forEach((r) => {
      chips.appendChild(el("span", "region-chip", `${r.region.replace(/_/g, " ")} (Δ${r.change_score.toFixed(2)})`));
    });
    wrap.appendChild(chips);
  }

  const cols = el("div", "field-grid");
  const mkCol = (title, findings) => {
    const c = el("div");
    c.appendChild(el("div", "hint", title));
    c.appendChild(renderFindingList(findings));
    return c;
  };
  cols.appendChild(mkCol("New", evo.new_findings));
  cols.appendChild(mkCol("Resolved", evo.resolved_findings));
  cols.appendChild(mkCol("Persistent", evo.persistent_findings));
  wrap.appendChild(cols);

  wrap.appendChild(el("div", "disclaimer-footer", evo.disclaimer));
  container.appendChild(wrap);
}

async function parseErrorResponse(resp) {
  try {
    const body = await resp.json();
    return body.detail || JSON.stringify(body);
  } catch {
    return `HTTP ${resp.status}`;
  }
}

// ---------- Single ECG tab ----------

const singleDropzone = document.getElementById("single-dropzone");
const singleFileInput = document.getElementById("single-file-input");
const singleCropArea = document.getElementById("single-crop-area");
const singleUploadPanel = document.getElementById("single-upload-panel");
const singleCrop = new CropSelector(document.getElementById("single-crop-stage"));

singleDropzone.addEventListener("click", () => singleFileInput.click());
singleDropzone.addEventListener("dragover", (e) => { e.preventDefault(); singleDropzone.classList.add("dragover"); });
singleDropzone.addEventListener("dragleave", () => singleDropzone.classList.remove("dragover"));
singleDropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  singleDropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleSingleFile(e.dataTransfer.files[0]);
});
singleFileInput.addEventListener("change", () => {
  if (singleFileInput.files.length) handleSingleFile(singleFileInput.files[0]);
});

let singleSelectedFile = null;

async function handleSingleFile(file) {
  singleSelectedFile = file;
  // The crop stage must already be visible (not display:none) before
  // loadFile() measures it via getBoundingClientRect() to position the
  // crop rectangle -- a hidden element always measures as 0x0.
  singleDropzone.style.display = "none";
  singleCropArea.style.display = "block";
  await singleCrop.loadFile(file);
}

document.getElementById("single-example-btn").addEventListener("click", async () => {
  const resp = await fetch(`${API_BASE}/api/example?condition=stemi&region=inferior`);
  const blob = await resp.blob();
  const file = new File([blob], "example.png", { type: "image/png" });
  await handleSingleFile(file);
});

document.getElementById("single-reset-btn").addEventListener("click", () => {
  singleSelectedFile = null;
  singleCropArea.style.display = "none";
  singleDropzone.style.display = "block";
  singleFileInput.value = "";
  document.getElementById("single-results").innerHTML = "";
  document.getElementById("single-error").style.display = "none";
});

document.getElementById("single-analyze-btn").addEventListener("click", async () => {
  if (!singleSelectedFile) return;
  const loading = document.getElementById("single-loading");
  const errorBox = document.getElementById("single-error");
  const results = document.getElementById("single-results");
  errorBox.style.display = "none";
  results.innerHTML = "";
  loading.style.display = "flex";

  const form = new FormData();
  form.append("image", singleSelectedFile);
  if (!singleCrop.getFullBoxSelected()) {
    form.append("crop_box", JSON.stringify(singleCrop.getNormalizedBox()));
  }
  const pxX = document.getElementById("single-px-x").value;
  const pxY = document.getElementById("single-px-y").value;
  if (pxX) form.append("px_per_mm_x", pxX);
  if (pxY) form.append("px_per_mm_y", pxY);

  try {
    const resp = await fetch(`${API_BASE}/api/interpret`, { method: "POST", body: form });
    if (!resp.ok) throw new Error(await parseErrorResponse(resp));
    const report = await resp.json();
    renderReport(results, report);
  } catch (e) {
    errorBox.textContent = `Could not analyze this image: ${e.message}`;
    errorBox.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
});

// ---------- Serial ECG tab ----------

const serialStudyList = document.getElementById("serial-study-list");
let serialStudyCount = 0;
const serialStudies = new Map(); // id -> { crop, file }

function addSerialStudy() {
  const id = serialStudyCount++;
  const card = el("div", "study-card");
  card.dataset.id = id;

  const header = el("div", "row between");
  header.appendChild(el("strong", null, `ECG ${id + 1}`));
  const removeBtn = el("button", "secondary", "Remove");
  removeBtn.type = "button";
  removeBtn.addEventListener("click", () => {
    if (serialStudyList.children.length <= 2) return;
    serialStudies.delete(id);
    card.remove();
  });
  header.appendChild(removeBtn);
  card.appendChild(header);

  const dz = el("div", "dropzone", "Click or drop an ECG image");
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  dz.appendChild(input);
  card.appendChild(dz);

  const cropStage = el("div", "crop-stage");
  cropStage.style.display = "none";
  card.appendChild(cropStage);

  const dtRow = el("div", "row");
  const dtLabel = el("label", null, "Date & time taken");
  const dtInput = document.createElement("input");
  dtInput.type = "datetime-local";
  const dtWrap = el("div");
  dtWrap.appendChild(dtLabel);
  dtWrap.appendChild(dtInput);
  dtRow.appendChild(dtWrap);
  card.appendChild(dtRow);

  const cropSelector = new CropSelector(cropStage);
  serialStudies.set(id, { crop: cropSelector, file: null, dtInput });

  const onFile = async (file) => {
    serialStudies.get(id).file = file;
    dz.style.display = "none";
    cropStage.style.display = "block";
    await cropSelector.loadFile(file);
  };
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files.length) onFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) onFile(input.files[0]);
  });

  serialStudyList.appendChild(card);
}

document.getElementById("serial-add-btn").addEventListener("click", addSerialStudy);
addSerialStudy();
addSerialStudy();

document.getElementById("serial-compare-btn").addEventListener("click", async () => {
  const loading = document.getElementById("serial-loading");
  const errorBox = document.getElementById("serial-error");
  const results = document.getElementById("serial-results");
  errorBox.style.display = "none";
  results.innerHTML = "";

  const entries = Array.from(serialStudies.values());
  if (entries.length < 2) {
    errorBox.textContent = "Add at least 2 ECGs to compare.";
    errorBox.style.display = "block";
    return;
  }
  for (const entry of entries) {
    if (!entry.file) {
      errorBox.textContent = "Every ECG needs an uploaded image.";
      errorBox.style.display = "block";
      return;
    }
    if (!entry.dtInput.value) {
      errorBox.textContent = "Every ECG needs a date/time.";
      errorBox.style.display = "block";
      return;
    }
  }

  loading.style.display = "flex";
  const form = new FormData();
  const cropBoxes = [];
  for (const entry of entries) {
    form.append("images", entry.file);
    form.append("timestamps", new Date(entry.dtInput.value).toISOString());
    cropBoxes.push(entry.crop.getFullBoxSelected() ? null : entry.crop.getNormalizedBox());
  }
  if (cropBoxes.some((b) => b !== null)) {
    form.append("crop_boxes_json", JSON.stringify(cropBoxes));
  }

  try {
    const resp = await fetch(`${API_BASE}/api/interpret-serial`, { method: "POST", body: form });
    if (!resp.ok) throw new Error(await parseErrorResponse(resp));
    const body = await resp.json();
    renderEvolution(results, body.evolution);
    const latestPanel = el("div", "panel");
    latestPanel.appendChild(el("h3", "section-title", "Latest study interpretation"));
    const latestInner = el("div");
    renderReport(latestInner, body.latest_report);
    latestPanel.appendChild(latestInner);
    results.appendChild(latestPanel);
  } catch (e) {
    errorBox.textContent = `Could not compare these ECGs: ${e.message}`;
    errorBox.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
});
