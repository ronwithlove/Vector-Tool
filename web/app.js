"use strict";

const DEFAULTS = {
  merge_de: 10, max_colors: 12, min_area: 40, strip_width: 2, denoise: 2,
  smooth: 1, tolerance: 1, corner_angle: 50,
  regularize: true, detect_circles: true, remove_bg: false,
};
const SLIDERS = ["merge_de", "max_colors", "min_area", "strip_width", "denoise", "smooth", "tolerance", "corner_angle"];
const TOGGLES = ["regularize", "detect_circles", "remove_bg"];

const $ = (id) => document.getElementById(id);
const state = {
  imgId: null, imgW: 0, imgH: 0, fileName: "vector",
  params: loadParams(), locked: null, result: null,
  scale: 1, tx: 0, ty: 0, mode: "vec", split: 0.5, anchors: false, seq: 0,
};

function loadParams() {
  try { return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem("vt.params") || "{}")); }
  catch { return Object.assign({}, DEFAULTS); }
}
function saveParams() { try { localStorage.setItem("vt.params", JSON.stringify(state.params)); } catch {} }

// ---------------------------------------------------------------- 参数面板
function fmt(id, v) {
  if (id === "min_area") return v + "";
  if (id === "corner_angle") return v + "°";
  return (+v).toString();
}
function bindControls() {
  for (const id of SLIDERS) {
    const el = $(id), out = el.parentElement.querySelector("output");
    el.value = state.params[id]; out.textContent = fmt(id, el.value);
    el.addEventListener("input", () => {
      out.textContent = fmt(id, el.value);
      state.params[id] = +el.value; saveParams(); schedule();
    });
  }
  for (const id of TOGGLES) {
    const el = $(id);
    el.checked = !!state.params[id];
    el.addEventListener("change", () => { state.params[id] = el.checked; saveParams(); schedule(); });
  }
  $("resetParams").onclick = () => {
    state.params = Object.assign({}, DEFAULTS); saveParams();
    for (const id of SLIDERS) { $(id).value = state.params[id]; $(id).parentElement.querySelector("output").textContent = fmt(id, state.params[id]); }
    for (const id of TOGGLES) $(id).checked = state.params[id];
    state.locked = null; schedule(0);
  };
  $("lockState").onclick = () => { if (state.locked) { state.locked = null; schedule(0); } };
}

// ---------------------------------------------------------------- 上传
async function uploadBlob(blob, name) {
  setBusy(true);
  const r = await fetch("/api/upload", { method: "POST", body: blob });
  if (!r.ok) { setBusy(false); alert((await r.json()).detail || "上传失败"); return; }
  const j = await r.json();
  state.imgId = j.id; state.imgW = j.width; state.imgH = j.height;
  state.fileName = (name || "vector").replace(/\.[^.]+$/, "");
  state.locked = null; state.result = null;
  const url = URL.createObjectURL(blob);
  $("origImg").src = url;
  $("vecLayer").innerHTML = "";
  $("empty").style.display = "none";
  $("stage").classList.add("on");
  $("stSize").textContent = `${j.width} × ${j.height}`;
  fit();
  run();
}
function bindUpload() {
  const drop = $("drop"), file = $("file");
  file.onchange = () => file.files[0] && uploadBlob(file.files[0], file.files[0].name);
  for (const el of [drop, $("viewport")]) {
    el.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    el.addEventListener("dragleave", () => drop.classList.remove("over"));
    el.addEventListener("drop", (e) => {
      e.preventDefault(); drop.classList.remove("over");
      const f = e.dataTransfer.files[0]; if (f) uploadBlob(f, f.name);
    });
  }
  window.addEventListener("paste", (e) => {
    for (const it of e.clipboardData.items) if (it.type.startsWith("image/")) { uploadBlob(it.getAsFile(), "pasted.png"); break; }
  });
  document.querySelectorAll("[data-sample]").forEach((b) => b.onclick = async () => {
    const name = b.dataset.sample;
    const blob = await (await fetch("/samples/" + name)).blob();
    uploadBlob(blob, name);
  });
}

// ---------------------------------------------------------------- 处理
let timer = null;
function schedule(ms = 250) { clearTimeout(timer); timer = setTimeout(run, ms); }
function setBusy(on) { $("busy").classList.toggle("on", on); }

async function run() {
  if (!state.imgId) return;
  const seq = ++state.seq;
  const params = Object.assign({}, state.params);
  if (state.locked) params.palette = state.locked;
  setBusy(true);
  try {
    const r = await fetch("/api/vectorize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.imgId, params }),
    });
    const j = await r.json();
    if (seq !== state.seq) return;
    if (!r.ok) throw new Error(j.detail || "处理失败");
    state.result = j;
    $("vecLayer").innerHTML = j.svg;
    renderPalette(); renderStats(); drawOverlay();
    $("download").disabled = false;
    $("stage").classList.toggle("transparent", !!state.params.remove_bg);
  } catch (e) {
    if (seq === state.seq) { $("stTime").textContent = e.message; $("stTime").className = "error"; }
  } finally {
    if (seq === state.seq) setBusy(false);
  }
}

function renderStats() {
  const s = state.result.stats;
  $("stColors").textContent = s.colors;
  $("stShapes").textContent = s.shapes;
  $("stAnchors").textContent = s.anchors;
  $("stTime").className = "muted";
  $("stTime").textContent = `${s.total_ms} ms`;
}

// ---------------------------------------------------------------- 调色板
function renderPalette() {
  const box = $("palette"); box.innerHTML = "";
  const pal = state.result.palette;
  pal.forEach((c, i) => {
    const chip = document.createElement("div"); chip.className = "chip";
    chip.innerHTML = `<div class="sw" style="background:${c.out}" title="${c.out}"><input type="color" value="${c.out}"></div>
      <div class="pct">${c.share >= 0.001 ? (c.share * 100).toFixed(c.share < 0.1 ? 1 : 0) : "<0.1"}%</div>
      <button class="x" title="删除，并入最近的颜色">×</button>`;
    chip.querySelector("input").addEventListener("change", (e) => {
      state.locked = pal.map((p, k) => ({ match: p.match, out: k === i ? e.target.value : p.out }));
      schedule(0);
    });
    chip.querySelector(".x").onclick = () => {
      if (pal.length <= 1) return;
      state.locked = pal.filter((_, k) => k !== i).map((p) => ({ match: p.match, out: p.out }));
      schedule(0);
    };
    box.appendChild(chip);
  });
  const ls = $("lockState");
  if (state.locked) { ls.textContent = "已锁定 · 点此解锁"; ls.classList.add("locked"); }
  else { ls.textContent = "自动识别"; ls.classList.remove("locked"); }
  $("merge_de").disabled = $("max_colors").disabled = !!state.locked;
}

// ---------------------------------------------------------------- 视图：缩放/平移/对比
function applyView() {
  const st = $("stage");
  st.style.width = state.imgW * state.scale + "px";
  st.style.height = state.imgH * state.scale + "px";
  st.style.transform = `translate(${state.tx}px, ${state.ty}px)`;
  $("origImg").style.imageRendering = state.scale >= 3 ? "pixelated" : "auto";
  $("zoomLabel").textContent = Math.round(state.scale * 100) + "%";
  applyMode();
  drawOverlay();
}
function fit() {
  const vp = $("viewport").getBoundingClientRect();
  if (!state.imgW) return;
  const s = Math.min((vp.width - 60) / state.imgW, (vp.height - 60) / state.imgH);
  state.scale = Math.max(0.02, s);
  state.tx = (vp.width - state.imgW * state.scale) / 2;
  state.ty = (vp.height - state.imgH * state.scale) / 2;
  applyView();
}
function zoomAt(f, cx, cy) {
  const ns = Math.min(64, Math.max(0.02, state.scale * f));
  const k = ns / state.scale;
  state.tx = cx - (cx - state.tx) * k;
  state.ty = cy - (cy - state.ty) * k;
  state.scale = ns; applyView();
}
function applyMode() {
  const m = state.mode, vec = $("vecLayer"), img = $("origImg"), dv = $("divider");
  img.style.visibility = m === "vec" ? "hidden" : "visible";
  vec.style.visibility = m === "orig" ? "hidden" : "visible";
  vec.style.clipPath = m === "split" ? `inset(0 0 0 ${state.split * 100}%)` : "none";
  dv.style.display = m === "split" ? "block" : "none";
  dv.style.left = state.split * 100 + "%";
}
function bindView() {
  const vp = $("viewport");
  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    const f = Math.exp(-(e.deltaY) * (e.ctrlKey ? 0.01 : 0.0018));
    zoomAt(f, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  let drag = null;
  vp.addEventListener("pointerdown", (e) => {
    if (e.target.closest("#divider")) {
      drag = { split: true }; vp.setPointerCapture(e.pointerId); return;
    }
    drag = { x: e.clientX, y: e.clientY, tx: state.tx, ty: state.ty };
    vp.classList.add("dragging"); vp.setPointerCapture(e.pointerId);
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (drag.split) {
      const r = $("stage").getBoundingClientRect();
      state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      applyMode(); drawOverlay(); return;
    }
    state.tx = drag.tx + e.clientX - drag.x; state.ty = drag.ty + e.clientY - drag.y; applyView();
  });
  const end = () => { drag = null; vp.classList.remove("dragging"); };
  vp.addEventListener("pointerup", end); vp.addEventListener("pointercancel", end);
  $("zoomIn").onclick = () => { const r = vp.getBoundingClientRect(); zoomAt(1.5, r.width / 2, r.height / 2); };
  $("zoomOut").onclick = () => { const r = vp.getBoundingClientRect(); zoomAt(1 / 1.5, r.width / 2, r.height / 2); };
  $("zoomFit").onclick = fit;
  document.querySelectorAll("#viewMode button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("#viewMode button").forEach((x) => x.classList.toggle("on", x === b));
    state.mode = b.dataset.mode; applyMode(); drawOverlay();
  });
  $("showAnchors").onchange = (e) => { state.anchors = e.target.checked; drawOverlay(); };
  window.addEventListener("resize", () => { resizeOverlay(); drawOverlay(); });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "a") { $("showAnchors").click(); }
    if (e.key === "0") fit();
    if (e.key === "1") document.querySelector('[data-mode="orig"]').click();
    if (e.key === "2") document.querySelector('[data-mode="vec"]').click();
    if (e.key === "3") document.querySelector('[data-mode="split"]').click();
  });
}

// ---------------------------------------------------------------- 锚点叠加层
function resizeOverlay() {
  const c = $("overlay"), r = $("viewport").getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  c.style.width = r.width + "px"; c.style.height = r.height + "px";
}
let path2dCache = { key: null, list: [] };
function drawOverlay() {
  const c = $("overlay"), ctx = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height);
  if (!state.anchors || !state.result || state.mode === "orig") return;
  const shapes = state.result.shapes;
  if (path2dCache.key !== state.result) path2dCache = { key: state.result, list: shapes.map((s) => new Path2D(s.d)) };
  const s = state.scale;
  ctx.save();
  if (state.mode === "split") {
    const x0 = state.tx + state.imgW * s * state.split;
    ctx.beginPath(); ctx.rect(x0 * dpr, 0, c.width, c.height); ctx.clip();
  }
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * state.tx, dpr * state.ty);
  ctx.lineWidth = 1 / s; ctx.strokeStyle = "#4c8dff";
  for (const p of path2dCache.list) ctx.stroke(p);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const a = 4.5; ctx.fillStyle = "#4c8dff"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
  for (const sh of shapes) for (const [x, y] of sh.anchors) {
    const X = state.tx + x * s, Y = state.ty + y * s;
    ctx.fillRect(X - a / 2, Y - a / 2, a, a);
  }
  ctx.restore();
}

// ---------------------------------------------------------------- 导出
function bindExport() {
  $("download").onclick = () => {
    if (!state.result) return;
    const blob = new Blob([state.result.svg], { type: "image/svg+xml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = state.fileName + ".svg"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
}

bindControls(); bindUpload(); bindView(); bindExport(); resizeOverlay();
