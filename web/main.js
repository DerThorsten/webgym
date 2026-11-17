// The page (main thread): layout, progress bars, controls. Python, the simulation and
// the WebGL2 rendering run in worker.js on an OffscreenCanvas.

import { setupPanel } from "./checkpoints.js";
import { plot } from "./plot.js";

const stage = document.getElementById("stage");
let canvas = document.getElementById("canvas"); // replaced on a worker restart
let worker = null; // the pyjs worker (replaced on a restart after a fatal error)
const send = (message, transfer) => worker?.postMessage(message, transfer);
const statusEl = document.getElementById("status");
const stateEl = document.getElementById("state");
const plotEl = document.getElementById("plot");
let stats = []; // training stats per iteration (from the worker)
const afterInstall = new Map(); // package -> the command to send once it is loaded
const requested = new Set(); // package groups asked for in this worker (see loadPackage)
const brokenPackages = new Set(); // package groups that crashed python (this page visit)
const bars = new Map(); // progress id -> {label, value, bar}

/** show a message (errors in red) */
function status(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

/** one progress bar per id (startup, core, pytorch, jax, ...): phase + bytes */
function showProgress({ id, phase, done, total }) {
  if (!bars.has(id)) {
    const el = document.createElement("div");
    el.className = "bar";
    el.innerHTML = `<label><span></span><span></span></label><progress></progress>`;
    document.getElementById("bars").append(el);
    bars.set(id, { el, label: el.querySelectorAll("span"), bar: el.querySelector("progress") });
  }
  const { el, label, bar } = bars.get(id);
  const mb = (n) => (n / 1e6).toFixed(1);
  label[0].textContent = `${id}: ${phase}`;
  label[1].textContent = total ? `${mb(done)} / ${mb(total)} MB` : "";
  if (total) bar.value = done / total;
  else bar.removeAttribute("value"); // indeterminate
  if (phase === "done" || phase === "ready") bar.value = 1;
  el.hidden = phase === "done"; // startup / core bars go away; package bars stay ("ready")
  // the "please wait" overlay on the canvas while anything is loading
  if (phase === "done" || phase === "ready") loading.delete(id);
  else loading.set(id, total ? `${phase} ${mb(done)} / ${mb(total)} MB` : phase);
  showBusy();
}

const loading = new Map(); // id -> what it is doing (startup, core, pytorch, jax)
/** the hourglass overlay over the canvas while the user waits: the start, or a
 * package an action waits for (the background loading shows in the panel only) */
function showBusy() {
  const el = document.getElementById("busy");
  const waiting = [...loading].filter(([id]) => id === "startup" || id === "core" || afterInstall.has(id));
  el.hidden = waiting.length === 0;
  if (el.hidden) return;
  const lines = waiting.map(([id, what]) => `${id}: ${what}`).join("<br>");
  el.innerHTML = `<div class="big">⏳ please wait</div><div class="small">${lines}</div>`;
}

// drawing-buffer px per CSS px: devicePixelRatio, but at most MAX_PIXELS in the
// buffer (a smaller buffer, stretched by the browser, draws faster); ?dpr= overrides
const MAX_PIXELS = 1e6;
const DPR_PARAM = parseFloat(new URLSearchParams(location.search).get("dpr"));
let scale = 1;

/** the drawing buffer follows the canvas' CSS size x scale */
function resize() {
  const w = Math.max(1, canvas.clientWidth);
  const h = Math.max(1, canvas.clientHeight);
  scale = DPR_PARAM > 0 ? DPR_PARAM : Math.min(window.devicePixelRatio || 1, Math.sqrt(MAX_PIXELS / (w * h)));
  send({ type: "resize", width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)) });
}

const openGroups = new Set(); // control groups the user opened (stay open on a rebuild)

/** the app's widgets (App.controls(): select / slider / button), rebuilt on change;
 * changes go back as commands */
function showControls(controls) {
  const root = document.getElementById("controls");
  root.replaceChildren();
  const command = (name, value) => send({ type: "command", name, value });
  const buttons = document.createElement("div");
  const groups = new Map(); // group name -> <details> (e.g. the algorithm's parameters)
  for (const c of controls) {
    let parent = root;
    if (c.group) {
      if (!groups.has(c.group)) {
        const details = Object.assign(document.createElement("details"), { className: "group", open: openGroups.has(c.group) });
        details.append(Object.assign(document.createElement("summary"), { textContent: c.group }));
        details.ontoggle = () => (details.open ? openGroups.add(c.group) : openGroups.delete(c.group));
        root.append(details);
        groups.set(c.group, details);
      }
      parent = groups.get(c.group);
    }
    if (c.type === "button") {
      const b = document.createElement("button");
      b.textContent = c.label;
      b.className = c.name === "train" ? "primary" : "";
      b.onclick = () => {
        command(c.name);
        b.blur(); // space would press it again
      };
      buttons.append(b);
      continue;
    }
    const el = document.createElement("div");
    el.className = "control";
    if (c.type === "hint") {
      root.append(Object.assign(document.createElement("div"), { className: c.active ? "hint active" : "hint", textContent: c.text }));
      continue;
    }
    if (c.type === "info") {
      // first line always visible, the rest on click
      const [first, ...rest] = c.text.split("\n");
      const details = Object.assign(document.createElement("details"), { className: "info" });
      const summary = Object.assign(document.createElement("summary"), { textContent: first });
      details.append(summary, rest.join("\n").trim());
      root.append(details);
      continue;
    }
    const label = document.createElement("label");
    label.innerHTML = `<span>${c.label}</span><span></span>`;
    const value = label.lastChild;
    let input;
    if (c.type === "checkbox") {
      el.className = "control checkbox";
      input = Object.assign(document.createElement("input"), { type: "checkbox", checked: c.value });
      input.onchange = () => {
        command(c.name, input.checked);
        input.blur();
      };
      const text = Object.assign(document.createElement("span"), { textContent: c.label });
      const wrap = document.createElement("label");
      wrap.append(input, text);
      el.append(wrap);
      root.append(el);
      continue;
    }
    if (c.type === "select") {
      input = document.createElement("select");
      // options: values, or [value, label] pairs
      for (const o of c.options) {
        const [v, text] = Array.isArray(o) ? o : [o, o];
        input.add(new Option(text, v, false, v === c.value));
      }
      input.onchange = () => {
        command(c.name, input.value);
        input.blur(); // the arrow keys go to the env again
      };
    } else {
      input = Object.assign(document.createElement("input"), { type: "range", min: c.min, max: c.max, step: c.step, value: c.value });
      // optional labels per slider position (e.g. "4x")
      const show = () => (value.textContent = c.labels ? c.labels[input.value] : input.value);
      show();
      input.oninput = show;
      input.onchange = () => {
        command(c.name, Number(input.value));
        input.blur();
      };
    }
    el.append(label, input);
    parent.append(el);
  }
  root.append(buttons);
  // the URL names the current env / algo: reload or share the link to get them
  const url = new URL(location.href);
  for (const c of controls) if (c.name === "env" || c.name === "algo") url.searchParams.set(c.name, c.value);
  history.replaceState(null, "", url);
}

const LOSS_COLORS = { policy_loss: "#e06c75", value_loss: "#98c379", entropy: "#c678dd" };
const lossPlotEl = document.getElementById("loss-plot");

/** the reward curves of the current run; the losses (if the algorithm has any, e.g.
 * PPO) in a second plot */
function showHistory() {
  plot(plotEl, [
    { label: "mean return", color: "#4fa3ff", values: stats.map((h) => h.mean_return) },
    { label: "best return", color: "#f5a742", values: stats.map((h) => h.best_return) },
  ]);
  const keys = Object.keys(LOSS_COLORS).filter((k) => stats.some((h) => k in h));
  lossPlotEl.hidden = keys.length === 0;
  if (keys.length)
    plot(lossPlotEl, keys.map((k) => ({ label: k.replace("_", " "), color: LOSS_COLORS[k], values: stats.map((h) => h[k] ?? 0) })));
}
new ResizeObserver(showHistory).observe(plotEl);

/** watch mode, the episode is over: its score over the canvas (+ best, the last few) */
function showScore(s) {
  const el = document.getElementById("score");
  el.hidden = !(s.mode === "watch" && s.episode_over && s.scores.length);
  if (el.hidden) return;
  const score = s.scores.at(-1);
  el.innerHTML =
    `<div class="big">Score ${Math.round(score)}${s.new_best ? " · new best!" : ""}</div>` +
    `<div class="small">best ${Math.round(s.best_score)} · last ${s.scores.map(Math.round).join(", ")}</div>`;
}

/** a message from the worker */
function onMessage({ data }) {
  if (data.type === "progress") showProgress(data);
  else if (data.type === "ready") {
    if (Date.now() > keepStatusUntil) status(""); // (a crash notice stays a while)
    // in the background, a bit later: the restore of the autosave goes first (the
    // installs keep python busy)
    setTimeout(() => Object.keys(PACKAGES).forEach(loadPackage), 3000);
  }
  else if (data.type === "controls") {
    showControls(data.controls);
    checkpoints.onControls(data.controls, freshWorker);
    freshWorker = false;
  } else if (data.type === "checkpoint" || data.type === "delete_checkpoint") checkpoints.onMessage(data);
  else if (data.type === "compile_request") compileRequest(data);
  else if (data.type === "need_package") {
    // e.g. Train with PPO (pytorch) before pytorch is loaded: load it, then train
    status(`loading ${data.package} for this algorithm…`);
    afterInstall.set(data.package, data.command);
    loadPackage(data.package); // (already loading in the background: then it just waits)
    showBusy();
  }
  else if (data.type === "history") {
    stats = data.reset ? data.entries : stats.concat(data.entries);
    showHistory();
  } else if (data.type === "state") {
    const s = data.state;
    showScore(s);
    const last = stats.at(-1);
    stateEl.textContent =
      `mode      ${s.mode}\n` +
      `policy    ${s.policy}\n` +
      (s.mode === "watch"
        ? `episode   #${s.episode}, return ${s.episode_return.toFixed(0)}` +
          (s.episode_over ? (s.auto_reset ? " (over, restarting)" : " (over, keeps going)") : "") +
          "\n"
        : `iteration ${s.iteration}\n`) +
      (last ? `mean/best ${last.mean_return.toFixed(0)} / ${last.best_return.toFixed(0)}\n` : "") +
      `frame     ${s.frame_ms.toFixed(1)} ms`;
  } else if (data.type === "installed") {
    console.info(`${data.name} installed:`, data.seconds, data.result);
    showPackageResult(data.name, `${data.result} (loaded in ${data.seconds.total.toFixed(1)} s)`);
    if (afterInstall.has(data.name)) {
      send({ type: "command", name: afterInstall.get(data.name) });
      afterInstall.delete(data.name);
    }
  } else if (data.type === "error") status(`error: ${data.error}`, true);
  else if (data.type === "fatal") restartAfterCrash(data.error);
}

// ---- the worker, restarted after fatal errors -------------------------------------
// A wasm trap (memory access out of bounds, unreachable, abort) leaves python's memory
// in an unknown state: the worker is thrown away and a fresh one starts on a fresh
// canvas (a canvas can be transferred only once). Env / algo survive in the URL.
// More than MAX_CRASHES within a minute: no more restarts (no crash loop).
const MAX_CRASHES = 3;
const crashes = [];
// checkpoints (IndexedDB): a new worker restores the autosave of its env + algo
const checkpoints = setupPanel(send);
let freshWorker = true;
let keepStatusUntil = 0; // the crash notice is not cleared by the restarted worker before

/** start a new worker rendering into a new canvas */
function startWorker() {
  const fresh = Object.assign(document.createElement("canvas"), { id: "canvas" });
  canvas.replaceWith(fresh);
  canvas = fresh;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onMessage;
  freshWorker = true;
  requested.clear(); // (a new worker loads them again, from the package cache)
  worker.onerror = (e) => {
    e.preventDefault();
    if (/RuntimeError: (unreachable|memory access out of bounds|null function|indirect call)|Aborted\(|out of memory/i.test(e.message))
      restartAfterCrash(e.message);
    else status(`worker error: ${e.message}`, true);
  };
  const offscreen = canvas.transferControlToOffscreen();
  send({ type: "init", canvas: offscreen, base: location.href }, [offscreen]);
  resize();
}

/** the worker crashed (`error`): restart it, unless it crashes too often */
function restartAfterCrash(error) {
  console.error("python crashed:", error);
  // a package that was loading / being checked: probably the cause, skipped from now on
  for (const id of loading.keys()) {
    if (id in PACKAGES) {
      brokenPackages.add(id);
      showPackageResult(id, `crashed python while loading / checking (${error.split("\n")[0]}), skipped`);
      afterInstall.delete(id);
    }
  }
  loading.clear();
  showBusy();
  worker.terminate();
  worker = null;
  const now = Date.now();
  crashes.push(now);
  while (crashes.length && now - crashes[0] > 60_000) crashes.shift();
  if (crashes.length > MAX_CRASHES) {
    status(`python crashed ${crashes.length} times within a minute, giving up:\n${error}`, true);
    const reload = Object.assign(document.createElement("button"), { textContent: "Reload page" });
    reload.onclick = () => location.reload();
    statusEl.append(document.createElement("br"), reload);
    return;
  }
  status(`python crashed and was restarted (training continues from the last autosave):\n${error.split("\n")[0]}`, true);
  keepStatusUntil = now + 10_000;
  setTimeout(() => statusEl.textContent.startsWith("python crashed and was") && status(""), 10_000);
  stats = [];
  showHistory();
  startWorker();
}

new ResizeObserver(resize).observe(stage);
startWorker();
// for test_page.py: a real wasm trap in the worker (checks the restart)
window.webgymCrashTest = () => send({ type: "command", name: "__crash__" });

// pointer input -> worker (drawing-buffer px): drag bodies, pan, wheel zoom. The
// listeners are on the stage: they survive the canvas being replaced on a restart
const sendEvent = (kind, e, delta = 0) => send({ type: "event", kind, x: e.offsetX * scale, y: e.offsetY * scale, delta });
stage.style.touchAction = "none";
stage.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || e.target !== canvas) return;
  canvas.setPointerCapture(e.pointerId);
  sendEvent("down", e);
});
stage.addEventListener("pointermove", (e) => e.buttons & 1 && sendEvent("move", e));
stage.addEventListener("pointerup", (e) => sendEvent("up", e));
stage.addEventListener("pointercancel", (e) => sendEvent("up", e));
stage.addEventListener("pointerleave", (e) => sendEvent("leave", e));
// keys -> worker (the human controller), unless the user types into a widget
const KEY_NAMES = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down", " ": "space" };
const keyName = (e) => KEY_NAMES[e.key] ?? (e.key.length === 1 ? e.key.toLowerCase() : null);
for (const kind of ["keydown", "keyup"]) {
  window.addEventListener(kind, (e) => {
    const key = keyName(e);
    if (!key || e.repeat || e.target.closest?.("input, select, textarea")) return;
    if (KEY_NAMES[e.key]) e.preventDefault(); // no scrolling with the arrows / space
    send({ type: "event", kind, key });
  });
}
window.addEventListener("blur", () => {
  for (const key of Object.values(KEY_NAMES)) send({ type: "event", kind: "keyup", key });
});
stage.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    sendEvent("wheel", e, e.deltaMode === 1 ? 16 * e.deltaY : e.deltaY);
  },
  { passive: false },
);

// The big packages, always loaded in the background after the start (from the
// worker's package cache after the first visit), one after the other; each is
// checked once loaded (the result shows in the packages panel). "webgpu": IREE (jax ->
// WebGPU), checked with a small benchmark on the GPU (webgym/gpu.py)
const PACKAGES = {
  pytorch: {
    packages: ["pytorch"],
    check: `
import time, torch
torch.autograd.set_multithreading_enabled(False)
t = time.perf_counter()
loss = torch.nn.Linear(8, 2)(torch.randn(64, 8)).pow(2).mean()
loss.backward()
f"torch {torch.__version__}, fwd+bwd {1e3 * (time.perf_counter() - t):.1f} ms"
`,
  },
  jax: {
    packages: ["jax"],
    check: `
import time, jax, jax.numpy as jnp
t = time.perf_counter()
jax.jit(jax.grad(lambda w: (w ** 2).sum()))(jnp.ones(8)).block_until_ready()
f"jax {jax.__version__}, jit+grad {1e3 * (time.perf_counter() - t):.1f} ms"
`,
  },
  webgpu: {
    // (the compiler runs in the compile worker, see below)
    packages: ["iree-base-runtime"],
    check: "from webgym import gpu\nawait gpu.probe()",
  },
};

// ---- the compile worker: IREE's compiler (StableHLO -> WebGPU) -----------------------
// jaxlib and the IREE compiler both bring MLIR python bindings, which abort python when
// loaded into one process: the compiler lives in its own compute-only worker, started
// on the first compile request (python start + 29 MB compiler, cached after that)
let compiler = null;

/** the compile worker: {exec(code) -> value}; resolves when the compiler is loaded */
function compileWorker() {
  compiler ??= new Promise((resolve, reject) => {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const pending = new Map();
    let nextId = 0;
    const exec = (code) =>
      new Promise((res, rej) => {
        const id = nextId++;
        pending.set(id, { res, rej });
        w.postMessage({ type: "exec", id, code });
      });
    w.onmessage = ({ data }) => {
      if (data.type === "progress") {
        // its own bars would mix with the main worker's: one "compiler" entry
        showProgress({ ...data, id: "compiler", phase: `${data.id}: ${data.phase}` });
        if (data.id === "startup" && data.phase === "done")
          w.postMessage({ type: "install", name: "compiler", packages: ["iree-base-compiler"], check: "'ok'" });
      } else if (data.type === "installed") {
        showProgress({ id: "compiler", phase: "done" });
        resolve({ exec });
      } else if (data.type === "result") pending.get(data.id).res(data.value);
      else if (data.type === "error" || data.type === "fatal") {
        for (const p of pending.values()) p.rej(new Error(data.error));
        pending.clear();
        reject(new Error(data.error));
      }
    };
    w.postMessage({ type: "init", canvas: null, base: location.href });
  });
  return compiler;
}

/** a compile request of the main worker: StableHLO -> IREE module for WebGPU (base64) */
async function compileRequest({ id, mlir }) {
  try {
    const { exec } = await compileWorker();
    const data = await exec(`from webgym import gpu\ngpu.compile_b64(${JSON.stringify(mlir)})`);
    send({ type: "compile_result", id, data });
  } catch (e) {
    send({ type: "compile_result", id, error: String(e?.message ?? e) });
  }
}

/** load a package group in the worker once (installs run one after the other); not
 * one that crashed python (wasm trap) while it was loading / checked: no crash loop */
function loadPackage(name) {
  if (requested.has(name) || brokenPackages.has(name)) return;
  requested.add(name);
  send({ type: "install", name, packages: PACKAGES[name].packages, check: PACKAGES[name].check });
}

/** the result of a package's check, in the packages panel */
function showPackageResult(name, text) {
  const id = `package-${name}`;
  let el = document.getElementById(id);
  if (!el) {
    el = Object.assign(document.createElement("div"), { id, className: "package-result" });
    document.getElementById("packages").append(el);
  }
  el.textContent = `${name}: ${text}`;
}
