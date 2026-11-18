// The page (main thread): layout, progress bars, controls. Python runs in workers
// (worker.js):
// * the display worker: the simulation you watch / play, WebGL2 on an OffscreenCanvas
// * the training worker (desktop): trains as fast as it can, no drawing; snapshots of
//   its policy go to the display worker (replay in real time)
// On phones / tablets (or ?workers=1) one worker does both (role "single").

import { setupPanel } from "./checkpoints.js";
import { createEditor, forgetFile, openFile, saveFile } from "./editor.js";
import { drawNetwork } from "./network.js";
import { plot } from "./plot.js";

const stage = document.getElementById("stage");
let canvas = document.getElementById("canvas"); // replaced on a worker restart
const params = new URLSearchParams(location.search);
const SMALL_DEVICE = matchMedia("(pointer: coarse)").matches || (navigator.hardwareConcurrency || 2) <= 4;
const TWO_WORKERS = (params.get("workers") ?? (SMALL_DEVICE ? "1" : "2")) === "2";
let display = null; // the display worker (role "display" / "single")
let trainer = null; // the training worker (role "trainer"; null with one worker)
const toDisplay = (message, transfer) => display?.postMessage(message, transfer);
const toTraining = (message) => (TWO_WORKERS ? trainer : display)?.postMessage(message);
let lastSnapshot = null; // the trainer's latest policy (for a display that (re)starts)
let autoTrain = params.get("train") !== "0"; // start training once the page is up
let trainerState = null; // the trainer's state message (two workers)
// commands of the watched env only: to the display (and to the trainer too: its
// controls show these values as well)
const DISPLAY_COMMANDS = new Set(["controller", "auto_reset", "reset", "show_n", "replay_speed", "follow", "throw", "throw_every"]);

/** a widget changed: route the command to the worker(s) concerned */
function command(name, value) {
  const message = { type: "command", name, value };
  if (!TWO_WORKERS) return toDisplay(message);
  if (DISPLAY_COMMANDS.has(name) || name === "env" || name === "code") toDisplay(message);
  toTraining(message);
}
const statusEl = document.getElementById("status");
const stateEl = document.getElementById("state");
const plotEl = document.getElementById("plot");
let stats = []; // training stats per iteration (from the worker)
const afterInstall = new Map(); // package -> the command to send once it is loaded
const requested = new Set(); // package groups asked for in this worker (see loadPackage)
const brokenPackages = new Set(); // package groups that crashed python (this page visit)
const bars = new Map(); // progress id -> {label, value, bar}

/** a message: errors in red at the top of the panel, everything else only in the log
 * ("Packages & log" section); "" clears the top */
function status(text, isError = false) {
  if (text) log(text, isError);
  if (!isError && text) return;
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

/** a line in the log (newest first, the last 100) */
function log(text, isError = false) {
  const list = document.getElementById("log");
  const line = Object.assign(document.createElement("div"), { className: isError ? "log error" : "log" });
  line.textContent = `${new Date().toLocaleTimeString()} ${text.split("\n")[0]}`;
  line.title = text;
  list.prepend(line);
  while (list.children.length > 100) list.lastChild.remove();
}

/** one progress bar per id (startup, core, pytorch, jax, ...): phase + bytes */
function showProgress({ id, phase, done, total }) {
  if (!bars.has(id)) {
    const el = document.createElement("div");
    el.className = "bar";
    el.innerHTML = `<label><span></span><span></span></label><progress></progress>`;
    // the page's own start at the top; everything in the background (packages, the
    // training / compile worker) in the "Packages & log" section
    const top = id === "startup" || id === "core";
    document.getElementById(top ? "bars" : "package-bars").append(el);
    bars.set(id, { el, label: el.querySelectorAll("span"), bar: el.querySelector("progress") });
  }
  const { el, label, bar } = bars.get(id);
  const mb = (n) => (n / 1e6).toFixed(1);
  label[0].textContent = `${id}: ${phase}`;
  label[1].textContent = total ? `${mb(done)} / ${mb(total)} MB` : "";
  if (total) bar.value = done / total;
  else bar.removeAttribute("value"); // indeterminate
  if (phase === "done" || phase === "ready") bar.value = 1;
  el.hidden = phase === "done" || phase === "ready"; // (finished: its result line stays)
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
  toDisplay({ type: "resize", width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)) });
}

// the Code section: the env's / algorithm's python source, editable; Apply sends it
// to the worker(s), which write and reload it
let code = {}; // {env: {path, text}, algo: {path, text}} (as edited)
const codeKind = document.getElementById("code-kind");
const editor = createEditor(
  document.getElementById("code"),
  (text) => code[codeKind.value] && (code[codeKind.value].text = text),
  () => applyCode(),
);
function showCode(next) {
  for (const kind of ["env", "algo"]) if (code[kind]?.path !== next[kind]?.path) forgetFile(kind);
  code = next;
  showCodeKind();
}
function showCodeKind() {
  const { path = "", text = "" } = code[codeKind.value] ?? {};
  document.getElementById("code-path").textContent = path;
  if (editor.get() !== text) editor.set(text);
}
codeKind.onchange = showCodeKind;
const fileName = () => (code[codeKind.value]?.path ?? "code.py").split("/").pop();
function applyCode() {
  log(`applying the edited ${codeKind.selectedOptions[0].text} code`);
  command("code", JSON.stringify({ kind: codeKind.value, text: editor.get() }));
}
document.getElementById("code-apply").onclick = applyCode;
document.getElementById("code-open").onclick = async () => {
  try {
    const file = await openFile(codeKind.value);
    if (!file) return;
    editor.set(file.text);
    log(`opened ${file.name}: press Apply to run it`);
  } catch (e) {
    status(`could not open the file: ${e.message}`, true);
  }
};
document.getElementById("code-save").onclick = async () => {
  try {
    const name = await saveFile(codeKind.value, fileName(), editor.get());
    if (name) log(`saved ${name}`);
  } catch (e) {
    status(`could not save the file: ${e.message}`, true);
  }
};

const openGroups = new Set(); // control groups the user opened (stay open on a rebuild)

/** the app's widgets (App.controls(): select / slider / button), rebuilt on change;
 * changes go back as commands */
function showControls(controls) {
  // each control in its section ("environment", "training"), buttons at its end
  const sections = new Map();
  for (const name of ["environment", "training"]) {
    const el = document.getElementById(`controls-${name}`);
    const buttons = document.createElement("div");
    el.replaceChildren(buttons);
    sections.set(name, { el, buttons });
  }
  const groups = new Map(); // group name -> <details> (e.g. the algorithm's parameters)
  for (const c of controls) {
    const { buttons } = sections.get(c.section ?? "environment");
    // (appended before the section's buttons)
    const root = { append: (...nodes) => buttons.before(...nodes) };
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
      b.className = c.name === "train" ? "primary" : "";
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
    if (c.help) {
      // the explanation: a tooltip on the name, a popup on click (with a small marker)
      label.firstChild.title = c.help;
      label.firstChild.dataset.help = c.help;
      label.firstChild.append(Object.assign(document.createElement("span"), { className: "help", textContent: " ⓘ" }));
    }
    let input;
    if (c.type === "checkbox") {
      el.className = "control checkbox";
      input = Object.assign(document.createElement("input"), { type: "checkbox", checked: c.value, disabled: Boolean(c.disabled) });
      input.onchange = () => {
        command(c.name, input.checked);
        input.blur();
      };
      const text = Object.assign(document.createElement("span"), { textContent: c.label, title: c.help ?? "" });
      if (c.help) text.dataset.help = c.help;
      if (c.help) text.append(Object.assign(document.createElement("span"), { className: "help", textContent: " ⓘ" }));
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
      // (e.g. the population of an env that runs one agent at a time)
      input.disabled = Boolean(c.disabled);
      if (c.disabled) el.title = "this environment runs one agent at a time";
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

/** the state panel: the watched env (display) + the training (trainer, two workers) */
function showState() {
  const s = displayState;
  if (!s) return;
  const last = stats.at(-1);
  const t = TWO_WORKERS ? trainerState : s;
  const training = t?.mode === "train";
  stateEl.textContent =
    `training  ${training ? `running, iteration ${t.iteration}` : "stopped"}` +
    (TWO_WORKERS ? " (training worker)" : "") +
    "\n" +
    `policy    ${s.policy}\n` +
    (!training || TWO_WORKERS
      ? `episode   #${s.episode}, return ${s.episode_return.toFixed(0)}` +
        (s.episode_over ? (s.auto_reset ? " (over, restarting)" : " (over, keeps going)") : "") +
        "\n"
      : "") +
    (last ? `mean/best ${last.mean_return.toFixed(0)} / ${last.best_return.toFixed(0)}\n` : "") +
    `frame     ${s.frame_ms.toFixed(1)} ms`;
}
let displayState = null;

/** a message from a worker: `role` "display" (also "single") or "trainer" */
function onMessage(role, { data }) {
  const training = role === "trainer" || !TWO_WORKERS; // (the worker that trains)
  if (data.type === "progress") {
    // the trainer's own start: its own bars (no hourglass: the page is usable)
    const id = role === "trainer" && (data.id === "startup" || data.id === "core") ? `training worker ${data.id}` : data.id;
    showProgress({ ...data, id });
  } else if (data.type === "ready") {
    if (role !== "trainer" && Date.now() > keepStatusUntil) status(""); // (a crash notice stays a while)
    if (role === "display" && lastSnapshot) toDisplay({ type: "snapshot", data: lastSnapshot });
    // the big packages in the training worker, in the background, a bit later: the
    // restore of the autosave goes first (the installs keep python busy)
    if (training) setTimeout(() => Object.keys(PACKAGES).forEach(loadPackage), 3000);
  } else if (data.type === "snapshot") {
    lastSnapshot = data.data;
    toDisplay({ type: "snapshot", data: data.data });
  } else if (data.type === "state") {
    if (role === "trainer") trainerState = data.state;
    else {
      displayState = data.state;
      showScore(data.state);
      // the watched policy's network (weights + live activations), if there is one
      const net = data.state.network;
      document.getElementById("network-section").hidden = !net;
      if (net) drawNetwork(document.getElementById("network"), net);
    }
    showState();
  } else if (!training) {
    // the display worker of two: only its errors matter beyond this point
    if (data.type === "error") status(`error: ${data.error}`, true);
    else if (data.type === "fatal") restartAfterCrash(role, data.error);
  } else if (data.type === "controls") {
    showControls(data.controls);
    const restored = checkpoints.onControls(data.controls, freshWorker);
    freshWorker = false;
    // the page starts training (after restoring the autosave: it continues the run);
    // ?train=0 starts watching
    if (autoTrain) {
      autoTrain = false;
      restored.then(() => setTimeout(() => command("train"), 300));
    }
  } else if (data.type === "checkpoint" || data.type === "delete_checkpoint") checkpoints.onMessage(data);
  else if (data.type === "code") showCode(data.code);
  else if (data.type === "result" && execs.has(data.id)) {
    execs.get(data.id)(data.value);
    execs.delete(data.id);
  }
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
    showState();
  } else if (data.type === "installed") {
    console.info(`${data.name} installed:`, data.seconds, data.result);
    showPackageResult(data.name, `${data.result} (loaded in ${data.seconds.total.toFixed(1)} s)`);
    if (afterInstall.has(data.name)) {
      command(afterInstall.get(data.name));
      afterInstall.delete(data.name);
    }
  } else if (data.type === "error") status(`error: ${data.error}`, true);
  else if (data.type === "fatal") restartAfterCrash(role, data.error);
}

// ---- the worker, restarted after fatal errors -------------------------------------
// A wasm trap (memory access out of bounds, unreachable, abort) leaves python's memory
// in an unknown state: the worker is thrown away and a fresh one starts on a fresh
// canvas (a canvas can be transferred only once). Env / algo survive in the URL.
// More than MAX_CRASHES within a minute: no more restarts (no crash loop).
const MAX_CRASHES = 3;
const crashes = [];
// checkpoints (IndexedDB): a new training worker restores the autosave of its env + algo
const checkpoints = setupPanel(toTraining);
let freshWorker = true; // the training (or single) worker just started
let keepStatusUntil = 0; // the crash notice is not cleared by the restarted worker before
const TRAP = /RuntimeError: (unreachable|memory access out of bounds|null function|indirect call)|Aborted\(|out of memory/i;

/** a new worker with `role`: "display" / "single" (rendering into a new canvas) or
 * "trainer" (no canvas) */
function startWorker(role) {
  const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  w.onmessage = (event) => onMessage(role, event);
  w.onerror = (e) => {
    e.preventDefault();
    if (TRAP.test(e.message)) restartAfterCrash(role, e.message);
    else status(`worker error: ${e.message}`, true);
  };
  if (role === "trainer" || !TWO_WORKERS) {
    freshWorker = true;
    requested.clear(); // (a new worker loads the packages again, from the package cache)
  }
  if (role === "trainer") {
    trainer = w;
    w.postMessage({ type: "init", canvas: null, base: location.href, role });
    return;
  }
  const fresh = Object.assign(document.createElement("canvas"), { id: "canvas" });
  canvas.replaceWith(fresh);
  canvas = fresh;
  display = w;
  const offscreen = canvas.transferControlToOffscreen();
  w.postMessage({ type: "init", canvas: offscreen, base: location.href, role }, [offscreen]);
  resize();
}

/** a worker crashed (`error`): restart it, unless it crashes too often */
function restartAfterCrash(role, error) {
  console.error(`python crashed (${role}):`, error);
  const training = role === "trainer" || !TWO_WORKERS;
  // a package that was loading / being checked: probably the cause, skipped from now on
  for (const id of training ? loading.keys() : []) {
    if (id in PACKAGES) {
      brokenPackages.add(id);
      showPackageResult(id, `crashed python while loading / checking (${error.split("\n")[0]}), skipped`);
      afterInstall.delete(id);
    }
  }
  loading.clear();
  showBusy();
  (role === "trainer" ? trainer : display).terminate();
  if (role === "trainer") trainer = null;
  else display = null;
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
  if (training) {
    stats = [];
    showHistory();
  }
  startWorker(role);
}

new ResizeObserver(resize).observe(stage);

// explanations: a click on a name with an explanation (data-help) shows it in a popup
// below; another click (anywhere) closes it
const helpPopup = Object.assign(document.createElement("div"), { id: "help-popup", hidden: true });
document.body.append(helpPopup);
document.addEventListener("click", (e) => {
  const target = e.target.closest?.("[data-help]");
  if (!target || helpPopup.dataset.for === target.dataset.help) {
    helpPopup.hidden = true;
    helpPopup.dataset.for = "";
    return;
  }
  e.preventDefault(); // (a click on a label would toggle its checkbox)
  const r = target.getBoundingClientRect();
  helpPopup.textContent = target.dataset.help;
  helpPopup.dataset.for = target.dataset.help;
  helpPopup.style.left = `${Math.max(8, r.left)}px`;
  helpPopup.style.top = `${r.bottom + 4}px`;
  helpPopup.hidden = false;
});

// the panel's sections (accordion): which are open is remembered in this browser
const SECTIONS = "webgym.sections";
try {
  const saved = JSON.parse(localStorage.getItem(SECTIONS) ?? "{}");
  for (const [id, open] of Object.entries(saved)) {
    const el = document.getElementById(id);
    if (el) el.open = open;
  }
} catch {
  // (no storage: the defaults)
}
for (const section of document.querySelectorAll("details.section")) {
  section.addEventListener("toggle", () => {
    try {
      const saved = JSON.parse(localStorage.getItem(SECTIONS) ?? "{}");
      localStorage.setItem(SECTIONS, JSON.stringify({ ...saved, [section.id]: section.open }));
    } catch {
      // (not remembered)
    }
    if (section.open) showHistory(); // (the plots get their size back)
  });
}

// the side panel's width: drag the resizer (remembered in this browser)
const SIDE_WIDTH = "webgym.sideWidth";
const setSideWidth = (px) => document.documentElement.style.setProperty("--side-width", `${px}px`);
try {
  const saved = Number(localStorage.getItem(SIDE_WIDTH));
  if (saved) setSideWidth(saved);
} catch {
  // (no storage: the default width)
}
const resizer = document.getElementById("resizer");
resizer.addEventListener("pointerdown", (e) => {
  resizer.setPointerCapture(e.pointerId);
  resizer.classList.add("dragging");
});
resizer.addEventListener("pointermove", (e) => {
  if (!resizer.hasPointerCapture(e.pointerId)) return;
  const width = Math.round(Math.min(Math.max(e.clientX, 260), window.innerWidth * 0.7));
  setSideWidth(width);
  try {
    localStorage.setItem(SIDE_WIDTH, String(width));
  } catch {
    // (not remembered)
  }
});
resizer.addEventListener("pointerup", () => resizer.classList.remove("dragging"));
startWorker(TWO_WORKERS ? "display" : "single");
if (TWO_WORKERS) startWorker("trainer");
// for test_page.py: a real wasm trap in the worker (checks the restart)
window.webgymCrashTest = () => toDisplay({ type: "command", name: "__crash__" });
// for experiments (web/compile_test.py): python in the training worker (with jax, once
// loaded) -> the value of the last expression (a string)
const execs = new Map();
window.webgymExec = (code) =>
  new Promise((resolve) => {
    const id = execs.size + Math.random();
    execs.set(id, resolve);
    toTraining({ type: "exec", id, code });
  });
// for the reproducers (web/repro_test.py): python in / an install into the compile worker
window.webgymCompilerExec = async (code) => (await compileWorker()).exec(code);
window.webgymCompilerInstall = async (packages, check) => (await compileWorker()).install(packages, check);
// a fresh compile worker (compiler options are global: per process)
window.webgymCompilerReset = async () => {
  if (compiler) (await compiler).terminate();
  compiler = null;
  return "compile worker reset";
};
// for compile experiments (web/compile_test.py): StableHLO -> WebGPU in the compile
// worker -> "ok <bytes>" or the error
window.webgymCompileTest = async (mlir) => {
  try {
    const { exec } = await compileWorker();
    const data = await exec(`from webgym import gpu\ngpu.compile_b64(${JSON.stringify(mlir)})`);
    return `ok ${Math.round((data.length * 3) / 4)} bytes`;
  } catch (e) {
    return `error ${String(e?.message ?? e).split("\n")[0]}`;
  }
};

// pointer input -> display worker (drawing-buffer px): drag bodies, pan, wheel zoom.
// The listeners are on the stage: they survive the canvas being replaced on a restart
const sendEvent = (kind, e, delta = 0) => toDisplay({ type: "event", kind, x: e.offsetX * scale, y: e.offsetY * scale, delta });
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
    if (!key || e.repeat || e.target.closest?.("input, select, textarea, [contenteditable]")) return;
    if (KEY_NAMES[e.key]) e.preventDefault(); // no scrolling with the arrows / space
    toDisplay({ type: "event", kind, key });
  });
}
window.addEventListener("blur", () => {
  for (const key of Object.values(KEY_NAMES)) toDisplay({ type: "event", kind: "keyup", key });
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
    const installs = []; // (web/repro_test.py) waiting for an extra install
    const install = (packages, check) =>
      new Promise((res) => {
        installs.push(res);
        w.postMessage({ type: "install", name: "extra", packages, check });
      });
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
        if (installs.length) installs.shift()(data.result);
        else resolve({ exec, install, terminate: () => w.terminate() });
      } else if (data.type === "result") pending.get(data.id).res(data.value);
      else if (data.type === "error" || data.type === "fatal") {
        for (const p of pending.values()) p.rej(new Error(data.error));
        pending.clear();
        for (const res of installs.splice(0)) res(`${data.type}: ${data.error}`);
        reject(new Error(data.error));
        // a crashed compiler (wasm trap) is unusable: the next request starts a new one
        if (data.type === "fatal") {
          w.terminate();
          compiler = null;
        }
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
    toTraining({ type: "compile_result", id, data });
  } catch (e) {
    toTraining({ type: "compile_result", id, error: String(e?.message ?? e) });
  }
}

/** load a package group in the worker once (installs run one after the other); not
 * one that crashed python (wasm trap) while it was loading / checked: no crash loop */
function loadPackage(name) {
  if (requested.has(name) || brokenPackages.has(name)) return;
  requested.add(name);
  toTraining({ type: "install", name, packages: PACKAGES[name].packages, check: PACKAGES[name].check });
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
