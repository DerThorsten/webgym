// The page (main thread): layout, progress bars, controls. Python runs in workers
// (worker.js):
// * the display worker: the simulation you watch / play, WebGL2 on an OffscreenCanvas
// * the training worker: trains as fast as it can, no drawing; snapshots of its policy
//   go to the display worker (replay in real time)
// * simulation helpers (CEM / ES): run shares of the trainer's population

import { compileRuntime as compileWasm32 } from "@derthorsten/pyjs/wasm32";
import { compileRuntime as compileWasm64 } from "@derthorsten/pyjs/wasm64";
import { pageArch } from "./arch.js";
import { setupPanel } from "./checkpoints.js";
import { createEditor, forgetFile, openFile, saveFile } from "./editor.js";
import { removeEdit, saveEdit, storedEdits } from "./edits.js";
import { drawNetwork } from "./network.js";
import { plot } from "./plot.js";
import { ask, startTour, tourDone, welcome } from "./tour.js";

const stage = document.getElementById("stage");
let canvas = document.getElementById("canvas"); // replaced on a worker restart
const params = new URLSearchParams(location.search);
// has this browser run webgym before (anything stored)? decided before the page stores its
// own settings: a first visit gets no release warning (checkRelease)
const VISITED = (() => {
  try {
    return Object.keys(localStorage).some((key) => key.startsWith("webgym"));
  } catch {
    return false;
  }
})();
const SMALL_DEVICE = matchMedia("(pointer: coarse)").matches || (navigator.hardwareConcurrency || 2) <= 4;
let display = null; // the display worker (role "display")
let trainer = null; // the training worker (role "trainer")
const toDisplay = (message, transfer) => display?.postMessage(message, transfer);
const toTraining = (message) => trainer?.postMessage(message);
let lastSnapshot = null; // the trainer's latest policy (for a display that (re)starts)
let autoTrain = params.get("train") !== "0"; // start training once the page is up
let trainerState = null; // the trainer's state message
// commands of the watched env only: to the display (and to the trainer too: its
// controls show these values as well)
const DISPLAY_COMMANDS = new Set(["auto_reset", "reset", "show_n", "replay_speed", "follow", "decorate",
  "toy", "rain", "toy_mode"]);

/** a widget changed: route the command to the worker(s) concerned */
function command(name, value) {
  const message = { type: "command", name, value };
  // another world (or its settings): it starts training at once, as on page load (its
  // autosave, if there is one, is restored first: that run continues)
  if (name === "env" || name.startsWith("setting.")) autoTrain = true;
  if (DISPLAY_COMMANDS.has(name) || name.startsWith("toy.") || name === "env" || name.startsWith("setting.") || name === "code") toDisplay(message);
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
  if (isError && text === statusEl.textContent) return; // (the same error again: shown already)
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

/** one progress bar per id (startup, pytorch): phase + bytes */
function showProgress({ id, phase, done, total }) {
  if (!bars.has(id)) {
    const el = document.createElement("div");
    el.className = "bar";
    el.innerHTML = `<label><span></span><span></span></label><progress></progress>`;
    // the page's own start at the top; everything in the background (packages) in the
    // "Packages & log" section
    const top = id === "startup";
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

const loading = new Map(); // id -> what it is doing (startup, pytorch)
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
  const edited = editedPaths.has(path);
  document.getElementById("code-path").textContent = path + (edited ? " · edited (kept in this browser)" : "");
  document.getElementById("code-revert").disabled = !edited;
  if (editor.get() !== text) editor.set(text);
}
codeKind.onchange = showCodeKind;
const fileName = () => (code[codeKind.value]?.path ?? "code.py").split("/").pop();
async function applyCode() {
  const { path } = code[codeKind.value] ?? {};
  const text = editor.get();
  log(`applying the edited ${codeKind.selectedOptions[0].text} code`);
  command("code", JSON.stringify({ kind: codeKind.value, text }));
  if (!path) return;
  // kept in this browser: every worker started from now on (a reload, a restart, the
  // simulation helpers) gets it; the running helpers are restarted with it
  await saveEdit(path, text, await version);
  editedPaths.add(path);
  showCodeKind();
  const n = wantedHelpers;
  if (n) {
    setHelpers(0);
    setHelpers(n);
  }
}
document.getElementById("code-apply").onclick = applyCode;
document.getElementById("code-revert").onclick = async () => {
  const { path } = code[codeKind.value] ?? {};
  if (!path || !confirm(`Back to the original ${fileName()}? Your edit is forgotten, the page reloads.`)) return;
  await removeEdit(path);
  location.reload();
};

// the edited files kept in this browser (edits.js), for this webgym version (the content
// hash of the sources archive): every worker gets them when it starts
const version = fetch(new URL("config.json", location.href), { cache: "no-cache" })
  .then((r) => r.json())
  .then((config) => config[pageArch(location.href)].archives[0].url)
  .catch(() => "");
let editedPaths = new Set();
let staleReported = false;
async function editedSources() {
  const { current, stale } = await storedEdits(await version);
  editedPaths = new Set(current.map((e) => e.path));
  if (stale.length && !staleReported) {
    staleReported = true;
    log(`not applied: your edits of ${stale.join(", ")} (made for another version of webgym)`);
  }
  return current;
}
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


/** a docstring's hard-wrapped lines as flowing text: paragraphs stay, and a line that
 * starts with a label ("obs:", "reward:") or a list item ("- ") starts a
 * new line */
const unwrap = (text) =>
  text
    .split(/\n\s*\n/)
    .map((paragraph) =>
      paragraph
        .split("\n")
        .map((line, i) => (i && /^(- |[A-Za-z][\w .()/-]*:)/.test(line) ? "\n" : i ? " " : "") + line.trim())
        .join(""),
    )
    .join("\n\n");

const openFolds = new Set(["Things to drop"]); // the folds that are open (kept on a rebuild)

/** the app's widgets (App.controls(): select / slider / button), rebuilt on change;
 * changes go back as commands */
// the app's controls of section "bar", as icons above the tabs: play / pause (train /
// stop; the title says which: Train / Continue training / Stop), new run, reset episode,
// the switches auto reset, camera follows, decoration (blue: on); below them the
// steppers show policies and replay speed
const quickTrain = document.getElementById("quick-train");
const SWITCHES = { auto_reset: "quick-reset", follow: "quick-follow", decorate: "quick-decorate" };
const PLAY = '<svg viewBox="0 0 24 24"><polygon points="6 3 20 12 6 21 6 3" /></svg>';
const PAUSE = '<svg viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></svg>';
quickTrain.innerHTML = PLAY;
document.getElementById("quick-new").onclick = () => command("new_run");
document.getElementById("quick-episode").onclick = () => command("reset");
function showQuick(controls) {
  const button = controls.find((c) => c.name === "train" || c.name === "stop");
  if (button) {
    const running = button.name === "stop";
    quickTrain.innerHTML = running ? PAUSE : PLAY;
    quickTrain.title = button.label;
    quickTrain.classList.toggle("running", running);
    quickTrain.onclick = () => command(button.name);
  }
  for (const [name, id] of Object.entries(SWITCHES)) {
    const c = controls.find((c) => c.name === name);
    if (!c) continue;
    const el = document.getElementById(id);
    el.classList.toggle("on", Boolean(c.value) && !c.disabled);
    el.disabled = Boolean(c.disabled); // (e.g. camera follows: a world with a fixed view)
    el.title = `${c.label}: ${c.value ? "on" : "off"}. ${c.help}`;
    el.onclick = () => command(name, !c.value);
  }
  // the steppers: − / + around the value (sliders of the app: min .. max, step; labels:
  // the value's text, e.g. "2x")
  for (const [name, id] of Object.entries(STEPPERS)) {
    const c = controls.find((c) => c.name === name);
    if (!c) continue;
    const el = document.getElementById(id);
    const text = c.labels ? c.labels[c.value] : String(c.value);
    el.querySelector(".value b").textContent = text;
    el.title = `${c.label}: ${text}. ${c.help}`;
    el.classList.toggle("disabled", Boolean(c.disabled));
    const step = (sign) => command(name, Math.min(c.max, Math.max(c.min, c.value + sign * c.step)));
    const less = el.querySelector(".less");
    const more = el.querySelector(".more");
    less.disabled = Boolean(c.disabled) || c.value <= c.min;
    more.disabled = Boolean(c.disabled) || c.value >= c.max;
    less.onclick = () => step(-1);
    more.onclick = () => step(1);
  }
}
const STEPPERS = { show_n: "quick-policies", replay_speed: "quick-speed" };

function showControls(controls) {
  showQuick(controls);
  // each control in its section ("environment", "training", "toys"; "bar": the icons
  // above the tabs, see showQuick), buttons at its end
  controls = controls.filter((c) => c.section !== "bar");
  const sections = new Map();
  for (const name of ["environment", "training", "toys"]) {
    const el = document.getElementById(`controls-${name}`);
    const buttons = document.createElement("div");
    el.replaceChildren(buttons);
    sections.set(name, { el, buttons });
  }
  const groups = new Map(); // group name -> a box with a title (e.g. the algorithm's parameters)
  for (const c of controls) {
    const { buttons } = sections.get(c.section ?? "environment");
    // (appended before the section's buttons)
    const root = { append: (...nodes) => buttons.before(...nodes) };
    let parent = root;
    if (c.group) {
      if (!groups.has(c.group)) {
        let box;
        if (c.fold) {
          // (a fold: opens / closes with a click on its title, e.g. the Toys tab's groups)
          box = Object.assign(document.createElement("details"), { className: "group", open: openFolds.has(c.group) });
          box.append(Object.assign(document.createElement("summary"), { className: "group-title", textContent: c.group }));
          box.ontoggle = () => (box.open ? openFolds.add(c.group) : openFolds.delete(c.group));
        } else {
          box = Object.assign(document.createElement("div"), { className: "group" });
          box.append(Object.assign(document.createElement("div"), { className: "group-title", textContent: c.group }));
        }
        root.append(box);
        groups.set(c.group, box);
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
    el.dataset.name = c.name;
    if (c.type === "hint") {
      root.append(Object.assign(document.createElement("div"), { className: c.active ? "hint active" : "hint", textContent: c.text }));
      continue;
    }
    if (c.type === "heading") {
      root.append(Object.assign(document.createElement("h4"), { textContent: c.text }));
      continue;
    }
    if (c.type === "tools") {
      // (the Toys tab) a row of buttons, the chosen one highlighted; a click on it again:
      // back to "grab" (the mouse drags bodies)
      el.classList.add("tools");
      for (const [value, text] of c.options) {
        const b = Object.assign(document.createElement("button"), { textContent: text });
        if (value === c.value) b.classList.add("active");
        b.onclick = () => {
          command(c.name, value === c.value ? "grab" : value);
          b.blur();
        };
        el.append(b);
      }
      parent.append(el);
      continue;
    }
    if (c.type === "info") {
      // (the env's description, in the World tab: its title line in bold, the rest below)
      const [first, ...rest] = c.text.split("\n");
      const info = Object.assign(document.createElement("div"), { className: "info" });
      info.append(Object.assign(document.createElement("b"), { textContent: first }), "\n", unwrap(rest.join("\n").trim()));
      root.append(info);
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
      el.className = c.inline ? "control checkbox inline" : "control checkbox"; // (inline: side by side)
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
      parent.append(el); // (in its group, if it has one)
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
        input.blur(); // (R resets again)
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
  worldKey = controls.find((c) => c.name === "env")?.key ?? "";
  showState();
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
  el.hidden = !(s.mode === "watch" && s.episode_over && s.scores.length && !s.toys);
  if (el.hidden) return;
  const score = s.scores.at(-1);
  el.innerHTML =
    `<div class="big">Score ${Math.round(score)}${s.new_best ? " · new best!" : ""}</div>` +
    `<div class="small">best ${Math.round(s.best_score)} · last ${s.scores.map(Math.round).join(", ")}</div>`;
}

/** the state panel: the watched env (display) + the training (trainer) */
function showState() {
  const s = displayState;
  if (!s) return;
  const last = stats.at(-1);
  const t = trainerState;
  const training = t?.mode === "train";
  stateEl.textContent =
    `training  ${training ? `running, iteration ${t.iteration}` : "stopped"} (training worker)\n` +
    `policy    ${s.policy}\n` +
    `episode   #${s.episode}, return ${s.episode_return.toFixed(0)}` +
    (s.episode_over ? (s.auto_reset ? " (over, restarting)" : " (over, keeps going)") : "") +
    "\n" +
    (last ? `mean/best ${last.mean_return.toFixed(0)} / ${last.best_return.toFixed(0)}\n` : "") +
    `frame     ${s.frame_ms.toFixed(1)} ms`;
  // the same, short, in the top bar
  const chip = (html, cls = "") => `<span class="chip ${cls}">${html}</span>`;
  document.getElementById("chips").innerHTML =
    (worldKey ? chip(`<b>${worldKey}</b>`, "world") : "") +
    (training ? chip(`training · iteration <b>${t.iteration}</b>`, "running") : chip("not training")) +
    (last ? chip(`mean <b>${last.mean_return.toFixed(0)}</b>`) + chip(`best <b>${last.best_return.toFixed(0)}</b>`, "best") : "") +
    (s.toys
      ? chip("watching: toys (no score)", "watch")
      : chip(`watching: episode ${s.episode}, return <b>${s.episode_return.toFixed(0)}</b>`, "watch"));
}
let displayState = null;
let worldKey = ""; // the env with its settings (e.g. "WaterCannon · objects 2"): a chip in the bar

/** a message from a worker: `role` "display" or "trainer" */
function onMessage(role, { data }) {
  const training = role === "trainer";
  if (data.type === "evaluate") {
    // the trainer's request for a simulation helper: part k goes to the k-th ready one
    readyHelpers()[data.part]?.postMessage(data);
    return;
  }
  if (data.type === "log") return log(data.text); // (a notice: only in the Log tab)
  if (data.type === "ready" && role === "trainer") {
    // the simulation helpers start once the trainer is up (not competing with the page's
    // start for the CPU); a restarted trainer: told how many are ready
    if (helpers.length < wantedHelpers) setHelpers(wantedHelpers);
    sendHelperCount();
  }
  if (data.type === "progress") {
    // the trainer's own start: its own bars (no hourglass: the page is usable)
    const id = role === "trainer" && (data.id === "startup" || data.id === "core") ? `training worker ${data.id}` : data.id;
    showProgress({ ...data, id });
  } else if (data.type === "ready") {
    if (role !== "trainer" && Date.now() > keepStatusUntil) status(""); // (a crash notice stays a while)
    if (role === "display" && lastSnapshot) toDisplay({ type: "snapshot", data: lastSnapshot });
    // the first visit: the welcome card (a tour, help or skip), once the page is up (not
    // in automated tests: they would click into it; ?tour=0 skips it too; ?welcome=1
    // shows it always)
    const firstVisit = !tourDone() && !navigator.webdriver && params.get("tour") !== "0";
    if (role !== "trainer" && (firstVisit || params.get("welcome") === "1")) setTimeout(welcome, 1000);
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
      if (net && !document.getElementById("tab-policy").hidden) drawNetwork(document.getElementById("network"), net);
    }
    showState();
  } else if (!training) {
    // the display worker: only its errors matter beyond this point
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
      const start = ++autoStart; // (a restore by the user meanwhile cancels it)
      restored.then(() => setTimeout(() => start === autoStart && command("train"), 300));
    }
  } else if (data.type === "checkpoint" || data.type === "delete_checkpoint") checkpoints.onMessage(data);
  else if (data.type === "code") showCode(data.code);
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
// (a checkpoint the user restores: a training about to start by itself doesn't)
let autoStart = 0;
const checkpoints = setupPanel((message) => {
  if (message.byUser) autoStart++;
  toTraining(message);
});
let freshWorker = true; // the training worker just started
let keepStatusUntil = 0; // the crash notice is not cleared by the restarted worker before
const TRAP = /RuntimeError: (unreachable|memory access out of bounds|null function|indirect call)|Aborted\(|out of memory/i;
// the pyjs runtime (wasm), fetched and compiled once here for all workers (also the
// restarted ones); undefined if that fails: then each worker compiles it itself
const ARCH = pageArch(location.href);
const runtime = (ARCH === "wasm64" ? compileWasm64 : compileWasm32)({ assetsUrl: new URL("pyjs/", location.href).href, cdnFallback: false })
  .catch((e) => console.warn("pyjs runtime: not compiled in the page", e));

/** a new worker with `role`: "display" (rendering into a new canvas) or "trainer" (no
 * canvas) */
function startWorker(role) {
  const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  w.onmessage = (event) => onMessage(role, event);
  w.onerror = (e) => {
    e.preventDefault();
    if (TRAP.test(e.message)) restartAfterCrash(role, e.message);
    else status(`worker error: ${e.message}`, true);
  };
  if (role === "trainer") {
    freshWorker = true;
    requested.clear(); // (a new worker loads the packages again, from the package cache)
    trainer = w;
    Promise.all([runtime, editedSources()]).then(([wasmModule, sources]) =>
      w.postMessage({ type: "init", canvas: null, base: location.href, role, wasmModule, sources }));
    return;
  }
  const fresh = Object.assign(document.createElement("canvas"), { id: "canvas" });
  canvas.replaceWith(fresh);
  canvas = fresh;
  display = w;
  const offscreen = canvas.transferControlToOffscreen();
  Promise.all([runtime, editedSources()]).then(([wasmModule, sources]) => {
    w.postMessage({ type: "init", canvas: offscreen, base: location.href, role, wasmModule, sources }, [offscreen]);
    resize(); // (after the init: before it, the worker has no canvas to resize)
  });
}

/** a worker crashed (`error`): restart it, unless it crashes too often */
function restartAfterCrash(role, error) {
  console.error(`python crashed (${role}):`, error);
  const training = role === "trainer";
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
  helpPopup.style.left = `${Math.max(8, Math.min(r.left, innerWidth - 330))}px`; // (at most 300 px + padding wide)
  helpPopup.style.top = `${r.bottom + 4}px`;
  helpPopup.hidden = false;
});

/** a remembered choice of this browser (localStorage; none: `fallback`) */
function remembered(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback; // (no storage)
  }
}
function remember(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // (not remembered)
  }
}

// the panel's tabs (World / Training / Policy / Log): the chosen one is remembered
const TAB = "webgym.tab";
function showTab(name) {
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("active", button.dataset.tab === name);
    document.getElementById(`tab-${button.dataset.tab}`).hidden = button.dataset.tab !== name;
  }
  remember(TAB, name);
}
for (const button of document.querySelectorAll("#tabs button")) {
  button.onclick = () => {
    showTab(button.dataset.tab);
    // (the Toys tab: playing with the world, so no auto reset, one policy in view, no
    // scores; another tab: scores again)
    command("toy_mode", button.dataset.tab === "toys");
  };
}
showTab(document.getElementById(`tab-${remembered(TAB, "world")}`) ? remembered(TAB, "world") : "world");

// the training progress under the world: open / closed (remembered; small devices:
// closed at first)
const PLOT = "webgym.plot";
const plotToggle = document.getElementById("plot-toggle");
function showPlots(open) {
  document.getElementById("plots").hidden = !open;
  plotToggle.textContent = `${open ? "▾" : "▸"} Training progress`;
  remember(PLOT, open ? "open" : "closed");
  if (open) showHistory(); // (the plots get their size back)
}
plotToggle.onclick = () => showPlots(document.getElementById("plots").hidden);
showPlots(remembered(PLOT, SMALL_DEVICE ? "closed" : "open") === "open");

document.getElementById("tour-link").onclick = (e) => {
  e.preventDefault();
  startTour("quick");
};
document.getElementById("tour-detailed-link").onclick = (e) => {
  e.preventDefault();
  startTour("detailed");
};

// the right panel (python editor): closed by default, the button opens / closes it
const codeToggle = document.getElementById("code-toggle");
codeToggle.addEventListener("click", () => {
  const panel = document.getElementById("code-panel");
  panel.hidden = !panel.hidden;
  document.getElementById("code-resizer").hidden = panel.hidden;
  codeToggle.classList.toggle("open", !panel.hidden);
});
document.getElementById("code-close").onclick = () => codeToggle.click(); // (phones: full screen)
// phones: the ⋯ menu (Tour, Help, ...) and, in landscape, the panel as a drawer (index.html)
const bar = document.getElementById("bar");
document.getElementById("menu-toggle").onclick = () => bar.classList.toggle("menu-open");
document.getElementById("links").addEventListener("click", () => bar.classList.remove("menu-open"));
document.getElementById("panel-toggle").onclick = () => document.getElementById("app").classList.toggle("drawer-open");

/** drag `resizer` to set the CSS width variable `name` (px; remembered in this browser);
 * width(clientX) -> the panel's width at that pointer position */
function resizable(resizer, name, width) {
  const set = (px) => document.documentElement.style.setProperty(name, `${px}px`);
  try {
    const saved = Number(localStorage.getItem(`webgym${name}`));
    if (saved) set(saved);
  } catch {
    // (no storage: the default width)
  }
  resizer.addEventListener("pointerdown", (e) => {
    resizer.setPointerCapture(e.pointerId);
    resizer.classList.add("dragging");
  });
  resizer.addEventListener("pointermove", (e) => {
    if (!resizer.hasPointerCapture(e.pointerId)) return;
    const px = Math.round(Math.min(Math.max(width(e.clientX), 260), window.innerWidth * 0.8));
    set(px);
    try {
      localStorage.setItem(`webgym${name}`, String(px));
    } catch {
      // (not remembered)
    }
  });
  resizer.addEventListener("pointerup", () => resizer.classList.remove("dragging"));
}
resizable(document.getElementById("resizer"), "--side-width", (x) => x); // (left panel)
resizable(document.getElementById("code-resizer"), "--code-width", (x) => window.innerWidth - x); // (editor)
// ---- simulation helpers: more workers that run episodes for the trainer -------------
// (CEM / ES: each runs a share of the population in its own python + box2d, on another
// CPU core; webgym/helpers.py). The slider: how many (remembered in this browser).
const HELPERS = "webgym.helpers";
const MAX_HELPERS = Math.max(0, (navigator.hardwareConcurrency || 2) - 2); // (the page + display + trainer)
const helpers = []; // {worker, ready}
let wantedHelpers = 0; // (the slider; started once the trainer is ready)
const readyHelpers = () => helpers.filter((h) => h.ready).map((h) => h.worker);
/** the trainer splits its population over the ready helpers */
const sendHelperCount = () => trainer?.postMessage({ type: "command", name: "helpers", value: readyHelpers().length });

function startHelper() {
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const helper = { worker, ready: false };
  helpers.push(helper);
  worker.onmessage = ({ data }) => {
    if (data.type === "ready") {
      helper.ready = true;
      sendHelperCount();
    } else if (data.type === "evaluated") trainer?.postMessage(data);
    else if (data.type === "error") status(`error (simulation worker): ${data.error}`, true);
    else if (data.type === "fatal") {
      console.error("a simulation worker crashed:", data.error);
      stopHelper(helper); // (the trainer goes on without it)
    }
  };
  Promise.all([runtime, editedSources()]).then(([wasmModule, sources]) =>
    worker.postMessage({ type: "init", canvas: null, base: location.href, role: "helper", wasmModule, sources }));
}

function stopHelper(helper) {
  helper.worker.terminate();
  helpers.splice(helpers.indexOf(helper), 1);
  if (helper.ready) sendHelperCount();
}

/** run `n` simulation helpers */
function setHelpers(n) {
  wantedHelpers = n;
  while (helpers.length < n) startHelper();
  while (helpers.length > n) stopHelper(helpers.at(-1));
  document.getElementById("helpers-value").textContent = String(n);
}

// The release (pyproject.toml's version, in config.json): a browser that ran an older one
// has saved data that may not fit any more (runs, checkpoints, code edits, cached
// packages, settings). It is asked (a warning) and then deleted; "Cancel" keeps it and
// asks again next time. A first visit has nothing to delete. Only webgym's own storage:
// other pages of the same site keep theirs.
const RELEASE = "webgym.release";

async function checkRelease() {
  try {
    const config = await (await fetch(new URL("config.json", location.href), { cache: "no-cache" })).json();
    const release = config[pageArch(location.href)].release;
    const seen = localStorage.getItem(RELEASE);
    if (!release || seen === release) return;
    // (only a browser with work to lose is asked: runs, checkpoints, code edits; old
    // settings and cached packages go without asking; a first visit has neither)
    if (VISITED && !(await savedWork())) await forgetEverything();
    else if (VISITED) {
      const ok = await ask(
        `<h2>webgym was updated</h2>
        <p>${seen ?? "An older version"} → <b>${release}</b>. What this browser saved for the old
          version may not fit any more: your training runs and checkpoints, your code edits,
          the cached packages and the page's settings.</p>
        <p><b>Delete and continue</b> starts fresh (the packages are downloaded again, once).
          <b>Keep for now</b> keeps them (old runs may fail to load); you'll be asked again next
          time.</p>`,
        [[false, "Keep for now"], [true, "Delete and continue"]],
      );
      if (!ok) return;
      await forgetEverything();
    }
    localStorage.setItem(RELEASE, release);
  } catch (error) {
    console.warn("release check:", error); // (no storage, no config: go on)
  }
}

/** does this browser hold webgym work: runs and checkpoints (IndexedDB "webgym"), code
 * edits ("webgym-code")? (the page opens them on every visit: their records count; an
 * error: assume so) */
async function savedWork() {
  const count = (name, store) =>
    new Promise((resolve) => {
      const request = indexedDB.open(name);
      request.onerror = () => resolve(1);
      request.onsuccess = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(store)) return db.close(), resolve(0);
        const counting = db.transaction(store).objectStore(store).count();
        counting.onsuccess = () => (db.close(), resolve(counting.result));
        counting.onerror = () => (db.close(), resolve(1));
      };
    });
  try {
    const names = (await indexedDB.databases()).map((d) => d.name); // (open() would create them)
    const runs = names.includes("webgym") ? await count("webgym", "checkpoints") : 0;
    const edits = names.includes("webgym-code") ? await count("webgym-code", "files") : 0;
    return runs + edits > 0;
  } catch {
    return true;
  }
}

/** delete what webgym stored in this browser (only webgym's) */
async function forgetEverything() {
  for (const name of ["webgym", "webgym-code"]) {
    await new Promise((done) => {
      const request = indexedDB.deleteDatabase(name);
      request.onsuccess = request.onerror = request.onblocked = () => done();
    });
  }
  if (self.caches) for (const key of await caches.keys()) if (key.startsWith("webgym")) await caches.delete(key);
  for (const key of Object.keys(localStorage)) if (key.startsWith("webgym")) localStorage.removeItem(key);
}

checkRelease().then(() => {
  startWorker("display");
  startWorker("trainer");
});
if (MAX_HELPERS > 0) {
  const slider = document.getElementById("helpers");
  let saved = null;
  try {
    saved = localStorage.getItem(HELPERS);
  } catch {
    // (no storage: the default)
  }
  slider.max = String(MAX_HELPERS);
  slider.value = params.get("helpers") ?? saved ?? String(Math.min(MAX_HELPERS, Math.floor(MAX_HELPERS / 2)));
  slider.oninput = () => {
    setHelpers(Number(slider.value));
    try {
      localStorage.setItem(HELPERS, slider.value);
    } catch {
      // (not remembered)
    }
  };
  document.getElementById("helpers-control").hidden = false;
  wantedHelpers = Number(slider.value); // (started when the trainer is ready)
  document.getElementById("helpers-value").textContent = slider.value;
}
// for test_page.py: a real wasm trap in the worker (checks the restart)
window.webgymCrashTest = () => toDisplay({ type: "command", name: "__crash__" });

// pointer input -> display worker (drawing-buffer px). The first pointer is the mouse:
// drag a body, drag the background to pan, the wheel zooms. A second finger starts a
// pinch: the first one's drag ends, the two zoom (about their midpoint) and pan, as
// "pan" / "move" / "wheel" events. Further fingers, and the one left after a pinch,
// are ignored until they lift. The listeners are on the stage: they survive the canvas
// being replaced on a restart
const send = (kind, x, y, delta = 0) => toDisplay({ type: "event", kind, x: x * scale, y: y * scale, delta });
const pointers = new Map(); // pressed pointer id -> {x, y} (canvas CSS px)
let primary = null; // the pointer that acts as the mouse
let pinch = null; // the two pinching pointer ids
const pinchAt = () => {
  const [a, b] = pinch.map((id) => pointers.get(id));
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) };
};
stage.style.touchAction = "none";
stage.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || e.target !== canvas) return;
  canvas.setPointerCapture(e.pointerId);
  if (e.isPrimary) {
    pointers.clear(); // (a new gesture: nothing left over from the last one)
    primary = pinch = null;
  }
  pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
  if (pointers.size === 1) {
    primary = e.pointerId;
    send("down", e.offsetX, e.offsetY);
  } else if (pointers.size === 2 && primary !== null && e.pointerType !== "mouse") {
    pinch = [primary, e.pointerId];
    primary = null;
    send("leave", 0, 0);
    const m = pinchAt();
    send("pan", m.x, m.y);
  }
});
stage.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const before = pinch?.includes(e.pointerId) && pinchAt();
  pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
  if (before) {
    const m = pinchAt();
    send("move", m.x, m.y); // (pan: the grabbed point follows the midpoint)
    if (before.d > 1 && m.d > 1) send("wheel", m.x, m.y, Math.log(before.d / m.d) / Math.log(1.0015)); // (App.event's wheel zoom)
  } else if (e.pointerId === primary) send("move", e.offsetX, e.offsetY);
});
const lift = (e) => {
  if (!pointers.delete(e.pointerId)) return;
  if (e.pointerId === primary || pinch?.includes(e.pointerId)) send("up", e.offsetX, e.offsetY);
  if (e.pointerId === primary) primary = null;
  if (pinch?.includes(e.pointerId)) pinch = null;
};
stage.addEventListener("pointerup", lift);
stage.addEventListener("pointercancel", lift);
// R: a new episode (unless the user types into a widget)
window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() !== "r" || e.repeat || e.target.closest?.("input, select, textarea, [contenteditable]")) return;
  toDisplay({ type: "event", kind: "keydown", key: "r" });
});
stage.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    send("wheel", e.offsetX, e.offsetY, e.deltaMode === 1 ? 16 * e.deltaY : e.deltaY);
  },
  { passive: false },
);

// The big package groups (pinned in config.json by build_page.py), loaded in the
// background after the start (from pyjs' package cache after the first visit); each
// is checked once loaded (the result shows in the packages panel)
const PACKAGES = {
  pytorch: `
import time, torch
torch.autograd.set_multithreading_enabled(False)
t = time.perf_counter()
loss = torch.nn.Linear(8, 2)(torch.randn(64, 8)).pow(2).mean()
loss.backward()
f"torch {torch.__version__}, fwd+bwd {1e3 * (time.perf_counter() - t):.1f} ms"
`,
};

/** load a package group in the worker once (installs run one after the other); not
 * one that crashed python (wasm trap) while it was loading / checked: no crash loop */
function loadPackage(name) {
  if (requested.has(name) || brokenPackages.has(name)) return;
  requested.add(name);
  toTraining({ type: "install", name, check: PACKAGES[name] });
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
