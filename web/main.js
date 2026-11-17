// The page (main thread): layout, progress bars, controls. Python, the simulation and
// the WebGL2 rendering run in worker.js on an OffscreenCanvas.

import { plot } from "./plot.js";

const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const stateEl = document.getElementById("state");
const plotEl = document.getElementById("plot");
let history = []; // training stats per iteration (from the worker)
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
}

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

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
  worker.postMessage({ type: "resize", width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)) });
}

/** the app's widgets (App.controls(): select / slider / button), rebuilt on change;
 * changes go back as commands */
function showControls(controls) {
  const root = document.getElementById("controls");
  root.replaceChildren();
  const send = (name, value) => worker.postMessage({ type: "command", name, value });
  const buttons = document.createElement("div");
  for (const c of controls) {
    if (c.type === "button") {
      const b = document.createElement("button");
      b.textContent = c.label;
      b.className = c.name === "train" ? "primary" : "";
      b.onclick = () => {
        send(c.name);
        b.blur(); // space would press it again
      };
      buttons.append(b);
      continue;
    }
    const el = document.createElement("div");
    el.className = "control";
    if (c.type === "info") {
      el.className = "info";
      el.textContent = c.text;
      root.append(el);
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
        send(c.name, input.checked);
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
      for (const o of c.options) input.add(new Option(o, o, false, o === c.value));
      input.onchange = () => {
        send(c.name, input.value);
        input.blur(); // the arrow keys go to the env again
      };
    } else {
      input = Object.assign(document.createElement("input"), { type: "range", min: c.min, max: c.max, step: c.step, value: c.value });
      value.textContent = c.display ?? c.value;
      input.oninput = () => (value.textContent = c.display === undefined ? input.value : "");
      input.onchange = () => {
        send(c.name, Number(input.value));
        input.blur();
      };
    }
    el.append(label, input);
    root.append(el);
  }
  root.append(buttons);
}

/** the reward curves of the current run */
function showHistory() {
  plot(plotEl, [
    { label: "mean return", color: "#4fa3ff", values: history.map((h) => h.mean_return) },
    { label: "best return", color: "#f5a742", values: history.map((h) => h.best_return) },
  ]);
}
new ResizeObserver(showHistory).observe(plotEl);

worker.onmessage = ({ data }) => {
  if (data.type === "progress") showProgress(data);
  else if (data.type === "ready") status("");
  else if (data.type === "controls") showControls(data.controls);
  else if (data.type === "history") {
    history = data.reset ? data.entries : history.concat(data.entries);
    showHistory();
  } else if (data.type === "state") {
    const s = data.state;
    const last = history.at(-1);
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
    status(`${data.name}: ${data.result} (${data.seconds.total.toFixed(1)} s)`);
  } else if (data.type === "error") status(`error: ${data.error}`, true);
};
worker.onerror = (e) => status(`worker error: ${e.message}`, true);

const offscreen = canvas.transferControlToOffscreen();
worker.postMessage({ type: "init", canvas: offscreen, base: document.baseURI }, [offscreen]);
new ResizeObserver(resize).observe(canvas);
resize();

// pointer input -> worker (drawing-buffer px): drag bodies, pan, wheel zoom
const sendEvent = (kind, e, delta = 0) =>
  worker.postMessage({ type: "event", kind, x: e.offsetX * scale, y: e.offsetY * scale, delta });
canvas.style.touchAction = "none";
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  canvas.setPointerCapture(e.pointerId);
  sendEvent("down", e);
});
canvas.addEventListener("pointermove", (e) => e.buttons & 1 && sendEvent("move", e));
canvas.addEventListener("pointerup", (e) => sendEvent("up", e));
canvas.addEventListener("pointercancel", (e) => sendEvent("up", e));
canvas.addEventListener("pointerleave", (e) => sendEvent("leave", e));
// keys -> worker (the human controller), unless the user types into a widget
const KEY_NAMES = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down", " ": "space" };
const keyName = (e) => KEY_NAMES[e.key] ?? (e.key.length === 1 ? e.key.toLowerCase() : null);
for (const kind of ["keydown", "keyup"]) {
  window.addEventListener(kind, (e) => {
    const key = keyName(e);
    if (!key || e.repeat || e.target.closest?.("input, select, textarea")) return;
    if (KEY_NAMES[e.key]) e.preventDefault(); // no scrolling with the arrows / space
    worker.postMessage({ type: "event", kind, key });
  });
}
window.addEventListener("blur", () => {
  for (const key of Object.values(KEY_NAMES)) worker.postMessage({ type: "event", kind: "keyup", key });
});
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    sendEvent("wheel", e, e.deltaMode === 1 ? 16 * e.deltaY : e.deltaY);
  },
  { passive: false },
);

// milestone 0: load the big packages from the channel and run a tiny forward/backward
const CHECKS = {
  pytorch: `
import time, torch
torch.autograd.set_multithreading_enabled(False)
t = time.perf_counter()
m = torch.nn.Linear(8, 2)
loss = m(torch.randn(64, 8)).pow(2).mean()
loss.backward()
f"torch {torch.__version__}, fwd+bwd {1e3 * (time.perf_counter() - t):.1f} ms"
`,
  jax: `
import time, jax, jax.numpy as jnp
t = time.perf_counter()
g = jax.jit(jax.grad(lambda w: (w ** 2).sum()))
g(jnp.ones(8)).block_until_ready()
f"jax {jax.__version__} on {jax.devices()}, jit+grad {1e3 * (time.perf_counter() - t):.1f} ms"
`,
};
for (const name of ["pytorch", "jax"]) {
  const button = document.getElementById(name === "pytorch" ? "load-torch" : "load-jax");
  button.addEventListener("click", () => {
    button.disabled = true;
    worker.postMessage({ type: "install", name, packages: [name], check: CHECKS[name] });
  });
}
