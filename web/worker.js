// The pyjs web worker: python, the simulation, training and WebGL2 rendering on the
// OffscreenCanvas the page transferred. Talks to main.js with messages:
//   page -> worker: {type: "init", canvas, base, role, wasmModule, sources} (wasmModule:
//                   the pyjs runtime compiled once by the page; optional; sources: the
//                   python files edited in this browser, [{path, text}], see edits.js),
//                   {type: "resize", width, height},
//                   {type: "snapshot", data} (display: the trainer's policy, base64),
//                   {type: "event", kind, x, y, delta, key} (pointer: drawing-buffer px; keys),
//                   {type: "command", name, value} (a widget changed),
//                   {type: "restore", data} (a checkpoint, base64 .safetensors),
//                   {type: "exec", id, code} (python; -> {type: "result", id, value}),
//                   {type: "install", name, check} (a package group of config.json, e.g. pytorch),
//                   {type: "evaluate", ...} (helper: a chunk of the population, helpers.py),
//                   {type: "evaluated", id, part, returns} (trainer: a helper's answer)
//   worker -> page: {type: "progress", id, phase, done, total},
//                   {type: "controls", controls}, {type: "history", entries, reset},
//                   {type: "state", state}, {type: "installed", name, seconds}, {type: "error", error},
//                   {type: "log", text} (a notice: only in the log),
//                   {type: "fatal", error} (wasm trap: the page restarts the worker),
//                   {type: "checkpoint", name, env, algo, iteration, mean_return, data},
//                   {type: "delete_checkpoint", name, env, algo},
//                   {type: "snapshot", data} (trainer: its current policy, for the display)
//
// Packages: pinned at build time (config.json, see build_page.py: no solver in the
// browser); pyjs caches them (installed layout + bytecode, Cache Storage) for the next visit.

import { initializePyjsRuntime as initWasm32 } from "@derthorsten/pyjs/wasm32";
import { initializePyjsRuntime as initWasm64 } from "@derthorsten/pyjs/wasm64";
import { pageArch } from "./arch.js";

const post = (message) => self.postMessage(message);
const progress = (id, phase, done = 0, total = 0) => post({ type: "progress", id, phase, done, total });
/** pyjs' progress ({item, phase, done, total}) as one of our progress bars */
const reportTo = (id) => (p) => progress(id, `${p.item}: ${p.phase}`, p.done, p.total);

let pyjs = null;
let config = null; // config.json of this architecture: {core, pytorch, archives}
const times = {};
const t0 = performance.now();
const lap = (name) => (times[name] = +((performance.now() - t0) / 1000).toFixed(2));

// (python: writes self.webgymSources ([{path, text}]) over the files; -> the skipped ones)
const EDITED_SOURCES = `
import json, pyjs
_skipped = []
for _source in json.loads(str(pyjs.js.self.webgymSources)):
    try:
        compile(_source["text"], _source["path"], "exec")
        with open(_source["path"], "w") as _f:
            _f.write(_source["text"])
    except (SyntaxError, OSError) as _e:
        _skipped.append(f"{_source['path']}: {_e}")
"; ".join(_skipped)
`;

/** start python with the core packages and webgym's sources, then the host (rendering
 * into `canvas`); `base`: the page's url (config.json, the archives and pyjs/ live there) */
async function init(canvas, base, role, wasmModule, sources = []) {
  self.canvas = canvas; // host_web.py reads it via pyjs.js.canvas (null: no drawing)
  // the URL query parameters (?env=LunarLander&algo=cem...) + the role ("display",
  // "trainer", "helper") for host_web.start
  self.webgymOptions = JSON.stringify({ ...Object.fromEntries(new URL(base).searchParams), role });
  progress("startup", "starting python");
  const arch = pageArch(base);
  times.arch = arch;
  config = (await (await fetch(new URL("config.json", base), { cache: "no-cache" })).json())[arch];
  pyjs = await (arch === "wasm64" ? initWasm64 : initWasm32)({
    assetsUrl: new URL("pyjs/", base).href,
    cdnFallback: false,
    // (our own Cache Storage names, "webgym-packages(-bytecode)": a new release of the
    // page can delete them without touching other pages of the same site, see main.js)
    cache: "webgym-packages",
    wasmModule, // (compiled by the page once for all workers; none: compiled here)
    urls: config.core, // numpy, pyopengl, pyb2d3 (pinned)
    // (cache: the archive's name has its content hash, so pyjs caches it like a package:
    // unpacked files + bytecode, no download on later visits)
    archives: config.archives.map((a) => ({ url: new URL(a.url, base).href, path: a.path, cache: true })),
    onProgress: reportTo("startup"),
  });
  lap("python");
  await pyjs.runPythonAsync('import sys, importlib\nsys.path.insert(0, "/app")\nimportlib.invalidate_caches()');
  // the python files edited in this browser (edits.js), over the originals, before
  // webgym is imported; one that doesn't compile is skipped (and reported)
  if (sources.length) {
    self.webgymSources = JSON.stringify(sources);
    const skipped = await pyjs.runPythonAsync(EDITED_SOURCES);
    if (String(skipped)) post({ type: "error", error: `edited code not applied (it doesn't compile): ${skipped}` });
  }
  // no role: a compute-only worker (benchmarks)
  if (role) await pyjs.runPythonAsync("from webgym import host_web\n_host = host_web.start()");
  // keep the bytecode of the startup imports (numpy, pyopengl, pyb2d3, webgym) for the
  // next visit: in a worker pyjs saves it only when asked (>= 5.2)
  await pyjs.saveBytecode();
  lap("running");
  progress("startup", "done");
  console.info("webgym loading times (s since start):", times, "pyjs:", pyjs.initTimings);
}

/** install a package group of config.json (e.g. "pytorch"), then run `check` (python) */
async function install(name, check) {
  const start = performance.now();
  progress(name, "downloading");
  await pyjs.installPackages(config[name], { onProgress: reportTo(name) });
  const downloaded = (performance.now() - start) / 1000;
  progress(name, "importing");
  let result;
  try {
    result = await pyjs.runPythonAsync(check);
    await pyjs.saveBytecode(); // (the next visit imports it without compiling the sources)
  } catch (e) {
    // a failed check (python exception): reported as the result, the page goes on
    result = `failed: ${String(e?.message ?? e).trim().split("\n").at(-1)}`;
  }
  progress(name, "ready");
  post({ type: "installed", name, seconds: { install: downloaded, total: (performance.now() - start) / 1000 }, result: String(result) });
}

// wasm traps / aborts: the runtime's memory is in an unknown state afterwards, the
// page restarts the worker (python exceptions are reported by host_web.py and the
// app goes on)
// (the JS RuntimeError of a wasm trap, not a python RuntimeError(...))
const FATAL = /RuntimeError: (unreachable|memory access out of bounds|null function|indirect call|divide by zero|integer overflow|call stack)|Aborted\(|out of memory/i;
const report = (e) => {
  const text = String(e?.stack ?? e?.message ?? e);
  post({ type: FATAL.test(text) ? "fatal" : "error", error: text });
};
self.addEventListener("error", (e) => {
  e.preventDefault();
  report(e.error ?? e.message);
});
self.addEventListener("unhandledrejection", (e) => {
  e.preventDefault();
  report(e.reason);
});

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init(data.canvas, data.base, data.role, data.wasmModule, data.sources);
    else if (data.type === "snapshot") self.webgymSnapshot?.(data.data);
    else if (data.type === "event") self.webgymEvent?.(data.kind, data.x ?? 0, data.y ?? 0, data.delta ?? 0, data.key ?? "");
    else if (data.type === "command") self.webgymCommand?.(data.name, data.value ?? null);
    else if (data.type === "restore") self.webgymRestore?.(data.data);
    // simulation helpers: a chunk of the population to run (helper) / its returns (trainer)
    else if (data.type === "evaluate") self.webgymEvaluate?.(JSON.stringify(data));
    else if (data.type === "evaluated") self.webgymEvaluated?.(JSON.stringify(data));
    else if (data.type === "resize") {
      if (!self.canvas) return; // (not initialized: the init message sets the size anyway)
      self.canvas.width = data.width;
      self.canvas.height = data.height;
    } else if (data.type === "install") await install(data.name, data.check);
    else if (data.type === "exec") {
      // compute-only workers: run python, the value of the last expression comes back
      const value = await pyjs.runPythonAsync(data.code);
      post({ type: "result", id: data.id, value: String(value), memory: pyjs.HEAPU8?.byteLength });
    }
  } catch (e) {
    console.error(e);
    report(e);
  }
};
