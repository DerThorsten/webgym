// The pyjs web worker: python, the simulation, training and WebGL2 rendering on the
// OffscreenCanvas the page transferred. Talks to main.js with messages:
//   page -> worker: {type: "init", canvas, base}, {type: "resize", width, height},
//                   {type: "event", kind, x, y, delta, key} (pointer: drawing-buffer px; keys),
//                   {type: "command", name, value} (a widget changed),
//                   {type: "restore", data} (a checkpoint, base64 .safetensors),
//                   {type: "exec", id, code} (python; -> {type: "result", id, value}),
//                   {type: "install", name, packages}
//   worker -> page: {type: "progress", id, phase, done, total}, {type: "ready"},
//                   {type: "controls", controls}, {type: "history", entries, reset},
//                   {type: "state", state}, {type: "installed", name, seconds}, {type: "error", error},
//                   {type: "fatal", error} (wasm trap: the page restarts the worker),
//                   {type: "checkpoint", name, env, algo, iteration, mean_return, data},
//                   {type: "delete_checkpoint", name, env, algo}
// (archive download / unpack: from pyb2d3/examples/web/main.js)

import { initializePyjsRuntime } from "@derthorsten/pyjs";

const post = (message) => self.postMessage(message);

// conda packages (pytorch, jax, ... from the channels) are immutable files: cache-first
// in Cache Storage, so a second visit loads them without the network. Everything else
// (repodata.json, ...) goes to the network as usual.
const PACKAGE_CACHE = "webgym-packages-v1";
const isPackage = (url) => /\.(tar\.bz2|conda)(\?|$)/.test(url);
const networkFetch = self.fetch.bind(self);
self.fetch = async (input, init) => {
  const url = typeof input === "string" ? input : input.url ?? String(input);
  if (!isPackage(url) || typeof caches === "undefined") return networkFetch(input, init);
  try {
    const cache = await caches.open(PACKAGE_CACHE);
    const hit = await cache.match(url);
    if (hit) return hit;
    const response = await networkFetch(input, init);
    if (response.ok) await cache.put(url, response.clone());
    return response;
  } catch (e) {
    console.warn("package cache:", e);
    return networkFetch(input, init);
  }
};
const progress = (id, phase, done = 0, total = 0) => post({ type: "progress", id, phase, done, total });

/** the bytes at `url` as a Blob, reporting bytes received */
async function download(url, onBytes) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  // a dev server answers a missing file with index.html (200)
  if ((response.headers.get("content-type") ?? "").includes("text/html"))
    throw new Error(`${url} is missing (an HTML page came instead): the archives were rebuilt, reload the page`);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(received);
  }
  return new Blob(chunks);
}

/** download an archive of config.json and unpack it into the file system */
async function unpackArchive(pyjs, archive) {
  const url = new URL(archive.url, base).href;
  const blob = await download(url, (n) => progress(archive.name, "downloading", n, archive.size));
  progress(archive.name, "unpacking");
  // pyjs picks the decompression by the url's ending: keep it after the blob url
  const blobUrl = URL.createObjectURL(blob);
  await pyjs.unpackArchive({ url: `${blobUrl}#${archive.url}`, path: archive.path });
  URL.revokeObjectURL(blobUrl);
  progress(archive.name, "done");
}

let pyjs = null;
// the page's url: config.json, the archives and pyjs/ live there (the bundled
// worker itself is in assets/)
let base = self.location.href;
const times = {};
const t0 = performance.now();
const lap = (name) => (times[name] = +((performance.now() - t0) / 1000).toFixed(2));

/** start python, unpack the core, start the host (rendering into `canvas`) */
async function init(canvas, pageUrl) {
  self.canvas = canvas; // host_web.py reads it via pyjs.js.canvas (null: compute only)
  base = pageUrl;
  // the URL query parameters (?env=LunarLander&algo=cem...) for host_web.start
  self.webgymOptions = JSON.stringify(Object.fromEntries(new URL(pageUrl).searchParams));
  progress("startup", "starting python");
  const config = await (await fetch(new URL("config.json", base), { cache: "no-cache" })).json();
  pyjs = await initializePyjsRuntime({
    assetsUrl: new URL("pyjs/", base).href,
    cdnFallback: false,
  });
  lap("python");
  for (const archive of config.archives) await unpackArchive(pyjs, archive);
  lap("archives");
  progress("startup", "importing");
  await pyjs.async_exec_eval(`
import sys, importlib
sys.path.insert(0, "/app")
sys.dont_write_bytecode = True  # code edited in the page is re-imported
importlib.invalidate_caches()
`);
  // without a canvas: a compute-only worker (benchmarks, later a training pool)
  if (canvas) await pyjs.async_exec_eval("from webgym import host_web\n_host = host_web.start()");
  lap("running");
  progress("startup", "done");
  console.info("webgym loading times (s since start):", times);
}

/** install conda packages from the channel (e.g. pytorch), then run `check` (python) */
async function install(name, packages, check) {
  const start = performance.now();
  progress(name, "downloading");
  await pyjs.installPackages(packages, { logger: { log: (m) => progress(name, String(m)), warn: console.warn, error: console.error } });
  const downloaded = (performance.now() - start) / 1000;
  progress(name, "importing");
  const result = await pyjs.async_exec_eval(check);
  progress(name, "ready");
  post({ type: "installed", name, seconds: { install: downloaded, total: (performance.now() - start) / 1000 }, result: String(result) });
}

// wasm traps / aborts: the runtime's memory is in an unknown state afterwards, the
// page restarts the worker (python exceptions are reported by host_web.py and the
// app goes on)
const FATAL = /RuntimeError|Aborted|unreachable|out of memory|out of bounds/i;
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
    if (data.type === "init") await init(data.canvas, data.base);
    else if (data.type === "event") self.webgymEvent?.(data.kind, data.x ?? 0, data.y ?? 0, data.delta ?? 0, data.key ?? "");
    else if (data.type === "command") self.webgymCommand?.(data.name, data.value ?? null);
    else if (data.type === "restore") self.webgymRestore?.(data.data);
    else if (data.type === "resize") {
      self.canvas.width = data.width;
      self.canvas.height = data.height;
    } else if (data.type === "install") await install(data.name, data.packages, data.check);
    else if (data.type === "exec") {
      // compute-only workers: run python, the value of the last expression comes back
      const value = await pyjs.async_exec_eval(data.code);
      post({ type: "result", id: data.id, value: String(value), memory: pyjs.HEAPU8?.byteLength });
    }
  } catch (e) {
    console.error(e);
    report(e);
  }
};
