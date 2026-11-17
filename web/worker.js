// The pyjs web worker: python, the simulation, training and WebGL2 rendering on the
// OffscreenCanvas the page transferred. Talks to main.js with messages:
//   page -> worker: {type: "init", canvas, base}, {type: "resize", width, height},
//                   {type: "event", kind, x, y, delta, key} (pointer: drawing-buffer px; keys),
//                   {type: "command", name, value} (a widget changed),
//                   {type: "install", name, packages}
//   worker -> page: {type: "progress", id, phase, done, total}, {type: "ready"},
//                   {type: "controls", controls}, {type: "history", entries, reset},
//                   {type: "state", state}, {type: "installed", name, seconds}, {type: "error", error}
// (archive download / unpack: from pyb2d3/examples/web/main.js)

import { initializePyjsRuntime } from "@derthorsten/pyjs";

const post = (message) => self.postMessage(message);
const progress = (id, phase, done = 0, total = 0) => post({ type: "progress", id, phase, done, total });

/** the bytes at `url` as a Blob, reporting bytes received */
async function download(url, onBytes) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
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
  self.canvas = canvas; // host_web.py reads it via pyjs.js.canvas
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
from webgym import host_web
_host = host_web.start()
`);
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

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init(data.canvas, data.base);
    else if (data.type === "event") self.webgymEvent?.(data.kind, data.x ?? 0, data.y ?? 0, data.delta ?? 0, data.key ?? "");
    else if (data.type === "command") self.webgymCommand?.(data.name, data.value ?? null);
    else if (data.type === "resize") {
      self.canvas.width = data.width;
      self.canvas.height = data.height;
    } else if (data.type === "install") await install(data.name, data.packages, data.check);
  } catch (e) {
    console.error(e);
    post({ type: "error", error: String(e?.stack ?? e) });
  }
};
