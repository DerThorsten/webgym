// Worker benchmark: CartPole env steps per second with 1..k compute-only pyjs workers
// (webgym/bench.py), separate worlds vs one world; plus startup time and wasm memory
// per worker. ?workers=1,2,3,4 ?n=64 ?steps=300. Result: console "bench result" (JSON).

const params = new URLSearchParams(location.search);
const COUNTS = (params.get("workers") ?? "1,2,3,4,6").split(",").map(Number);
const N = Number(params.get("n") ?? 64);
const STEPS = Number(params.get("steps") ?? 300);
const out = document.getElementById("out");
const log = (line) => (out.textContent = (out.textContent === "running…" ? "" : out.textContent) + line + "\n");

/** a compute-only worker: {worker, exec(code) -> {value, memory}}, resolves when python runs */
async function startWorker() {
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const pending = new Map();
  let nextId = 0;
  const ready = new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.type === "progress" && data.id === "startup" && data.phase === "done") resolve();
      else if (data.type === "result") pending.get(data.id)(data);
      else if (data.type === "error" || data.type === "fatal") reject(new Error(data.error));
    };
  });
  worker.postMessage({ type: "init", canvas: null, base: location.href });
  await ready;
  const exec = (code) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      worker.postMessage({ type: "exec", id, code });
    });
  return { worker, exec };
}

/** run bench.run in all `workers` at once (each N candidates): total env steps / s */
async function parallel(workers, kind) {
  const code = `from webgym import bench\nbench.run(${N}, ${STEPS}, ("${kind}",))`;
  const results = await Promise.all(workers.map((w) => w.exec(code)));
  return results.reduce((sum, r) => sum + JSON.parse(r.value)[kind], 0);
}

async function main() {
  const result = { cores: navigator.hardwareConcurrency, n: N, steps: STEPS, startup_s: [], rows: [] };
  log(`hardwareConcurrency ${result.cores}, ${N} candidates x ${STEPS} steps per worker`);
  const workers = [];
  for (const k of COUNTS) {
    while (workers.length < k) {
      const t0 = performance.now();
      workers.push(await startWorker());
      result.startup_s.push(+((performance.now() - t0) / 1000).toFixed(2));
    }
    const active = workers.slice(0, k);
    await parallel(active, "separate"); // warm up (imports, first allocations)
    const separate = await parallel(active, "separate");
    const oneWorld = await parallel(active, "one_world");
    const memory = (await active[0].exec("1")).memory;
    const row = { workers: k, separate: Math.round(separate), one_world: Math.round(oneWorld), wasm_mb: Math.round(memory / 1e6) };
    result.rows.push(row);
    log(`${k} worker(s): separate worlds ${row.separate} steps/s, one world ${row.one_world} steps/s, wasm memory/worker ${row.wasm_mb} MB`);
  }
  log(`startup per worker (s): ${result.startup_s.join(", ")}`);
  console.info("bench result", JSON.stringify(result));
  for (const w of workers) w.worker.terminate();
}

main().catch((e) => {
  log(`error: ${e.message}`);
  console.error(e);
});
