// Checkpoints in IndexedDB, per env + algorithm: the autosave of the current run (the
// worker sends it while training / on stop) and named saves; plus the pretrained runs
// that come with the page (public/pretrained/, read-only). Records:
// {key: "env|algo|name", name, env, algo, iteration, mean_return, saved, data (base64
// .safetensors)}. Without IndexedDB (some private windows) nothing is stored.

const DB = "webgym";
const STORE = "checkpoints";
export const AUTOSAVE = "autosave"; // (= host_web.AUTOSAVE)
const keyOf = (env, algo, name) => `${env}|${algo}|${name}`;

let dbPromise = null;
/** the database, or null without IndexedDB */
function db() {
  dbPromise ??= new Promise((resolve) => {
    try {
      const request = indexedDB.open(DB, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "key" });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return dbPromise;
}

/** run `action(store)` in a transaction, resolve with its request's result */
async function transaction(mode, action) {
  const database = await db();
  if (!database) return null;
  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE, mode);
    const request = action(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(request.result);
    tx.onerror = () => reject(tx.error);
  });
}

export const put = (record) =>
  transaction("readwrite", (s) => s.put({ ...record, key: keyOf(record.env, record.algo, record.name), saved: Date.now() }));
export const remove = (env, algo, name) => transaction("readwrite", (s) => s.delete(keyOf(env, algo, name)));
export const get = (env, algo, name) => transaction("readonly", (s) => s.get(keyOf(env, algo, name)));
/** the checkpoints of env + algo, the autosave first, then the newest; then the
 * pretrained ones of env that come with the page, of every algorithm (a restore
 * switches to it), the best first */
export async function list(env, algo) {
  const all = (await transaction("readonly", (s) => s.getAll())) ?? [];
  const own = all
    .filter((r) => r.env === env && r.algo === algo)
    .sort((a, b) => (b.name === AUTOSAVE) - (a.name === AUTOSAVE) || b.saved - a.saved);
  const shipped = (await pretrained).filter((r) => r.env === env).sort((a, b) => b.mean_return - a.mean_return);
  return [...own, ...shipped.map((r) => ({ ...r, name: `Pretrained · ${r.label ?? r.algo}`, pretrained: true }))];
}

// the pretrained runs that come with the page (public/pretrained/index.json, written by
// build_page.py): [{file, env, algo, label, iteration, mean_return}]
const pretrained = fetch("pretrained/index.json")
  .then((r) => (r.ok ? r.json() : []))
  .catch(() => []);

/** a pretrained run's checkpoint, base64 (as the worker takes it) */
async function pretrainedData(file) {
  const bytes = new Uint8Array(await (await fetch(`pretrained/${file}`)).arrayBuffer());
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** the checkpoints panel: save with a name, the list of env + algo with restore /
 * delete. `send(message)` talks to the worker */
export function setupPanel(send) {
  const nameInput = document.getElementById("ckpt-name");
  const listEl = document.getElementById("ckpt-list");
  let current = { env: null, algo: null };

  async function refresh() {
    const records = await list(current.env, current.algo);
    listEl.replaceChildren();
    if (!records.length) listEl.textContent = `no checkpoints for ${current.env} / ${current.algo} yet`;
    for (const r of records) {
      const row = document.createElement("div");
      row.className = "ckpt";
      const mean = r.mean_return == null ? "" : ` · mean ${Math.round(r.mean_return)}`;
      const when = r.pretrained
        ? "comes with the page"
        : new Date(r.saved).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
      row.innerHTML = `<div><b></b><div class="meta">iteration ${r.iteration}${mean} · ${when}</div></div>`;
      row.querySelector("b").textContent = r.name;
      const restore = Object.assign(document.createElement("button"), { textContent: "Restore" });
      restore.onclick = async () => send({ type: "restore", data: r.pretrained ? await pretrainedData(r.file) : r.data });
      row.append(restore);
      if (!r.pretrained) {
        // (the pretrained ones can't be deleted: they come with the page)
        const del = Object.assign(document.createElement("button"), { textContent: "Delete" });
        del.onclick = async () => {
          await remove(r.env, r.algo, r.name);
          refresh();
        };
        row.append(del);
      }
      listEl.append(row);
    }
  }

  const save = () => {
    const name = nameInput.value.trim();
    if (!name || name === AUTOSAVE) return;
    send({ type: "command", name: "save", value: name });
    nameInput.value = "";
    nameInput.blur();
  };
  document.getElementById("ckpt-save").onclick = save;
  nameInput.addEventListener("keydown", (e) => e.key === "Enter" && save());

  return {
    /** the worker sent a checkpoint / a deletion */
    async onMessage(data) {
      if (data.type === "checkpoint") await put(data);
      else if (data.type === "delete_checkpoint") await remove(data.env, data.algo, data.name);
      refresh();
    },
    /** the app's controls arrived: on a new env / algo (also after a page load or a
     * worker restart: `fresh`) its autosave is restored (the run continues) */
    async onControls(controls, fresh) {
      const control = (name) => controls.find((c) => c.name === name);
      // (env: its key, the name with its settings, e.g. "WaterCannon · balls 2")
      const next = { env: control("env")?.key ?? control("env")?.value, algo: control("algo")?.value };
      if (!fresh && next.env === current.env && next.algo === current.algo) return;
      current = next;
      refresh();
      const autosave = await get(current.env, current.algo, AUTOSAVE);
      if (autosave) send({ type: "restore", data: autosave.data });
    },
  };
}
