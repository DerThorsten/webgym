// Checkpoints in IndexedDB, per env + algorithm: the autosave of the current run (the
// worker sends it while training / on stop) and named saves. Records:
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
/** the checkpoints of env + algo, the autosave first, then the newest */
export async function list(env, algo) {
  const all = (await transaction("readonly", (s) => s.getAll())) ?? [];
  return all
    .filter((r) => r.env === env && r.algo === algo)
    .sort((a, b) => (b.name === AUTOSAVE) - (a.name === AUTOSAVE) || b.saved - a.saved);
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
      const when = new Date(r.saved).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
      row.innerHTML = `<div><b></b><div class="meta">iteration ${r.iteration}${mean} · ${when}</div></div>`;
      row.querySelector("b").textContent = r.name;
      const restore = Object.assign(document.createElement("button"), { textContent: "Restore" });
      restore.onclick = () => send({ type: "restore", data: r.data });
      const del = Object.assign(document.createElement("button"), { textContent: "Delete" });
      del.onclick = async () => {
        await remove(r.env, r.algo, r.name);
        refresh();
      };
      row.append(restore, del);
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
      const value = (name) => controls.find((c) => c.name === name)?.value;
      const next = { env: value("env"), algo: value("algo") };
      if (!fresh && next.env === current.env && next.algo === current.algo) return;
      current = next;
      refresh();
      const autosave = await get(current.env, current.algo, AUTOSAVE);
      if (autosave) send({ type: "restore", data: autosave.data });
    },
  };
}
