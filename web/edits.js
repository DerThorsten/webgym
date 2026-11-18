// Edited python files (the code panel), kept in IndexedDB: a reload brings them back.
// Records: {path (e.g. /app/webgym/envs/cartpole.py), text, version (the sources
// archive of the page: an edit of another webgym version is not applied), saved}.
// Without IndexedDB (some private windows) nothing is kept.

const DB = "webgym-code";
const STORE = "files";

let dbPromise = null;
/** the database, or null without IndexedDB */
function db() {
  dbPromise ??= new Promise((resolve) => {
    try {
      const request = indexedDB.open(DB, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "path" });
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

/** keep the edited `text` of the file `path` (of webgym `version`) */
export const saveEdit = (path, text, version) =>
  transaction("readwrite", (s) => s.put({ path, text, version, saved: Date.now() }));

/** forget the edit of `path` (the original comes back with the next reload) */
export const removeEdit = (path) => transaction("readwrite", (s) => s.delete(path));

/** the kept edits: {current: [{path, text}] of this `version`, stale: [paths] of others} */
export async function storedEdits(version) {
  const all = (await transaction("readonly", (s) => s.getAll())) ?? [];
  return {
    current: all.filter((r) => r.version === version).map(({ path, text }) => ({ path, text })),
    stale: all.filter((r) => r.version !== version).map((r) => r.path),
  };
}
