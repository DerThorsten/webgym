# Handover: webgym on pyjs 5

pyjs 5 (npm `@derthorsten/pyjs` 5.1.0, branch `pixi-build` of pyjs, PR
emscripten-forge/pyjs#153) was built from `REPORT_PYJS_CACHING.md`: almost
every point of it is in pyjs now. This is how webgym gets the most out of it.

## Status of pyjs 5

- Not on npm yet. Until it is published, build it locally and use the package
  dir (e.g. `"@derthorsten/pyjs": "file:../../pyjs/build_root/npm/package"` in
  `web/package.json`, or the packed `.tgz`):

  ```bash
  cd ~/src/pyjs && pixi run npm-package     # wasm32 + wasm64 -> build_root/npm/package, build_root/npm/tgz/*.tgz
  ```

- Still Python 3.14 (`"pyjs": {"python": ...}` in its package.json).
- Breaking: the classic API is gone (`bootstrap_from_empack_*`, `init_phase_*`,
  `_untar`, `_install_conda_file`, untarjs / `untarjs.wasm`, the
  `extraPackages` option, now `packages`).

## The report, point by point

| report | pyjs 5 |
|---|---|
| 1 / 9: solving against the network on every visit | **pinned packages**: `urls: [{url, sha256}]` (init) and `installPackages([{url, sha256}])`, no solver, no repodata. Solve at build time: `npx @derthorsten/pyjs lock torch jax ... -o lock.json` |
| 2: no package cache (webgym's fetch wrapper) | built in, on by default (Cache Storage, by package URL). **Remove the `self.fetch` wrapper** |
| 3: unpacking on every visit | the cache keeps the **installed layout** (uncompressed tar): a warm install only untars, no bz2 / zstd |
| 4: shared-library wasm compiled on every visit | measured: not the cost (compiling 14 MB is ~10 ms, lazy). Linking (instantiation, dylink) is what is left, ~0.3 s for scipy. Not cached; nothing to do on the app side |
| 5 / 6: runtime fetched several times, every worker repeats everything | `compileRuntime()` once in the main thread, post the module, `initializePyjsRuntime({ wasmModule })` in each worker; the package cache is shared by all workers of the origin |
| 7: progress only as log strings | `onProgress({item, phase, done, total})`, phases `solve`, `download` (bytes), `cached`, `extract`, `link`, `start` |
| 8: `.pyc` written in vain | the opposite now: python's bytecode is **cached** per package (see below). Keep `sys.dont_write_bytecode = True` only where webgym re-imports *edited* code (or invalidate those modules), not globally |
| 10: untarjs separate | archives / packages are extracted by the runtime itself (libarchive), straight into its file system; `.conda` and `.tar.bz2` |

Measured on a second visit (numpy + scipy + pandas + matplotlib, chromium):
init + imports went from 5.4 s to 1.4 s. Package unpacking dropped from 2.1 s to
0.08 s, imports from 2.2 s to 0.5 s (bytecode cache), python start from
0.36 s to 0.09 s.

## Recommended setup (fastest)

**1. Pin everything at build time, no solver in the browser.**

```bash
npx @derthorsten/pyjs lock pytorch jax iree-base-runtime --arch wasm64 -o lock.json
# (or: node <pyjs package>/bin/pyjs.mjs lock ...; default: both archs)
```

It solves against the python shipped in that pyjs version and writes
`{"wasm64": [{url, sha256}, ...]}` with all dependencies. Make one lock per
worker role (display / trainer / compiler) if they need different packages,
so the IREE compiler and jaxlib never end up in one solve.

**2. Main thread: compile the runtime once, give it to every worker.**

```js
import { compileRuntime } from "@derthorsten/pyjs/wasm64";
const wasmModule = await compileRuntime();       // (assetsUrl: same options as the init)
for (const worker of [display, trainer, compiler]) worker.postMessage({ wasmModule, lock });
```

**3. Each worker: one init with everything it needs.**

```js
import { initializePyjsRuntime } from "@derthorsten/pyjs/wasm64";

self.onmessage = async ({ data }) => {
  const pyjs = await initializePyjsRuntime({
    wasmModule: data.wasmModule,                // no second fetch / compile
    urls: data.lock.trainer,                    // pinned: no solve, cached after the first visit
    archives: [{ url: coreArchiveUrl, path: "/" }], // webgym's own core archive (byte progress built in)
    onProgress: (p) => postMessage({ progress: p }),
  });
  console.log(pyjs.initTimings);                // where the time went
  pyjs.exec("import torch");
  pyjs.saveBytecode();                          // keep the bytecode of these imports now
};
```

- Everything starts in parallel: runtime instantiation, downloads (cache
  first), the archive. The extraction order is env, then packages, then
  archives, then files, then python starts, then shared libraries. In a worker
  they are not preloaded (`preloadSharedLibs` defaults to false there; `dlopen`
  at `import` may compile synchronously in workers).
- Install the big packages **at init** rather than with `installPackages`
  afterwards: the init overlaps downloads with the runtime start. Use
  `installPackages([...pinned])` later only for packages a user asks for; it
  takes the same pinned form and reports `pyjs.installTimings`.
- `archives` replaces webgym's own download + `blob:` URL + `unpackArchive`
  dance: pyjs reports byte progress for it and detects the format from the
  content (no `#<url>` suffix trick). With a content hash in the archive's
  file name, add `cache: true` (pyjs 5.1): it is then cached like a package
  (unpacked files + bytecode, no download on later visits).

**4. The caches (automatic, both Cache Storage, per origin):**

- `pyjs-packages-v2`: the installed layout of every pinned / solved package,
  keyed by URL. Shared by all workers, survives reloads. A warm install only
  untars.
- `pyjs-packages-v2-bytecode`: python's `.pyc` files per package (and for the
  python environment itself). Saved when new ones appear (checked every few
  seconds in workers too) or on `pyjs.saveBytecode()`. Call it once after the
  heavy imports (torch, jax): then even a visit that ends early leaves the
  bytecode behind.
- `cache: false` turns both off, `cache: "name"` uses another name,
  `bytecodeCache: false` only the bytecode.

## Pitfalls

- **First visit with several workers:** the caches are written in the
  background after a package is installed. Workers that start at the same
  moment each download on the very first visit. If that matters, start the
  trainer first and the others a moment later, or accept it once.
- **The URL is the cache key.** emscripten-forge package URLs are immutable.
  Never serve different content under one URL (own builds: put the hash in
  the file name).
- `verifySha256: true` checks the lock's sha256 values. Pinned packages with a
  sha256 then always come from the network: the cached layout cannot be
  checked against the package file's hash. Off by default.
- jaxlib and iree-compiler still cannot share one process (REPORT_BUGS.md):
  keep the separate compile worker. With `wasmModule` and the shared package
  cache it costs much less than before.
- Cache Storage needs a secure context (https / localhost). Browsers may evict
  it under storage pressure; for a training app, call
  `navigator.storage.persist()`.
- Bytecode of code that is edited and re-imported at runtime: python
  revalidates `.pyc` against the source's mtime + size, so edits are picked up.
  Only identical size *and* mtime would hide an edit (rare, e.g. a file
  rewritten within the same second with the same length).

## Reference

pyjs docs, npm page (options table, caching, progress, the lock CLI,
workers): `docs/npm.md` in the pyjs repository, or the published docs once
#153 is merged. Every option is typed in `dist/types.d.ts` of the package.
