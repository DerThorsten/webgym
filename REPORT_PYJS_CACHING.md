# pyjs (npm 4.4.1): loading and caching. What webgym needed and what pyjs could do

These observations come from webgym. It runs 2 to 3 pyjs workers per page: display, trainer, IREE compiler. It installs pytorch, jax and IREE at runtime with `installPackages`.

## Measurements (webgym, Chrome, desktop)

- **Core:** python start plus a 7 MB core archive (numpy, pyb2d3, pyopengl). It is running after about 1.8 s, both cold and on a reload.
- **pytorch:** about 12.5 s on the first visit. On a second visit it still takes about 9 s, even with every package file served from Cache Storage.
- **What fills those 9 s:** solving, decompressing, untarring, writing to MEMFS, and compiling the wasm of the shared libraries. None of this is cached.
- **jax:** about 20 s ("loaded in 20.5 s"). It is imported in the training worker.

## What webgym had to add itself

1. **A package cache.** `worker.js` wraps `self.fetch` so that `.tar.bz2` and `.conda` URLs are served cache-first from Cache Storage. Without this wrapper, every visit downloads pytorch again.
2. **Byte progress for its own archives.** We download the archive ourselves and pass a `blob:` URL to `unpackArchive`. A `#<original url>` suffix keeps the extension, because pyjs picks the decompression from the URL's ending.
3. **A compile worker.** jaxlib and iree-compiler cannot share one process (see REPORT_BUGS.md). So one more pyjs runtime starts, with its own python and its own copy of the environment.

## Findings and suggestions (most gain first)

### 1. `installPackages` solves against the network on every visit
`solve()` (`runtime.ts:140`) calls mambajs `install`, which fetches repodata for every channel each time. With conda-forge noarch, that is several MB of JSON. Even with all packages cached, a second visit therefore needs the network and solver time.

- **Option A:** cache the solved lock, keyed by specs, channels and the base lock, in Cache Storage or IndexedDB. Revalidate it in the background.
- **Option B:** an API that installs from a given lock or list of pinned URLs, such as `installLock(lock)`. An app could then solve at build time (webgym's PLAN already wants pinned URLs and sha256 in `config.json`) and never solve in the browser.

### 2. No built-in package cache
Conda package files are immutable, keyed by filename and sha256. pyjs could cache them by default, for example with `packageCache: "pyjs-packages"` (Cache Storage, cache-first, sha256-checked), instead of every app patching `fetch`. It should also cover the runtime's own environment tarballs (python, pip, …).

### 3. Unpacking runs on every visit
`installPackagesToEmscriptenFS` runs bz2/zstd decompression, untar and `FS.writeFile` for every file, on every visit. For pytorch this is a large part of the 9 s.

- Cache the extracted result: one uncompressed tar, or a file list plus data, in OPFS or IndexedDB, keyed by package sha256.
- Or let a prefix live on a persistent file system (OPFS-backed) so installed packages survive reloads.

### 4. The shared libraries' wasm is compiled on every visit
`loadDynamicLibrary` compiles from bytes in MEMFS, so V8's code cache never applies. Chrome caches compiled code only for `WebAssembly.compileStreaming` of HTTP or Cache Storage responses.

- Compile large `.so` files with `compileStreaming(await caches.match(url))`, or keep the original response per library.
- This is worth the most for torch and jaxlib, which have big `.so` files and need seconds to compile.

### 5. Runtime files are fetched several times per worker
In the dev/test server log, each worker start shows:
- `empack_env_meta.json` 2×;
- `pyjs_runtime_browser.wasm` 2–3× (200, 304, 200);
- each environment tarball (`python-3.14…tar.gz`, `pip…`, `ca-certificates…`) 2×;
- `untarjs.wasm` 2×.

Some of this may be preload/module-script duplication; it needs a closer look. Each fetch is a round trip, and a `200` re-download of the 8+ MB runtime wasm is expensive.

- Fetch each file once, and use `compileStreaming` for the runtime wasm.
- Consider an asset manifest with content hashes (the `copy-assets` file names are not hashed). An app could then serve them `immutable` and skip 304 revalidation.

### 6. Several workers repeat everything
Every worker fetches, compiles and extracts the same runtime and environment. webgym runs 2 or 3 of them.

- A compiled `WebAssembly.Module` can be posted to workers. pyjs could accept `{ wasmModule }`, or offer `prepareRuntime()` on the main thread and `initializePyjsRuntime({ prepared })` in each worker.
- The extracted environment could likewise be shared once as an `ArrayBuffer`.
- The same applies to packages: the trainer and the compile worker both install IREE parts separately.

### 7. Progress for `installPackages`
Today progress only arrives as `logger.log` strings; webgym shows them as the bar label. Structured events would allow real progress bars for the big downloads:
`onProgress({ package, phase: "download" | "extract" | "link", done, total })`.

### 8. Small things
- **Solver bundle:** the lazily loaded solver (mambajs plus rattler, a few MB) has to be compiled again on each visit (see 4). It isn't needed at all when a lock is given (see 1B).
- **Shared-library preloading in workers:** `preloadSharedLibs: !isWorker()` is a good default. Document it: in workers, libraries load synchronously when imported.
- **Leftover `.pyc` files:** `sys.dont_write_bytecode` is not set by default. webgym sets it because edited code is re-imported. With MEMFS, writing `.pyc` files is wasted work on every visit anyway.

### 9. Pre-solve the environment at build time, so the browser needs no mamba
This goes further than 1B. An app knows its packages when it is built (webgym: pytorch, jax, iree-base-runtime/-compiler, per architecture). It could solve them then and pass pyjs only the result: a list of pinned package URLs (+ sha256, + the package name and subdir that pyjs needs to place the files).

- **How:** a build step solves for `emscripten-wasm32` / `emscripten-wasm64` with pixi, rattler or conda-lock and writes the URLs into the app's config. webgym's `build_page.py` already resolves its core packages this way; doing the same for pytorch / jax is a small step.
- **API:** e.g. `initializePyjsRuntime({ packages: [{ url, sha256 }] })` and `pyjs.installPackages({ urls: [...] })`. pyjs fetches and installs exactly these, with no solve.
- **Gain:**
  - no repodata download, so the second visit works offline;
  - no mambajs / rattler bundle to load and compile;
  - no solve time;
  - reproducible environments: every visitor gets the same builds;
  - the sha256 values make a safe cache key for the package cache (2).
- **Keep the solver as an option** for interactive installs, such as a user who types `install scipy` in the page.

### 10. Untar and decompress inside the pyjs runtime, not in a separate `untarjs.wasm`
Today archives go through `untarjs`, a wasm module of its own (`pyjs/untarjs.wasm`, fetched and compiled for each worker). It decompresses and untars into JS memory. pyjs then copies every file into MEMFS with `FS.writeFile`. So each file is copied twice, between two wasm memories via JS.

- **Proposal:** link the decompressors (zlib, bzip2, zstd for `.conda`) and a small tar reader (e.g. libarchive, all on emscripten-forge) **into the runtime** (`pyjs_runtime_browser.wasm`), not into `libpyjs`, the Python extension. The runtime then extracts straight into its own file system: one fetch of the archive bytes into the runtime's memory, then files are written in place.
- **Gain:**
  - one wasm module fewer to fetch and compile per worker;
  - no copying file by file through JS;
  - it can stream: decompress while downloading;
  - it is the natural place to cache extracted packages (3).
- **wasm64:** there is a single `untarjs.wasm` for both architectures. With 64-bit memory the runtime could unpack archives beyond 4 GB; inside the runtime this comes for free.

## Suggested order
0. Pre-solved environments (9) and extraction inside the runtime (10). They make several of the points above unnecessary: no solver (1, part of 8), simpler caching (2, 3), and fewer files to fetch per worker (5, 6).
1. Built-in package cache (2) and a lock or pinned install (1): the second visit works offline and with no solve.
2. Cache the extracted files (3) and use `compileStreaming` for shared libraries (4): the second visit gets much faster.
3. Share the runtime between workers (6) and remove duplicate fetches (5).
