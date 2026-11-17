# Handover: a Gymnasium-like RL environment library that runs in the browser

Goal: something like [Gymnasium](https://gymnasium.farama.org/)
([repo](https://github.com/Farama-Foundation/Gymnasium)), but **in the browser**:
Python in WebAssembly via **pyjs** + packages from **emscripten-forge**. This file
collects what the previous sessions learned about using pyjs in web pages and
about getting packages (own ones included) into a deployment.

Repos on this machine (all of the user):
- `/Users/thorstenbeier/src/pyjs`: pyjs (Python <-> JS FFI, npm package `@derthorsten/pyjs`)
- `/Users/thorstenbeier/src/recipes`: emscripten-forge recipes (branch `emscripten-6x` upstream)
- `/Users/thorstenbeier/src/pyb2d3`: box2d 3 bindings + a browser sandbox. **The
  reference project**: everything below is used there.

## 1. The stack in one paragraph

pyjs = CPython 3.14 compiled with emscripten 6 + a JS <-> Python FFI. The npm package
`@derthorsten/pyjs` (current: 4.4.1, later `@emscripten-forge/pyjs`) contains the
runtime (`.js` + `.wasm`) for wasm32 and wasm64 plus a packed Python environment (stdlib
+ the `pyjs` module). Conda packages built for `emscripten-wasm32` / `emscripten-wasm64`
come from the channel `https://repo.prefix.dev/emscripten-forge/6x`; pure python
`noarch` packages also from conda-forge. wasm32 works in all browsers; wasm64
(memory64) allows > 4 GB memory and needs a browser with memory64 support.

**jax and pytorch are available on emscripten-forge for `emscripten-wasm64`**: training
RL agents in the browser is possible, which makes **wasm64 the likely target** of this
project (import pyjs from `dist/wasm64.js` / `@derthorsten/pyjs/wasm64`; recipes built
with `--target-platform emscripten-wasm64`). Check which other packages the project
needs exist for wasm64 too.

## 2. Using pyjs in a web page

### Plain page, pyjs from the CDN (no npm, no bundler)

```html
<script type="module">
  const { initializePyjsRuntime } = await import(
    "https://cdn.jsdelivr.net/npm/@derthorsten/pyjs@4.4.1/dist/index.js"
  );
  const pyjs = await initializePyjsRuntime({ extraPackages: ["numpy"] });
  pyjs.exec("import numpy as np; print(np.arange(10).sum())");
</script>
```

- Pin the version (JS and wasm must match). Import `dist/index.js` (wasm32) or
  `dist/wasm64.js`, never jsDelivr's `/+esm` (breaks the `import.meta.url` asset paths).
- Works in module workers too (`new Worker(url, { type: "module" })`).

### npm + bundler (vite, webpack, rollup, esbuild, parcel are tested)

`npm install @derthorsten/pyjs`, `import { initializePyjsRuntime } from "@derthorsten/pyjs"`.
Bundlers do not copy the runtime assets: by default they are fetched from jsDelivr
(exact version); to self-host: `npx @derthorsten/pyjs copy-assets public/pyjs` and
`initializePyjsRuntime({ assetsUrl: "/pyjs/", cdnFallback: false })`.

### Self-contained page without npm (what pyb2d3 does)

`examples/web/build_page.py` in pyb2d3 downloads the npm tarball
(`https://registry.npmjs.org/@derthorsten/pyjs/-/pyjs-4.4.1.tgz`) at build time and copies
its `dist/` + `assets/` (wasm32 only) into the page dir: no CDN for pyjs at runtime.

### API (returned object = the emscripten Module)

`exec(code)`, `eval(expr)`, `exec_eval(script)` (value of the last expression),
`await async_exec_eval(script)` (top-level `await`), `eval_file(path)`, `to_js(obj)`,
`FS` (emscripten file system), `installPackages(specs)`, `unpackArchive({url, path})`,
`pyjsArch`, `pythonVersion`. Types: `pyjs/npm/src/types.ts`. Docs: `pyjs/docs/npm.md`,
`cdn.md`, `JavaScript_API.md`, `usage.md`.

`initializePyjsRuntime` options: `extraPackages`, `channels`, `archives`, `assetsUrl`,
`cdnUrl`, `cdnFallback`, `preloadSharedLibs`, `print`, `printErr`, `logger`, `verbose`,
`moduleOptions`.

From Python: `import pyjs` / `from pyjs import js` (`js.document`, `js.window`, ...);
`pyjs.create_callable(fn)` hands Python functions to JS (event listeners);
`pyjs.set_canvas(canvas)` + `pyjs.request_animation_frame_loop(cb)` for a render loop;
asyncio runs on a browser event loop (`webloop`), so `await asyncio.sleep(...)` works;
`pyjs.run_sync(promise)` blocks on a JS promise where the browser supports JSPI
(`pyjs.can_run_sync()`).

## 3. Getting packages into a deployment

Python version: packages must be built for the python of the pyjs release (pyjs 4.4.x:
**3.14**, builds named `py314...`). The npm package's `package.json` says which python a
pyjs has (`"pyjs": {"python": ...}`).

### a) Published packages, solved + installed in the browser

`extraPackages: ["numpy", "scipy>=1.14"]` or later `await pyjs.installPackages([...])`.
mambajs solves against `channels` (default: emscripten-forge 6x, conda-forge) in the
browser. Simple, but every page load solves + downloads (a few seconds).

What can come from where:
- compiled packages (C/C++/Cython/Rust): only from emscripten-forge (built for
  `emscripten-wasm32`).
- pure python: conda-forge works **if the conda-forge package is `noarch: python`**.
  Packages built per platform on conda-forge (like `gymnasium`, see 5.) are not
  installable; they need an emscripten-forge (or own) recipe.

### b) Own packages: recipe + rattler-build -> a self-hosted channel

A package = a rattler-build recipe (`recipe.yaml`). Build for the browser:

```bash
rattler-build build --package-format tar-bz2 \
  -c https://repo.prefix.dev/emscripten-forge/6x -c conda-forge \
  --target-platform emscripten-wasm32 \
  -m variant.yaml --recipe <recipe dir> --output-dir <output dir>
```

- `variant.yaml` = the pinnings of emscripten-forge:
  `https://raw.githubusercontent.com/emscripten-forge/recipes/refs/heads/emscripten-6x/variant.yaml`
  (in the recipes checkout: `variant.yaml`).
- In the recipes repo: `pixi run -e rattler-build-env build-emscripten-pkg recipes/recipes_wasm/<name>`
  builds wasm32 + wasm64 (the user's rule: always test both archs there); the output
  goes to `recipes/output/`. A local `output/` shadows prefix.dev packages of the same
  name in later builds.
- Pure python packages: `noarch: python` in the recipe, then no `--target-platform`
  needed and one build serves both archs.
- Compiled packages: see pyb2d3 `emscripten_forge/recipe/` (`recipe.yaml.template`,
  `build_pkg.sh`: cross-python, `python-dev` in host, `-DPython_*` cmake args, the
  extension as an emscripten side module via `overwriteProp.cmake`) and
  `scripts/build_emscripten_forge_pkg.sh` (builds the local checkout; `pixi run emscripten`).
- Tests in recipes: `pytester` (runs pytest in node with the built package).

**The output dir is a channel**: `noarch/`, `emscripten-wasm32/`, each with a
`repodata.json`, all static files. Deploy it next to the page (GitHub Pages works) and
list it **first** (channel order = priority, so a local build beats a published one):

```js
const pyjs = await initializePyjsRuntime({
  channels: [new URL("channel", location.href).href,
             "https://repo.prefix.dev/emscripten-forge/6x", "conda-forge"],
  extraPackages: ["my-package"],
});
```

- Give local builds a high build number (pyb2d3: `number: 1000`) and the real version,
  otherwise the solver may prefer a published package.
- rattler-build may leave a stale `repodata.json` when it overwrites a package with
  the same file name: delete the output and rebuild when in doubt.
- Reference: pyjs `pyjs_test_driver/test_channel.py` generates a minimal channel (one
  noarch + one arch package with repodata) and `tests/npm/test_channel.py` installs
  from it, including the priority over conda-forge.

### c) Files that are not a package (python sources, data, assets)

Pack them as a tar(.gz) archive; pyjs unpacks it (untarjs) into the file system:
`archives: [{ url: "env_src.tar.gz", path: "/src" }]` (before python starts) or
`await pyjs.unpackArchive({ url, path })` later, then
`sys.path.insert(0, "/src")` and `importlib.invalidate_caches()`.
Best for the project's own code during development: no recipe, no rebuild of a package.

### d) Pre-packed packages, no solver in the browser (fastest start; pyb2d3 does this)

`pyb2d3/examples/web/build_page.py`:
1. downloads the needed `.tar.bz2` from the channel's `repodata.json` at build time
   (newest build for `py314`), plus the local build of the own package;
2. repacks only the runtime files (drops tests, headers, docs, `.pyi`, ...) into one
   `packages-<hash>.tar.gz` (unpacked at `/`) and the sources into
   `examples-<hash>.tar.gz` (content hash in the name = no stale cache);
3. writes `config.json` (pyjs url, archives, the shared libraries in them).

`main.js` then starts pyjs **without** `extraPackages`, unpacks the archives (cached in
IndexedDB by hash), and **preloads the shared libraries** (`pyjs.loadDynamicLibrary`
with `loadAsync: true`; plain libs global + by name first, then the extension
modules by path). Without that, importing numpy fails in the main thread: browsers
refuse to compile a large wasm (libopenblas ~9 MB) synchronously. (`installPackages`
does this preloading itself.) Copy these two files as the starting point.

### e) Upstream

Recipes for emscripten-forge: PR against the `emscripten-6x` branch of
`emscripten-forge/recipes` (`recipes/recipes_wasm/<name>/recipe.yaml`). Once merged the
package is on `repo.prefix.dev/emscripten-forge/6x` and a) works.

## 4. Rendering and the main loop (learned in pyb2d3)

- Reference frontend: `pyb2d3/examples/pyb2d3_sandbox_webgl/webgl_frontend.py` (WebGL2
  through PyOpenGL on the page's canvas, HTML panel built from Python, pointer/key
  events forwarded with one small JS listener per event type), started from
  `examples/web/main.js`; touch gestures in `gestures.py`.
- PyOpenGL: set `OpenGL.ERROR_CHECKING = False` **before** the first `import OpenGL.GL`
  (glGetError per call costs ~35 µs in the browser: 30x slower drawing).
- Every Python <-> JS call costs: batch data (numpy arrays -> one `glBufferSubData`),
  avoid per-object callbacks into Python, avoid per-draw GL state changes.
- Never block the main thread: the frame loop is `requestAnimationFrame`
  (`pyjs.request_animation_frame_loop`), not `while True`. Long training loops belong in
  a module worker (no canvas there unless OffscreenCanvas) or must yield
  (`await asyncio.sleep(0)`).
- Simplest rendering for gym envs: `render_mode="rgb_array"` -> numpy (H, W, 3) ->
  `ImageData` on a 2D canvas (one JS call per frame); WebGL only when needed.
- Mobile: `100dvh`, pinch zoom only on empty space, panel as bottom sheet / drawer:
  see `pyb2d3/examples/web/index.html`.

## 5. Gymnasium specifics (open, to check first)

- conda-forge `gymnasium` 1.3.0 is **not noarch**: per platform builds (`np2py311...`,
  linux/osx/win only), depends on `numpy`, `cloudpickle`, `typing_extensions`,
  `farama-notifications`, `jax-jumpy`. -> needs a recipe for emscripten-forge (pure
  python: `noarch: python` built from the PyPI sdist is likely enough; check whether
  `jax-jumpy` is really needed / noarch). First check whether emscripten-forge 6x
  already has `gymnasium` (search its `noarch/` and `emscripten-wasm32/` repodata).
- Gymnasium's own renderers use **pygame** (classic control) and **box2d-py** (box2d
  2.x, Box2D envs) and MuJoCo: check availability on emscripten-forge; pyb2d3 (box2d 3,
  different API) is available as an emscripten package (`pyb2d3` recipe in
  emscripten-forge) and could back new physics envs.
- Agents: jax and pytorch exist for `emscripten-wasm64` (see 1.), so environments and
  learning code can run together in the page; training belongs in a module worker
  (it would block the main thread), the environment's rendering in the main thread.
- Decide early: a browser-only library with a Gymnasium-compatible API (`Env`,
  `spaces`, `reset/step`, wrappers), or Gymnasium itself + browser renderers/frontends.

## 6. Pitfalls

- Version coupling: pyjs version <-> python version <-> package builds (`py314`). Pin
  pyjs in pages.
- Packages are downloaded from prefix.dev / conda.anaconda.org with CORS; prefix.dev
  sometimes answers without CORS headers (retry).
- Browser caching during development: serve with `Cache-Control: no-store` (a tiny
  `http.server` subclass) and put content hashes into file names / `?v=` query params.
- `sys.dont_write_bytecode = True` if code is edited and re-imported in the page.
- Headless testing: playwright (chromium with `--use-angle=swiftshader
  --enable-unsafe-swiftshader` for WebGL, webkit = Safari engine). pyb2d3 has
  `pixi run -e web-test web-test`.

## 7. The user's working rules (also in ~/.claude/CLAUDE.md)

- One plain shell command per call, literal absolute paths, no pipes / `&&` / `$(...)` /
  shell variables; files changed only with Edit / Write; temp files in the session
  scratchpad. Subagent prompts must repeat these rules.
- Keep it simple, less code is better; pixi for new projects.
- Every function gets a short docstring / comment about its intent.
- No literature citations ("Author & Author", "et al.") in texts / comments.
- Commits / pushes only when asked; the user usually does them.
