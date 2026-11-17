# webgym — plan

## Context
A Gymnasium-like RL library on pyb2d3 (box2d 3) that runs natively and in the browser
(pyjs + emscripten-forge). Goal is an impressive, fun demo of *live* training: you watch a
population of agents improve in real time, can drag/poke them with the mouse, and can edit or
upload the python env + training code in the page (the reason Python runs in the browser).
`/Users/thorstenbeier/src/webgym` is empty except docs; pyb2d3's sandbox + web page are the
reference to copy from.

## Decisions
- **API:** own minimal, gymnasium-style signatures, no gymnasium dependency (no recipe).
- **Vectorize only where it pays off:** the base is a plain single env. Population training uses
  a generic `VecEnv` (list of independent envs/worlds, stepped in a loop). An env *may* offer a
  native batched version (N agents in ONE world, collision-filtered, `b2d.Bodies` batch obs) when
  measurements show it is faster and it makes sense (cartpole, swarm: likely; walker on random
  terrain, interacting agents: likely not). Either way the view overlays the population: several
  worlds can be drawn with the same camera into one `begin_draw`/`end_draw` pass, translucent
  "ghosts", best agent highlighted. Benchmark both for CartPole in milestone 1.
- **ML:** showing that **pytorch and jax run in the browser is a core goal**: both are
  first-class (torch PPO, jax trainer), not optional extras. But the page must not wait for
  them: the numpy core (MLP + CEM/ES) is usable as soon as the core is loaded; torch / jax are separate
  packages fetched from the emscripten-forge channel while it runs (never part of the
  deployment; see Loading), with a
  clear notice ("PyTorch: ~N MB, downloading… / cached"), size + progress per archive, and
  their algos selectable once loaded. First visit: fast start, big downloads in background.
  **Second visit: much faster**, everything from the local cache (see Loading).
- **Threads:** ONE pyjs module worker does sim + training + WebGL2 rendering on an
  OffscreenCanvas. Main thread: layout, controls, reward chart, Monaco editor, file system
  access; forwards pointer/key/resize events via `postMessage`.
- **Envs:** CartPole first; then lander, swarm, walker.
- **No auto-training:** after loading, the page shows the env running live and interactive
  (random / zero policy, or a pretrained one from `?weights=`), and the user chooses env +
  algo and clicks **Train**. Meanwhile pytorch (and jax) download in the background right
  after the core is ready, so they are usually there by the time the user picks them;
  picking one that isn't loaded yet just shows its progress bar and starts when ready.
- **Checkpoints / saves (browsable):**
  - Format: one small **`.safetensors`** file (standard for weights: JSON header + raw
    little-endian arrays, no pickle, readable from torch / jax / numpy / JS / HF Hub).
    Tensors = policy weights + optional trainer state (ES mean/sigma, optimizer moments) to
    **resume training**. `__metadata__` (string map, one key holding JSON): env name +
    version, obs/act spec, policy arch, algo, hyperparameters, iteration, reward, reward
    curve. Read/write with ~20 lines of numpy in `checkpoints.py` (no dependency; the
    `safetensors` package is optional natively). The JS panel parses only the header for
    listing.
    torch / jax policies are exported to the same numpy format, so any checkpoint can be
    *watched* with the numpy MLP, without downloading torch / jax.
  - Sources in one "Checkpoints" panel, filtered by env: (1) a **built-in gallery** we train
    natively and ship in the deployment (KBs each, e.g. per env "beginner / ok / expert");
    (2) **auto-saves** during training (every k iterations + best so far) in IndexedDB,
    browsable as a timeline to compare stages; (3) user saves (name it, export/import file via
    File System Access); (4) `?weights=<url>` (e.g. a Hugging Face or GitHub URL).
  - Actions: watch (play policy live), resume training, download, copy share link.
  - Natively: same files in a `checkpoints/` dir, `pixi run view cartpole --weights ...`.
  - Third-party pretrained models (e.g. stable-baselines3 on Hugging Face) are trained on
    gymnasium's own (box2d 2 / analytic) envs: different dynamics and obs, so not usable as
    is. Our own gallery is the reliable source; Hugging Face Hub can later host community
    checkpoints.
- **jax + WebGPU:** jax may use a WebGPU backend. Detect support in the worker
  (`navigator.gpu` + `requestAdapter()`), show "jax: WebGPU" or "jax: CPU" and pick the
  backend accordingly; fall back to CPU. Check in milestone 0 what the emscripten-forge jax
  build actually offers (WebGPU plugin / PJRT) and which browsers expose WebGPU in workers.
- **Progress bars for all loading:** startup (runtime, core archive, unpack, import) and each
  big package (torch, jax: MB downloaded / total, cached, unpacking, importing). Details under
  Loading.

## Status / findings (2026-11-17)
- Milestone 0 done: pyjs (npm 4.4.1, py 3.14) in a module worker, WebGL2 on an OffscreenCanvas
  via PyOpenGL, pyb2d3 1.0.0 from emscripten-forge. Page start ~1.6 s (localhost), core archive
  7.3 MB. pytorch from the channel: 8 s download + 3.6 s import; jax: 6 s + 2.7 s, **CPU device
  only** (this jaxlib build has no WebGPU backend).
- Envs: CartPole, LunarLander, Walker; numpy MLP + CEM (all three learn; CartPole < 1 s native).
- Plots live in the host UI (HTML canvas in the page, implot natively), fed by `App.history`;
  only the world is drawn in GL.
- Watch mode extras: human controller (keyboard via `Env.human_action`), auto reset switch,
  end-of-episode pause; training draws the leader (best alive candidate) solid over the ghosts.

## Error handling
- **Python exceptions** (env / trainer / edited code): reported with traceback, the app falls
  back to watch mode, python keeps running.
- **Hard errors** (wasm traps: memory access out of bounds, unreachable, abort, OOM): the
  runtime's memory is undefined afterwards. The worker reports them as `fatal` (also via its
  global error handlers); the page terminates it and starts a fresh worker on a fresh canvas
  (~1.5 s, cached), env / algo kept via the URL, notice shown. > 3 crashes / minute: no more
  restarts, error + "Reload page". Tested with a real trap (`webgymCrashTest()`,
  `test_page.py --crash`).
- **Prevention**: guards that turn would-be corruption into python exceptions, e.g.
  `core.new_world()` (box2d silently returns an invalid world beyond `b2d.MAX_WORLDS` = 128:
  the cause of the first crash), populations freed right away (`Env.close()`), population
  capped at 120. Upstream idea: pyb2d3 should raise on an invalid new world itself.
- Later: auto-saved checkpoints (IndexedDB) so a restart resumes the last policy.

## Layout
```
pixi.toml                  envs: default (native: pyb2d3, numpy, pyopengl, imgui-bundle, pytest),
                           pytorch, jax; web (build page), web-test (playwright-python)
webgym/
  core.py                  Env base (batched), Box/Discrete spaces, autoreset per agent
  envs/cartpole.py, ...    one file per env, registry dict `ENVS`
  policies.py              numpy MLP: flat params <-> weights, batched forward (N, obs) -> (N, act)
  algos/cem.py, es.py      numpy trainers, written as generators (see below)
  algos/ppo_torch.py       torch PPO (lazy import)
  algos/jax_*.py           jax trainer (lazy import)
  app.py                   SHARED app: env + trainer + frame budget, camera, GL draw (pyb2d3 GL
                           batches, ghost tint), mouse drag, reward curve as GL overlay lines,
                           params/commands schema, code reload. API: app.frame(w, h),
                           app.event(kind, x, y, ...), app.command(name, value), app.state()
  host_native.py           thin: imgui-bundle window -> GL context, events, widgets from schema;
                           watches source files and reloads on change (same reload path)
  host_web.py              thin: set_canvas(OffscreenCanvas), rAF loop, events/commands from
                           postMessage, stats/schema back to main thread
  gl/                      copied from pyb2d3_sandbox_opengl (draw, shader, shaders, debug_draw_gl)
  checkpoints.py           save/load .safetensors (weights + metadata + optional trainer state)
  sources.py               load env/algo/weights from path or URL (.py / archive / checkpoint)
  cli.py                   headless train: `pixi run train cartpole --algo cem` (app without GL)
web/
  build_page.py            adapted from pyb2d3/examples/web/build_page.py (+ webgym incl.
                           webgym/gl in the core archive; torch/jax only as pinned channel URLs
                           + sha256 in config.json, never deployed)
  package.json, vite.config.js   pyjs + monaco-editor (+ vite-plugin-pwa) deps, service worker
  index.html, main.js      UI thread: canvas.transferControlToOffscreen(), controls, chart, events
  worker.js                pyjs init, archive unpack + IndexedDB cache, preload, message bridge
  editor.js                from pyb2d3, fs/exec calls replaced by messages
  test_page.py             adapted playwright test
tests/                     every env N random steps; every algo a few iterations; API checks
```

## Key designs
- **Shared native/web code:** everything except the hosts lives in `webgym` and is identical in
  both: envs, algos, app (simulation, training, rendering, interaction, reload). Hosts only
  provide a GL context, a frame tick, raw events and a widget renderer. Controls are a small
  python schema (sliders / dropdowns / buttons with values) that imgui (native) and an HTML
  builder in `main.js` (web) both render generically; values go back as `app.command(...)`.
  The reward chart is drawn by the app in GL, so it looks the same everywhere. The same
  playwright/pytest tests drive `app.event`/`app.command` headless. Browser-only extras: the
  Monaco editor and file open/save (natively you use your own editor + file watching).
- **Env API:** `env = make("CartPole")`; `obs, info = env.reset(seed)`;
  `obs, rew, term, trunc, info = env.step(action)`; spaces; `env.world` exposed for drawing and
  mouse interaction. Fixed dt + substeps per step, snapshot-based reset.
  `vec = make_vec("CartPole", n=64)` -> generic `VecEnv` (arrays of length n, per-env autoreset),
  or the env's own batched class if it registers one. Trainers only see the vec interface.
- **Trainers are generators:** `def train(env, policy, **hp): ... yield stats` (one yield per
  env step chunk or iteration). The runner advances the generator until the frame budget
  (~10 ms) is used, then draws. Training code stays a plain readable loop, is user-editable,
  and never blocks the frame loop. Same runner natively (headless: no budget, no draw).
- **What the trainer sees vs. the view:** ES/CEM evaluate the population in the visible world,
  so the user literally watches the candidates being evaluated. "Fast mode": draw every k-th step.
- **Worker messages:** main→worker `{event: pointer/key/wheel}`, `{resize: w,h,dpr}`,
  `{cmd: play/pause/reset/speed/set_env/set_algo/set_hp}`, `{code: {path: src}}`;
  worker→main `{status, progress}`, `{stats}` (iteration, mean/best reward -> chart), `{error:
  traceback}`, `{ready}`, `{files}` (sources for the editor).
- **Interaction:** mouse drag of bodies (pyb2d3 `b2d.MouseDrag`, motor joint), pan, wheel zoom
  — pull a cart while it balances. Messing with the env is for **watch mode** (a policy
  controls the env, no training running): disturbances must not pollute training rollouts, so
  during training the pointer only pans / zooms.
- **Small drawing buffer:** the canvas buffer is capped (~1 MP, `?dpr=` overrides) and
  stretched by the browser: drawing cost does not grow with the window size.
- **Code editing / upload:** editor tabs for the env file and the algo file; "Run" writes them to
  `/user/` in the worker FS, reloads the modules (`sys.dont_write_bytecode = True`), restarts
  training. "Open…"/"Save" via File System Access API (`showOpenFilePicker`, keep the handle to
  save back; `<input type=file>` + download as fallback).
- **Fetching code / models from sources (shareable links, milestone 7):** query params name sources, e.g.
  `?env=<url>&algo=<url>&weights=<url>&seed=3&hp.lr=0.01`. A source is a `.py` file, a
  `.zip`/`.tar.gz` package, or (later) a GitHub/gist shorthand; weights are `.safetensors` checkpoints
  (see Checkpoints). One shared python loader `webgym/sources.py`: `load(spec)` accepts
  a local path or an http(s) URL, writes into `/user/` (web) or a cache dir (native), imports
  it; modules register envs / algos via `@register_env` / `@register_algo`. Natively the same
  specs work as CLI args (`pixi run view --env <url-or-path>`). Fetched sources open in the
  editor. Before running code from another origin the page shows the URL and asks to confirm
  (a shared link runs arbitrary python in the visitor's tab). Fetching needs CORS (raw GitHub
  and gists allow it).
- **Web tooling: vite** (nodejs from pixi, `npm` only for JS deps):
  - pyjs from npm (`@derthorsten/pyjs`, assets via its `copy-assets`) instead of downloading
    the tarball in build_page; Monaco from npm (vite bundles its workers) instead of the
    cdnjs AMD loader; the module worker via `new Worker(new URL("./worker.js",
    import.meta.url), {type: "module"})`.
  - hashed asset file names for free (replaces the manual `?v=hash`), dev server with reload.
  - `build_page.py` shrinks to building the python archives + `config.json` into `public/`;
    `pixi run web-build` = build_page + `vite build`; `pixi run web-dev` = `vite`.
- **Service worker for caching** (vite-plugin-pwa / workbox, or ~50 lines by hand if simpler):
  cache-first in Cache Storage for everything immutable — hashed JS/wasm/archives, pyjs
  runtime, and the sha256-pinned torch/jax package URLs from prefix.dev / conda-forge (CORS
  responses are cacheable); network-first for `index.html` / `config.json`. So the second
  visit needs no network for big files, and the worker code just uses plain `fetch`. Old
  hashed entries are dropped when a new `config.json` arrives.
- **Loading (first visit fast, second visit much faster):**
  - **in the deployment** (like pyb2d3): one pre-packed `core` archive (numpy, openblas,
    pyopengl, pyb2d3, webgym sources, ...) with content hash, no solver in the browser,
    `?v=hash` on modules, `config.json` no-cache (from pyb2d3's build_page).
  - **NOT in the deployment: pytorch and jax** (and their compiled deps like jaxlib, scipy).
    They are fetched at runtime directly from the emscripten-forge channel
    (`repo.prefix.dev/emscripten-forge/6x`, noarch deps from conda-forge).
    `build_page.py` resolves them at build time and writes the exact package URLs + sha256
    into `config.json` (pinned, no solver in the browser); the worker downloads and
    extracts those conda packages. Fallback / simplest first version: pyjs
    `installPackages(["pytorch"])` (mambajs solve in the browser) — compare speed.
    Retry on prefix.dev CORS hiccups.
  - the service worker removes the network from the second visit; what's left is unpack +
    wasm compile + import. Add the IndexedDB cache of *unpacked* files (from pyb2d3 main.js)
    only if measurements show unpacking matters (likely for torch / jax). Request
    `navigator.storage.persist()` so the big packages are not evicted. Check whether mambajs
    already caches when `installPackages` is used.
  - no shared-lib preload in the worker (synchronous wasm compile is allowed there);
    `.so`s compile on import, so torch/jax are imported lazily, only when their algo is used.
  - measure the second-visit breakdown (fetch / unpack / wasm compile / import) per archive
    early (milestone 0) and optimize the biggest part (e.g. compile cache, skipping files of
    the torch package that are not needed at runtime, like the Inductor sysroot, when
    extracting).
  - **progress bars:** one for the startup (phases: pyjs runtime, core archive download /
    cache, unpack, import, first frame; bytes-based where possible, like pyb2d3 main.js) and
    one per big package (torch, jax) with size, "downloading x% (a / b MB)" / "cached" /
    "unpacking" / "importing" / "ready". The worker posts `{progress: {id, phase, done,
    total}}`; main.js renders them. Startup bar hides once the env is shown; package bars stay
    in a small "packages" panel.

## Reuse (copy into this repo, then adapt — don't rewrite)
Everything below is **copied** into webgym (self-contained, no dependency on a sibling
checkout), with a one-line comment naming the origin file. Only fixes that belong to pyjs or
the pyb2d3 *bindings* go upstream (webgym branches) and arrive as packages.
- `pyb2d3/examples/web/build_page.py` (download from channel, strip, pack, hash, config.json),
  `main.js` lines ~30-100 + 153-174 (IndexedDB cache, download, unpack, shared-lib preload) ->
  `worker.js`; `editor.js`; `test_page.py`.
- `pyb2d3/examples/pyb2d3_sandbox_opengl/` `draw.py`, `shader.py`, `debug_draw_gl.py`
  (`GLDebugDraw`, `Camera`, `BatchDebugDraw` path) — DOM-free, usable in the worker.
- `webgl_frontend.py`: `create_webgl2_context` (135-159), `_CANVAS_EVENTS_JS` event tuple
  (172-208) and `_on_canvas_event` (540-586) as the event format; camera methods (594-629).
- `pyb2d3_sandbox` MouseDrag (motor joint drag) and `FrontendBase` fixed-step logic as
  reference for `app.py` (not the sample/frontend classes themselves: the app is the shared
  layer, hosts stay thin).
- pyjs: `set_canvas(OffscreenCanvas)` (`module/pyjs/core.py:420`),
  `request_animation_frame_loop` (`core.py:370`), `unpackArchive` / `installPackages`
  (`npm/src/runtime.ts`).
- The GL drawing files go to `webgym/gl/` (draw.py, shader.py + shaders, debug_draw_gl.py), so
  the same code is imported natively and in the core archive.

## Upstream changes (branches `webgym` in pyb2d3, pyjs, recipes)
Fix things where they belong instead of working around them in webgym:
- **pyjs:** worker + OffscreenCanvas path of the npm package (e.g. the `specialHTMLTargets`
  guard in `include/pyjs/pre_js/canvas.js`), a worker test in `tests/npm`.
- **pyb2d3:** only binding-level needs of the envs (only if measured necessary). The GL drawing
  is copied into webgym and adapted there (ghost tint/alpha etc.).
- **recipes:** rebuild pyb2d3 / pyopengl packages if the above changes need it (wasm32 + wasm64).
- **New compiled package if needed:** if profiling (native + wasm) shows a python/numpy hot
  path is too slow (e.g. stepping many env instances + reading observations, rollout loops),
  move it into compiled code: either into the pyb2d3 bindings or a small new package
  (C++/nanobind) with a recipe on emscripten-forge (wasm32 + wasm64) and conda-forge (native).
  Python stays the readable, editable layer; compiled code only for measured hot spots.
- **recipes: jaxlib vs. iree-base-compiler (found 2026-11-17):** both ship their own MLIR
  python bindings; in the emscripten builds they register into the same nanobind domain,
  so importing both aborts python ("Critical nanobind error: refusing to add duplicate key
  NotSpeculatable ... Speculatability" -> Aborted() -> wasm `unreachable`). Natively (PyPI)
  they coexist. Fix upstream: build one of them with its own `NB_DOMAIN`. Workaround in
  webgym: the IREE compiler runs in a separate compile worker (jax only in the main one).
webgym then builds its page from local builds of these packages (local channel first) until
they are published.

## Milestones
0. **Spike (risk first):** worker + npm pyjs + OffscreenCanvas + PyOpenGL WebGL2 + rAF, drawing
   a falling-boxes pyb2d3 world. Watch the `specialHTMLTargets` guard in pyjs `canvas.js`
   (if not linked, `"#canvas"` is silently unregistered) and the pyb2d3 + pyopengl packages for
   pyjs 4.4.x / py314. Also: fetch torch and jax from the channel in the worker while the world
   renders, import, run a tiny forward/backward; record sizes and cold vs. warm (second
   visit) timings.
1. **Native core:** pixi workspace, `core.py`, CartPole, generic `VecEnv`, numpy MLP, CEM;
   headless train solves CartPole; tests. Benchmark VecEnv vs a one-world batched CartPole
   (native + wasm) and keep the batched one only if clearly faster.
2. **Shared app + native host:** `app.py` + `host_native.py`, live training with ghost
   population, schema widgets, file-watch reload.
3. **Browser page:** build_page, `worker.js` + `host_web.py`, startup + package progress bars,
   schema-driven HTML controls,
   playwright test (cold + cached reload, load times logged).
4. **Editor + files + checkpoints:** Monaco, run/reload, File System Access open/save,
   query params for built-ins (`?env=CartPole&algo=cem&seed=3&hp.lr=…`). Checkpoints panel (gallery, auto-save
   timeline in IndexedDB, user saves, watch / resume / download); built-in gallery trained
   natively by a `pixi run gallery` task.
5. **More envs:** lander, swarm, walker (+ OpenAI-ES / numpy PPO as needed).
6. **torch + jax trainers:** torch PPO (`set_multithreading_enabled(False)`), jax trainer
   (e.g. jit-compiled ES or PPO); same generator interface, same live view; natively too.
   Can start right after milestone 3 (independent of 4/5).
7. **Bring your own method (last):** workflow: develop an env / algo natively, train, publish
   code + `.safetensors` somewhere (GitHub repo, gist, Hugging Face), open our deployment with
   `?env=<url>&algo=<url>&weights=<url>` -> fetched, shown in the editor, confirmed, run.
   Pieces: `sources.py` (path / URL, `.py` or archive, register decorators), the same specs as
   native CLI args, a `pixi run export` that writes code + checkpoint into one folder and
   prints the share link, "copy share link" in the page, a short how-to doc.

## Verification
- `pixi run test` — pytest: envs (random actions, shapes, determinism with seed), algos (few
  iterations), CartPole CEM reaches max reward within a time limit.
- `pixi run train cartpole` headless prints reward curve; `pixi run view cartpole` opens the
  native window with live training.
- `pixi run -e web-test web-test` — builds page, playwright chromium (swiftshader), fails on
  console errors / stuck status, checks a training-stats message arrives and reward increases;
  `--screenshot` for a visual check; a run with `?env=<served test env .py>&algo=...` checks
  source fetching; a second (cached) load must be clearly faster than the cold one, with
  torch/jax archives coming from the cache. Manual check in Chrome / Firefox / Safari (worker rAF,
  OffscreenCanvas WebGL2).
