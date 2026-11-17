# Handoff: a gymnasium-like RL project on pyb2d3, native and in the browser

Goal of the new project: RL environments in the style of gymnasium
(https://gymnasium.farama.org/) built with **pyb2d3** (box2d 3.2 bindings), training with
PyTorch / JAX, and everything also running **in the browser** (pyjs + emscripten-forge).

This file collects what we learned building pyb2d3 and its sample sandbox (native OpenGL /
pygame and a browser WebGL page). The pyb2d3 repo (`~/src/pyb2d3`, v1.0.0) is the reference;
read its files when in doubt. Items marked *untested* are suggestions, not experience.

---

## 1. pyb2d3 in a nutshell (what an env author needs)

```python
import pyb2d3 as b2d

world = b2d.World(gravity=(0, -10))                       # thread_pool=b2d.ThreadPool() natively
ground = world.create_static_body(position=(0, 0))
ground.create_shape(b2d.shape_def(), b2d.box(20, 0.5))
body = world.create_dynamic_body(position=(0, 4))
body.create_shape(b2d.shape_def(density=1.0, material=b2d.surface_material(friction=0.6)),
                  b2d.circle(radius=0.5))
world.step(1 / 60, 4)                                       # time step, sub steps
body.position, body.linear_velocity, body.apply_force(...)  # single object API
```
- Free box2d C functions are bound as properties / methods of handle types (`Body`, `Shape`,
  `Joint`, `Chain`, `WorldView`); docstrings come from box2d's header comments.
- Factories: `b2d.body_def(...)`, `b2d.shape_def(...)`, `b2d.xxx_joint_def(...)` and
  `world.create_xxx_joint(...)`; geometry `b2d.circle/box/capsule/polygon/segment`,
  `b2d.chain_box`, `PathBuilder`.
- **Batch API (use it for observations / actions):** `b2d.Bodies([...])` (also `Shapes`,
  `Joints`) mirror the single-object API with numpy: `bodies.position` -> `(n, 2) float32`,
  `bodies.linear_velocity = arr` (or one value, broadcast), `bodies.apply_force(forces, points,
  wake)` per entry or broadcast, `bodies.destroy()`. One C++ loop instead of n python calls.
  Batch *creation* was measured and dropped (only pays off when creating bodies often).
- **Events / queries:** contact, sensor and body move events, ray casts
  (`world.cast_ray_closest(origin, translation)`, `cast_ray` with callback), overlaps
  (`overlap_aabb`) -> lidar-like observations, collision-based rewards / terminations.
- **Snapshots (important for RL):** `world.get_snapshot()` / `world.restore_from_snapshot(...)`
  and `b2d.World.from_snapshot(image)`: fast `reset()` to a stored start state and cloning
  worlds for vectorized envs, instead of rebuilding the scene. Also recordings / replays
  (`Recording`, `Replay`: step/seek/diverge detection) for debugging episodes.
- **Determinism:** fixed `time_step` + `sub_step_count`, no wall-clock dependence (the sandbox's
  real-time catch-up loop is for viewing only, see §3). box2d v3 aims for deterministic
  results; *verify* cross-platform (native vs wasm) equality before relying on it for
  reproducible seeds.
- **Threads:** natively opt-in `b2d.ThreadPool` per world; **none in wasm** (pyb2d3 is built
  with `PYB2D3_NO_THREADING` there). In the browser, many small worlds stepped in a loop will
  be single-threaded.
- **wasm SIMD is on** (box2d's contact solver uses simd128; verified in the emscripten-forge
  `box2d-static` package and in the linked pyb2d3 module).

## 2. Installing pyb2d3
- Native: `pip install pyb2d3` (abi3 wheels for CPython 3.12+: linux x86_64/aarch64,
  windows, macos arm64) or the dev setup with pixi (box2d from conda-forge).
- Browser: the **emscripten-forge** package `pyb2d3` (channel
  `https://repo.prefix.dev/emscripten-forge/6x`, platform `emscripten-wasm32`). pyb2d3 can
  build its own local package (`pixi run emscripten`, rattler-build, recipe in
  `emscripten_forge/recipe/`).
- Use pixi for the new project too (the user prefers it).

## 3. Sandbox architecture worth reusing (pyb2d3 `examples/`)
- `pyb2d3_sandbox` (core, frontend independent):
  - `SampleBase`: owns the world, `update(dt)` steps it, hooks `pre/post_update`,
    `pre/post_debug_draw`, `aabb()` (initial view), mouse drag of bodies via a motor joint,
    keys, `help_text()` from the docstring (8 lines max, enforced by a test).
  - `FrontendBase`: camera, events, pause / single step, sample switching, fixed `hertz`;
    `update_physics()` catches up with real time in fixed steps (capped by
    `max_steps_per_frame`); `update_physics_single_step()` is the deterministic step.
  - `@register_sample(category, name)` registry (import order = browser order),
    `run_sample_browser(start=...)`.
  - widgets: FloatSlider, IntSlider, Checkbox, Button, Dropdown, RadioButtons, TextInput,
    Separator, rendered by every frontend (imgui natively, HTML in the browser).
  - extra options: `get_extra_options()` (URL query params in the browser, `-o name=value`
    natively; `b64:` values decoded) -> e.g. `?env=Walker&seed=3`.
- Frontends: `pyb2d3_sandbox_opengl` (imgui-bundle + instanced GL shaders),
  `pyb2d3_sandbox_webgl` (same GL renderers via PyOpenGL into WebGL2 + HTML panel),
  `pyb2d3_sandbox_pygame` (simple; also renders **headless to PNG frames / video**:
  `sample_to_video.py`), `tests/headless_test_frontend.py` (no drawing).
- **Mapping to gymnasium** (*suggestion*):
  - The env must not depend on the sandbox: `Env.reset(seed)`, `Env.step(action)` ->
    `(obs, reward, terminated, truncated, info)`, spaces, a fixed physics step per env step.
  - `render_mode="human"`: a thin `SampleBase` subclass wrapping an env (+ a policy) so it runs
    in every sandbox frontend, native and browser, with widgets (speed, policy on/off, reset).
  - `render_mode="rgb_array"`: headless drawing; pygame's headless mode already produces
    frames; or the batch debug draw arrays (below) rasterized offscreen.
  - Vectorized envs: many worlds (snapshots to clone) or several agents in one world,
    observations read with the batch API.

## 4. Rendering: what mattered for performance
1. **Debug draw collected in C++**: `b2d.BatchDebugDraw` (`include/pyb2d3/batch_debug_draw.hpp`)
   records `world.draw(...)` into flat float32 arrays per primitive (circles, solid circles,
   capsules, polygons, points, lines) in the GPU instance layout of the GL shaders; one call
   per frame. Per-shape python callbacks (`b2d.DebugDraw` subclass) dominated frame time with
   thousands of bodies, much worse in wasm. Use the batch arrays for any custom renderer.
2. **Instanced GL rendering** (`pyb2d3_sandbox_opengl/draw.py` + `shaders/`): one draw call
   per primitive type, same code natively and in WebGL2 (PyOpenGL in the browser).
3. **PyOpenGL error checking off** (`OpenGL.ERROR_CHECKING = False` before the first
   `OpenGL.GL` import, done in `pyb2d3_sandbox_opengl/__init__.py`): otherwise every GL call
   costs a `glGetError` round trip.
4. Cache small conversions on hot paths (e.g. color -> RGBA8 per color).
5. `?dpr=1` on the page caps the drawing buffer resolution: tells whether a device is fill-rate
   bound.

## 5. The browser: how the pyb2d3 page works (`examples/web/`)
Files: `build_page.py` (builds the static page), `index.html`, `main.js`, `editor.js`,
`test_page.py` (headless test), `message_link.py`. Plain ES modules, **no bundler**.

Startup path (≈1–2 s to a running sandbox on a local server, headless chromium):
1. **pyjs self-hosted**: copied from the pinned npm tarball (`@derthorsten/pyjs` 4.4.1,
   python 3.14) into the page. Loading pyjs from jsDelivr gave 502/503s and a hanging
   "starting python…". Pin pyjs and the python version together: every packed package must be
   built for that python.
2. **No package install (mamba) in the browser.** `build_page.py` downloads the newest builds
   of the needed packages for that python from the emscripten-forge channel (repodata.json;
   needs a User-Agent header), strips what is not needed at runtime (tests, headers, docs,
   `.pyc`, `.a`, metadata, ...) and packs them with the local pyb2d3 build into
   `packages-<hash>.tar.gz`; the sandbox sources go into `examples-<hash>.tar.gz`
   (mounted at `/examples`). pyjs unpacks them (`pyjs.unpackArchive`).
3. **Cache busting:** content hash in archive names and `?v=<hash>` on the JS modules;
   `config.json` (lists archives, hashes, sizes, shared libs) fetched with `cache: "no-cache"`.
   Deterministic tar.gz (mtime 0, no owners) so unchanged content keeps its hash.
4. **IndexedDB cache** of the *unpacked* files, keyed by archive name + hash (reload ~1.5 s vs
   ~2.2 s cold). Falls back gracefully without IndexedDB (private windows).
5. **Preload shared libraries** before importing: `pyjs.loadDynamicLibrary(..., {loadAsync:
   true, nodelete: true, allowUndefined: true})`, plain libs first (global, by file name),
   then extension modules (by path). Browsers refuse to compile large wasm synchronously on
   import (libopenblas ~9 MB failed).
6. Progress bar by phases / bytes, status line, loading times logged to the console
   (`test_page.py` reads them).
7. Start script: `sys.dont_write_bytecode = True` (edited code is re-imported; a .pyc with
   the same mtime could win), import samples, `set_extra_options({...query params...})`,
   `run_sample_browser(frontend_class=WebglFrontend)`, which **returns immediately**: frames
   come from `pyjs.request_animation_frame_loop`.
8. Monaco editor (cdnjs) to edit a sample's python in the page and re-run (modules reloaded,
   registry classes replaced).
9. Mobile: panel becomes bottom sheet / drawer; pointer gestures (`gestures.py`).

### Consequences for torch / jax in the browser
- **Packages get much bigger**: packing + IndexedDB caching + async shared-lib preloading
  matter even more. Expect to extend `PACKAGES` in `build_page.py`; check every shared lib
  is preloaded (large `.so`s must not be compiled synchronously). Consider loading the ML
  stack lazily (second archive) so the env viewer starts fast. *untested with torch/jax*
- **Python runs on the browser main thread** (as in the pyb2d3 page): any long blocking loop
  (training) freezes rendering and input. Structure training as **work per animation frame**
  (k env steps / one optimizer step per frame) or move it to a **Web Worker** with its own
  pyjs and stream results / weights to the viewer. pyb2d3 only did the per-frame approach;
  workers are *untested*.
- No threads in wasm (pyb2d3 and probably the ML libs): vectorized envs run sequentially.
- WebGL drawing and physics share the frame budget: offer "train fast, draw every n-th step"
  and a pause / speed control.
- URL parameters (`get_extra_options`) are a cheap way to share runs: env, seed,
  hyperparameters, or a pretrained-weights URL.

## 6. Testing that paid off (copy the pattern)
- Every registered sample runs headless in the test suite, and every widget is driven through
  its values (`tests/test_samples.py`). For RL: every env runs N random-action steps headless,
  plus a gymnasium API check (`gymnasium.utils.env_checker.check_env`) if gymnasium is used.
- Real-window smoke test natively (`examples/opengl_smoke_test.py`).
- Browser test: `examples/web/test_page.py` with playwright in headless chromium
  (conda package **`playwright-python`**, browsers via `playwright install chromium` into
  `.pixi`), cold load + cached reload, fails on console errors / stuck status, options
  `--sample`, `--query`, `--screenshot`. WebGL without GPU: launch chromium with
  `--enable-unsafe-swiftshader --use-angle=swiftshader`.
- Docs CI builds the emscripten package, then the page, and deploys to GitHub Pages.

## 7. Sandbox UX the user liked
- Native and web look the same: accordion sections (sample tree open, controls open,
  simulation / debug draw collapsed, sample widgets open), icon-only control row at the top
  (prev / play-pause / step / reset / next) with tooltips, sample help in a fixed-height
  scrollable box, widget labels and separators, a "rebuild" button that keeps settings.
- hello_imgui: `runner_params.callbacks.default_icon_font =
  hello_imgui.DefaultIconFont.font_awesome6` (the default FA4 lacks icons).
- Samples: simple, interactive (drag / shake things), many particles with emergent behaviour,
  physics that actually uses the engine, short help texts, no literature citations.

## 8. Working conventions (from the user's CLAUDE.md files)
- Keep it simple, less code, no repetition; pixi for new projects.
- Shell: literal absolute paths, one plain command per call (no `$(...)`, `;`, `&&`, loops,
  shell variables); edit files only with Edit / Write (no `sed -i`, heredocs); temporary files
  in the session scratchpad; don't read outside the repo (ask instead).
- Format after edits, run the tests before calling something done; commit / push / tag only
  when asked.

## 9. Suggested start
1. pixi workspace: pyb2d3, numpy, torch / jax, gymnasium (if the API is followed), pygame for
   headless frames; reuse the sandbox packages from pyb2d3's `examples/` (copy or depend).
2. One env first (e.g. cart-pole or a 2D walker): pure env class, batch API observations,
   snapshot-based `reset`, fixed step; headless test with random actions.
3. A `SampleBase` adapter that shows an env + policy in the sandbox (native first).
4. Browser: copy `build_page.py` / `main.js` / `index.html` / `test_page.py`, add the env
   package and torch / jax to the packed packages, measure load time and frame budget early.
5. Training in the browser: per-frame chunks first; measure; then decide on a Web Worker.
