we want something like https://gymnasium.farama.org/ https://github.com/Farama-Foundation/Gymnasium
but it should work **in the browser** via emscripten-forge / pyjs

There is a handover from two important sessions that worked on important projects related to this prohect

* RL_GYM_HANDOFF.md 
    is a document from another session that focused on running pyb2d3 (our python bindings for box2dm, that also run iun wasm0)

* HANDOVER_browser_gym.md
    is a handover from a session with knowledge about pyjs, recipes and pyb2d3


Read them carefully.





Goals:
    * the focus here is having something impressive in the browser.
    * it should a) showcase the impressive things we can do with emscripten-forge
    * but also be fun
    * therefore we focus on box2d environments (at the moment)
    * it should also work outside the browser st. we can iterate on the native version

    * we should focus on things that can be trained very fast, st. we can show a "live view" of 
    the current state, ie how is the agent currently doing, ie we see live what the agent is doing to the world

Things we need to justify:
    * we somewhat need to justify running python in the browser,so the python code should be editable in the browser.
    * we should add an option to upload python code for the environments (the thing we want to optimize/controll/learn)
      and also the code for solving /training these models.
    * this upload could be done with nice file-system-apis that open files from a machine


Technical considerations:
    * what runs in main thread / what runs in a worker.
    * if we can use pyopengl also on an offscreen canvas, it would be ideal to run box2d / the environments
    * if we want to allow the human to interact with the world in the offscreen canvas, we need to "pipe" event into the workers.
    * should the optimizer and the model live in the same worker or each in differnt works and communicate via messages?
    * what algorithms are fast enough
    * how can we make loading the page fast / how can we cache things optimal


## Facts (state 2026-11-18)

### What it is
- A batched RL playground on pyb2d3 (box2d 3). The same python package `webgym` runs
  natively (tests, headless CLI) and in the browser (pyjs 5 in Web Workers, WebGL2 on an
  OffscreenCanvas via PyOpenGL). Deployed to GitHub Pages from `main` (`.github/workflows/pages.yml`).
- Envs (`webgym/envs/`, registered with `@register_env`): CartPole, LunarLander, Walker,
  Crawler, Car (was Tank; an unknown env / algo in the URL falls back to the default with
  a log note, a checkpoint of one is refused), DoubleCartPole (CartPole with a second pole on
  top; replaced the swing-up Pendulum), WaterCannon (the CarDesigner was dropped 2026-11-19: open loop, one design
  action, nothing to watch live; it didn't fit). **Users read and edit these files** (Code panel): keep them short and plain.
  An env builds ONE agent (`build_agent(i)` returns its parts by name; `Env` turns each into
  a batch over all agents: `self.cart` = `b2d.Bodies`, `self.joints` = `b2d.RevoluteJoints`,
  lists of k per agent -> n * k entries), then works batched: `randomize(rng)`, `act`,
  `observe()` (a list of columns), `reward()`. Spaces follow from `obs_names` /
  `action_names` (actions always in [-1, 1]). The how-to is the docstring of `core.py`;
  CartPole is the template. Walker and Crawler share `Legged` (`envs/walker.py`; joint motors
  via `RevoluteJoints`, feet sensors). Every env advertises discrete actions for DQN
  (`discrete_actions` + names; the walkers: bang-bang per joint / joint pair). Also `keys` +
  `human_action` (keyboard play), `camera_target` ("Camera follows").
- Env settings (`Env.settings = {name: (default, choices, help)}`, e.g. the WaterCannon's
  `balls`): selects in the Environment section; a change rebuilds the env as a new run;
  obs / action names may depend on them (properties). Runs and checkpoints are kept per
  env key ("WaterCannon · balls 2", `core.env_key`; checkpoints carry `settings`; the page's
  IndexedDB uses the key). A checkpoint whose policy doesn't fit the env (other sizes) is
  refused with a message.
- `batchable = False` (Car: it pushes loose crates and gravel, `Env.prop`): `make(name, n > 1)` returns
  `Separate`, n one-agent envs in their own worlds behind the batched API (at most
  `MAX_SEPARATE` = 32); the view shows one agent; while training the leader's world is drawn.
- Decoration (looks only): `make(name, decorated=True)` builds it (e.g. the lander's flags,
  which collide with nothing, as in the classic lander) and `Env.decorate()` runs after each
  watched step (the lander's exhaust: pyb2d3 emitters, `pyb2d3.extensions.RadialEmitter`).
  Only the watched env is decorated, and only where something is drawn (`App(draws=False)`
  in the training worker) and the "Decoration" checkbox is on (`App.show_deco`); training
  envs never are.
- Algorithms (`webgym/algos/__init__.py`, `ALGOS`): CEM and ES (numpy), PPO and DQN (pytorch).
  Each declares its parameters (shown with help texts) and `requires` (pytorch: loaded in the
  background in the browser). Trainers are generators: yield None per env step, stats per
  iteration; `state` resumes a run.
- **Decided 2026-11-18:** pytorch only, no jax, no GPU (the networks are tiny: PPO on CartPole
  has 387 parameters; the GPU only paid off for the update itself). The jax / WebGPU / IREE
  code, its reproducers and the Hopper env are in commit `bb4c07a` (see REPORT_BUGS.md,
  REPORT_UPSTREAM_WEBGPU.md). "Throw things at it" was removed, to be redone differently.

### Code layout
- `webgym/core.py`: `Env` (n agents in ONE box2d world, collision filters keep them apart;
  batched numpy API; snapshot reset; building blocks `scenery`, `prop`, `body`, `shape`,
  `sensor`), `Separate`, `make(name, n)`. Needs pyb2d3 >= 1.2.0 (emitters, joint batches;
  more than 128 worlds raise).
- `webgym/app.py`: the shared `App` (watch / train modes, controls schema in two sections
  "environment" / "training", `command()`, `event()`, checkpoints, the network view, the code
  view + `set_code` = write file + reload module).
- `webgym/host_web.py`: the browser host (roles "single", "display", "trainer"; async train
  loop in 0.25 s chunks; autosave; snapshots of the trainer's policy for the display).
- `webgym/policies.py` (numpy MLP, flat params; every trained policy is exported to it, so
  watching never needs torch), `checkpoints.py` (safetensors in pure numpy), `gl/` (drawing,
  copied from pyb2d3's sandbox), `cli.py` (`python -m webgym.cli CartPole --algo cem`).
- Page layout (`web/index.html`): a top bar always in view (Train / Stop / New run:
  controls with `"section": "bar"` in `App.controls()`; live numbers as chips, the first
  one the env key; Code, the "powered by emscripten-forge" badge (`public/emscripten-forge.svg`),
  Tour, Help, About (= `help.html#about`, About + Credits: drafts with TODOs for the author),
  GitHub), the left panel with four tabs (World: the env select,
  its settings, its description (the module docstring, unwrapped), the watching options /
  Training: algorithm, parameters, population, simulation workers / Policy: network +
  checkpoints / Log: state, packages, log; the tab is remembered), the world with the training progress plot in a strip below it (closable,
  remembered). First visit: a short tour (`web/tour.js`; not under automation,
  `navigator.webdriver`; `?tour=0`; the Tour link restarts it).
- Edited code (the code panel) is kept in this browser (`web/edits.js`, IndexedDB
  "webgym-code", per file path + webgym version = the sources archive's hash): every worker
  start (display, trainer, helpers) writes the stored files over the originals before
  importing webgym (`worker.js`, skips files that don't compile); Revert forgets an edit.
- Pretrained runs: `python -m webgym.cli Walker --iterations 300 --save
  web/public/pretrained/Walker-cem.safetensors` (trains with the headless App: a page
  checkpoint); `build_page.py` writes `public/pretrained/index.json` from the files'
  metadata; the Policy tab lists them as "pretrained" (restore only) for their env key +
  algo. Shipped: Walker / CEM, 300 iterations (mean 148, best 227).
- `web/`: `index.html`, `main.js` (UI, two workers, compiles the pyjs runtime once with
  `compileRuntime` and passes it to the workers, crash restart), `worker.js` (pyjs init with
  pinned packages, message bridge), `arch.js` (wasm32 / wasm64 choice), `checkpoints.js`
  (IndexedDB, per env + algo, autosave + named saves), `editor.js` (CodeMirror 6 + local file
  open / save), `network.js`, `plot.js`, `build_page.py`, `test_page.py`, bench page.

### Browser specifics (pyjs 5.1)
- pyjs 5.2.0 from npm (`web/package.json`). Our content-hashed sources archive is passed with
  `cache: true`: after the first visit pyjs serves it (unpacked + bytecode) from its cache.
- Rule (pyjs 5.2: workers no longer save bytecode by themselves): every
  worker calls `await pyjs.saveBytecode()` after each group of heavy imports. `worker.js`
  does it after the startup imports (host start) and after the pytorch install + check; a
  future import group (a new package, a user install) needs its own call. Keep
  `sys.dont_write_bytecode` off. (HANDOVER_PYJS_5.2.0.md)
- `web/build_page.py` pins packages per architecture with pyjs' lock CLI (no solver in the
  browser): "core" (numpy, pyopengl, pyb2d3; installed at startup) and "pytorch" (the rest,
  installed in the background). The webgym sources are a small content-hashed archive
  unpacked at /app. All written to `web/public/config.json`. pyjs caches packages (installed
  layout + bytecode) in Cache Storage.
- Architecture: wasm64 where the browser has memory64 (Chrome, Firefox), else wasm32 (Safari);
  `?arch=wasm32|wasm64` forces one. Both are tested.
- Two workers on desktop (display: rendering + watch; trainer: trains, sends policy
  snapshots), one on small devices (`?workers=1|2`).
- Simulation helpers (desktop, CEM / ES only, `Algo.parallel`): more workers (role "helper",
  core packages only) run a share of the population (`webgym/helpers.py`: the trainer's
  `Pool` posts "evaluate" requests through the page, the helper answers "evaluated";
  `rollout` in `algos/cem.py` runs the local share, then yields "waiting" until the
  answers are in; no answer in 60 s: those candidates count as the worst). Slider
  "Simulation workers" (0 .. cores - 2, default half, remembered; `?helpers=N`); started
  once the trainer is ready. Measured: Walker, 20 s: 20 iterations alone, 72 with 3.
- Query params: `env`, `algo`, `population`, `seed`, `train=0` (no autostart), `workers`,
  `arch`, `dpr`.
- Crash handling: a wasm trap in a worker -> the page restarts it (autosave restored); more
  than 3 crashes a minute -> stop. Python exceptions are only reported.
- Measured (headless chromium, wasm64): page running after ~3.2 s cold, ~0.9 s on reload
  (startup bytecode cached); pytorch ~8 s on a first visit, ~2.4 s from the cache.

### Commands
- `pixi run test` (native tests); `pixi run -e ml python -m pytest tests` (with pytorch).
- `pixi run web-dev` (npm install + build_page + vite dev server; a change of webgym/*.py
  re-packs the sources (`build_page.py --sources`, keeps the pinned packages) and reloads
  the page, see the plugin in `web/vite.config.js`).
- `pixi run -e web-test web-test` (build + playwright: load, train, checkpoints, crash restart).
  More: `python -u web/test_page.py --packages` (pytorch + PPO), `--query arch=wasm32`.
- Never commit or push: the user does that.

### Related repos / docs
- pyb2d3, pyjs, recipes: checkouts next to this repo, branch `webgym` each.
- PYB2D3_NOTES.md (what webgym wanted from pyb2d3; answered in PYB2D3_WEBGYM_REPLY.md and
  pyb2d3's HANDOVER_WEBGYM_NEXT_RELEASE.md: all in pyb2d3 1.2.0 (native + emscripten-forge),
  used by webgym: joint batches, sensors, emitters, worlds beyond 128 raise),
  REPORT_PYJS_CACHING.md (done in pyjs 5:
  HANDOVER_PYJS_5.1.0.md), ENV_IDEAS.md (car designer and more), gpu_blog.md (the author of
  the emscripten-forge torch / jax / IREE ports; unpublished, do not commit).