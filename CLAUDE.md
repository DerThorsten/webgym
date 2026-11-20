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
  Crawler (the Car (was Tank) was dropped 2026-11-19: its pretrained runs were poor; it is
  in git history before that; an unknown env / algo in the URL falls back to the default
  with a log note, a checkpoint of one is refused), MultiCartPole (setting "poles" 2..5, 2026-11-20;
  replaced DoubleCartPole (which replaced the swing-up Pendulum) and a short-lived
  TripleCartPole; 4 / 5 poles get a little hinge friction between the poles,
  `HINGE_FRICTION` per pole count, as little as works (body angular damping barely
  helped); suggested per pole count (`suggested` is a property): CEM up to 4, PPO for 5;
  pretrained per pole count, `pretrain.py SETTINGS`), Ragdoll (pyb2d3's ragdoll as a walker; its main
  body `torso`; feet on ankles, every one of its 11 joints a motor (action) and observed, plus
  the heels' heights; limp motors at action 0, so it falls untrained), WaterCannon (the CarDesigner was dropped 2026-11-19: open loop, one design
  action, nothing to watch live; it didn't fit), Blob (2026-11-20: a ring of 10 balls on
  springs, own collision group per blob, 5 actuated springs across, rolling losses
  (friction / rolling resistance / angular damping, else it rolls on by itself), an
  air-time bonus; springs drawn as lines via `Env.draw`), Drone (2026-11-20: 2 rotors,
  10 gates with solid posts, stall end after 3 s without getting closer; PPO learns it,
  CEM slowly). Pool and Slingshot (unbatched, a tower of props) were tried and dropped
  2026-11-20. LunarLander is a port of Gymnasium's (2026-11-20: legs on motor springs,
  engine impulses with dispersion, landed = asleep; the surface is fixed; Gymnasium's
  heuristic lands 63/64, mean 281; its sensors weigh nothing, `core.sensor(density=0)`;
  the other envs' sensors keep density 1, their runs were trained so). Third-party
  notes: THIRD_PARTY.md. **Users read and edit these files** (Code panel): keep them short and plain.
  An env builds ONE agent (`build_agent(i)` returns its parts by name; `Env` turns each into
  a batch over all agents: `self.cart` = `b2d.Bodies`, `self.joints` = `b2d.RevoluteJoints`,
  lists of k per agent -> n * k entries), then works batched: `randomize(rng)`, `act`,
  `observe()` (a list of columns), `reward()`. Spaces follow from `obs_names` /
  `action_names` (actions always in [-1, 1]). The how-to is the docstring of `core.py`;
  CartPole is the template. **Every env file is self-contained** (decided 2026-11-20: the
  Code panel shows only that file): no env imports another env, no env base classes (the
  `Legged` base of Walker / Crawler / Ragdoll was removed; each repeats its ~15 lines of
  reward / stall rule). Shared helpers are plain functions in `core.py`: `ruler`,
  `draw_ruler_numbers`, `bang_bang`. Joint motors via `RevoluteJoints`, feet sensors. Flat ground (bumpy terrain for the walkers was tried and removed 2026-11-19). Agent bodies
  never sleep (`core.body`: motors don't wake bodies). Watched episodes: a new seed each; the next starts
  when agent 0's ends (the ghosts don't hold it back); another run
  (new run, other algorithm / network, a restore: `App.run` in the snapshot) starts a new
  watched episode at once with the new policy; the same run trained on: the newest
  snapshot takes over with the next watched episode (`App._pending`; an episode = one
  policy, the "watching: iteration N" chip); the trainer sends a snapshot on every
  algo / hp / env command. `Env.max_return` (known best, e.g. CartPole 500; None:
  unknown): "mean x / 500", "perfect!". Env settings also from the URL (`?poles=4`). Most envs advertise discrete actions for DQN (not
  the Drone, the Ragdoll and the WaterCannon: DQN learned nothing / too little there, 2026-11-20; it does on
  the Blob, all 32 combinations of its 5 springs)
  (`discrete_actions` + names; the walkers: bang-bang per joint / joint pair). Also
  `camera_target` ("Camera follows"). No keyboard play (removed 2026-11-19; R = reset stays).
- `Env.suggested` = {budget: {algo: params}} for "quick" / "medium" / "long" (~1 / 5 / 20 min),
  best first; guesstimates, not measured. The "Time budget" select (Training tab) and opening
  an env pick the first one with its params (`App.suggest`); ★ marks them in the algorithm
  select; an algo given explicitly (URL `?algo=`) is kept.
- Env settings (`Env.settings = {name: (default, choices, help)}`, e.g. the WaterCannon's
  `balls`): selects in the Environment section; a change rebuilds the env as a new run;
  obs / action names may depend on them (properties). Runs and checkpoints are kept per
  env key ("WaterCannon · balls 2", `core.env_key`; checkpoints carry `settings`; the page's
  IndexedDB uses the key). A checkpoint whose policy doesn't fit the env (other sizes) is
  refused with a message.
- A batchable car (own crates + gravel per car via collision groups, filter joints between
  the car's parts, rays via `world.cast_ray` skipping other groups) was tried and dropped
  2026-11-19: no faster than separate worlds (0.6x driving: every ray meets all cars'
  pebbles). One world vs separate worlds, agent-steps/s, 32 agents: CartPole 24x,
  DoubleCartPole 25x, LunarLander 14x, Walker 11x, WaterCannon 6x, Crawler 2.4x.
- `batchable = False` (no env now; was the Car: it pushed loose crates and gravel, `Env.prop`;
  tested with CartPole marked so): `make(name, n > 1)` returns
  `Separate`, n one-agent envs in their own worlds behind the batched API (at most
  `MAX_SEPARATE` = 32); the view shows one agent.
- Decoration (looks only): `make(name, decorated=True)` builds it (e.g. the lander's flags,
  which collide with nothing, as in the classic lander) and `Env.decorate()` runs after each
  watched step (the lander's exhaust: pyb2d3 emitters, `pyb2d3.extensions.RadialEmitter`).
  Only the watched env is decorated, i.e. where something is drawn (`App(draws=False)` in
  the training worker: never); no option for it any more (2026-11-20).
- Drawing (2026-11-20): `Env.draw(debug_draw)` adds the env's own looks after the world
  (ruler numbers, the blob's springs as lines, the cart poles' hinge discs (agent 0 only,
  `draw_solid_circle`), the water cannon's `draw_liquid`).
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
- `webgym/host_web.py`: the browser host (roles "display", "trainer", "helper"; async train
  loop in 0.25 s chunks; autosave; snapshots of the trainer's policy for the display).
- `webgym/policies.py` (numpy MLP, flat params; every trained policy is exported to it, so
  watching never needs torch), `checkpoints.py` (safetensors in pure numpy), `gl/` (drawing,
  copied from pyb2d3's sandbox), `cli.py` (`python -m webgym.cli CartPole --algo cem`).
- Page layout (`web/index.html`): a top bar always in view (live numbers as chips, the first
  one the env key; Code, the "powered by emscripten-forge" badge (`public/emscripten-forge.svg`),
  Tour, Help, About (= `help.html#about`, About, About the author, Acknowledgments (QuantStack, Bloomberg), Credits; TODOs for the author),
  GitHub), the left panel: on top two rows of line icons (SVG, Feather style), the app's
  controls of section "bar", drawn by `showQuick` in main.js (the only place they are):
  `#quick` train (a barbell, a switch: blue while training; ▶/⏸ read as "pause the
  replay"; its title = the app's button label), new run, | reset
  episode, pause / continue (⏸ / ▶, Space: the watched world only, `App.paused`; training
  goes on), | switches auto reset, camera follows (disabled for fixed views), blue when on; `#quick-watch` steppers − / + for show policies and slower /
  faster for replay speed. No panel below the tabs any more. Then five tabs, in this order (World: the env select,
  its settings, its description (the module docstring, unwrapped) /
  Training: algorithm, parameters, population, simulation workers / Policy (see below) / Toys (`webgym/toys.py`,
  controls section "toys", commands "toy" / "rain" to both workers: a click on a dynamic
  body always grabs it (mouse joint); a click into empty space (`toys.empty`) uses the
  chosen tool of `toys.TOOLS` (box, ball, plank, explosion, ramp, step, seesaw; all dynamic,
  ramps & co. heavy ("heaviness") on the ground below the click), chosen with buttons (control type
  "tools") in the folds, a click on the active one puts it away ("grab"); the world's gravity (its own fold "Gravity",
  `toys.gravity`: "gravity x" sideways, "gravity y" down, both in the world's own g,
  -2.5..2.5 (y below 0: falls up), set on the watched world every step, only while the
  Toys tab is open, drawn as an arrow in the view's top right corner, a ring = 1 g,
  `toys.draw_gravity`, only while the Gravity fold is open (main.js `showGravity` sends
  "gravity_shown" to the display on a change only: an open fold fires its toggle again
  on every controls rebuild); not 1 g down: nothing sleeps; a "Reset gravity" button, command
  "gravity_reset"; a fold may hold buttons); weather (rain); sliders from `toys.PARAMS` in
  folds per `toys.GROUPS` (controls with "group" + "fold": `<details>` in main.js, open
  state kept in `openFolds`; commands "toy.<name>"); an
  explosion is drawn as growing rings + a flash (`toys.draw_blasts`, as pyb2d3's tank
  sample); opening the tab (command "toy_mode") or a toy turns auto reset off and shows
  one policy; R clears them: the snapshot reset drops later bodies; leaving the tab starts
  a new episode in the world as it was: no rain, the env's gravity, no toy on a click,
  auto reset on again; the
  settings stay for the next visit) / Policy: network +
  checkpoints / Log: state, packages, log; the tab is remembered), the world with the training progress plot in a strip below it (closable,
  remembered). First visit: a welcome card at once, while python loads (a status line
  "loading" → "ready"; the quick tour puts the steps on what python builds, the world
  select and the world, last; a step whose element python hasn't built yet waits:
  `setReady`, resolved on the first "controls") (`web/tour.js` `welcome()`: what it is, then
  Quick tour / Detailed tour / Help page / Skip; on phones a note "made for the
  desktop, may still work"; not under automation,
  `navigator.webdriver`; `?tour=0` skips it, `?welcome=1` always shows it; the Tour links
  start the tours again).
- Phones (CSS media queries in `index.html`: portrait `max-width: 700px`, landscape
  `max-height: 500px`): a compact bar (no logo; chips: training + mean;
  Tour / Help / About / GitHub in the ⋯ menu `#links`; the badge stays), the editor full
  screen (✕). Portrait: the world on top (55dvh), the tabs below. Landscape: the world
  fills the screen, the panel is a drawer (☰, `#app.drawer-open`; the tour opens it for
  its steps). Touch (`main.js`): one finger = the mouse; a second finger cancels its drag
  and pinches: "pan" (App.event: pan from here, never grabs a body) + "move" at the
  midpoint, "wheel" with the distance ratio (zoom about the midpoint).
- The release = `pyproject.toml`'s version (now 0.2.0), written into config.json
  (`release`, per arch) by build_page.py. Bump it when saved data no longer fits: a browser
  that ran another release (or webgym before releases existed: it has `webgym*` keys) gets a
  confirm() warning, OK deletes webgym's own storage only (IndexedDB "webgym" +
  "webgym-code", Cache Storage "webgym-packages*" (pyjs `cache: "webgym-packages"`),
  localStorage `webgym*`), Cancel keeps it and asks again. A first visit: no dialog.
- Edited code (the code panel) is kept in this browser (`web/edits.js`, IndexedDB
  "webgym-code", per file path + webgym version = the sources archive's hash): every worker
  start (display, trainer, helpers) writes the stored files over the originals before
  importing webgym (`worker.js`, skips files that don't compile); Revert forgets an edit.
- Pretrained runs: `pixi run -e ml python web/pretrain.py` trains one run per env and
  suggested algorithm (`webgym.cli --save --minutes`: the env's suggested params, the
  checkpoint of the best mean iteration) into `web/public/pretrained/` (not in git:
  `--upload` puts them on the GitHub release `pretrained-<version>`, `--download` gets
  them; the Pages workflow downloads before building). `build_page.py` writes
  `public/pretrained/index.json`; the Policy tab lists them named after their algo
  without "_torch" ("ppo"; "pretrained · <algo label>" below; restore only, fetched on click) for their env key, every algorithm, the best
  mean first (a restore switches algo + network). Limits per run: numpy 6 min, pytorch
  12 min (the minutes decide: the CLI continues a trainer that ends by itself, `--hp
  NAME=VALUE` sets algorithm parameters). Envs with runs per setting: `pretrain.py
  SETTINGS` (MultiCartPole poles 2..5, WaterCannon balls 1 / 2; files
  `<env>-<algo>-<setting><value>`, `--set poles=4` trains only those). WaterCannon has
  balls 1 or 2 (3 dropped 2026-11-20; squares were balls again the same day: setting
  "balls"; two: a wider cart, a cannon at each end). Its water (the current policy's
  drops) is drawn as a liquid: metaballs (`gl/draw.py` `GLLiquid`: soft blobs added up
  in a half-resolution offscreen texture, then a threshold pass `liquid.fs`; envs call
  `debug_draw.draw_liquid(positions, radius, color)` in `Env.draw`; the blob radius grows
  with the spray's spread); the ghosts' drops: plain faint balls. The look (2026-11-20):
  solid fills, darker outlines, a sky gradient (shaders), `color=` per body, ghosts drawn
  first (`Env.show_ghosts`, `draw_world(show=/hide=)`), sensors hidden, the walkers' ground
  a ruler with numbered metres (`core.ruler(env, start, end, depth)`,
  `draw_ruler_numbers(debug_draw, start, end)`; also the Blob, the Drone and the cart
  poles: there 6 m, -3..3, as the old ground). Later the same day: the world is flushed
  before `Env.draw`, so an env's own drawing (hinge discs, springs) lies on top; liquids
  per color (`GLDebugDraw.liquids`, they don't merge); the Toys rain drawn as a light cyan
  liquid (`toys.RAIN_WATER`, blobs 3.5x the drop, default drop size 0.03; rain off removes
  its drops). Shapes: an agent's polygons get rounded corners (`core.rounded`, in
  `Env.shape`: same outer size, radius `ROUND` = 4 cm or less; thinner than 4 cm, e.g. the
  cart poles' poles, stay sharp; scenery (rulers, posts, the lander's ground) is never
  rounded); leg parts are capsules whose round ends are centered on the joints (Walker,
  Crawler, Ragdoll legs; the lander's legs capsules too). WaterCannon: a nozzle of two
  flaps (capsules on motor hinges at the barrel's end) opens as wide as the spray; the
  water passes them. The runs of Walker, Crawler, Ragdoll, WaterCannon were retrained
  for these shapes.
  `--upload` doesn't remove assets that are
  gone locally: `gh release delete-asset`. After an env changes, retrain its runs and
  check them all: `pixi run -e ml python web/check_pretrained.py` (restores each as the
  page does, replays its policy on the current env vs its training mean).
  Help videos (2026-11-20): `pixi run -e web-test python web/help_videos.py [names]`
  (`SCENES`: pretrained runs, NEAT's first ragdoll episodes, the network view, the whole
  page; chrome's screencast frames -> ffmpeg (in web-test): 16:9, 960x540 mp4 + poster jpg
  + gif; one policy, no score card, each clip starts a new episode). Not in git: the
  release "videos" (`--upload`; the Pages workflow `--download`s mp4 + jpg into
  public/help/videos/; the README links the gifs on the release). help.html: `<video
  preload="none" poster>`, played while in view (IntersectionObserver).
  Help screenshots: `pixi run -e web-test python web/help_shots.py` (uses the pretrained
  PPO walker / ragdoll; the Toys scenes: ramp, explosion, rain).
- PPO and A2C scale the rewards (`ppo_torch.RewardScale`, as gymnasium's NormalizeReward
  in CleanRL / RL-Zoo): without it PPO failed on LunarLander (mean -88 vs 277 with it);
  about neutral on the other envs (3 min each, one seed). Default lr 0.003 (0.001 slower).
- `web/`: `index.html`, `main.js` (UI, two workers, compiles the pyjs runtime once with
  `compileRuntime` and passes it to the workers, crash restart), `worker.js` (pyjs init with
  pinned packages, message bridge), `arch.js` (wasm32 / wasm64 choice), `checkpoints.js`
  (IndexedDB, per env + algo, autosave + named saves), `editor.js` (Monaco, python only, loaded when the Code panel first shows; `window.codeText()` for tests; + local file
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
- Always two workers, phones too (display: rendering + watch, never trains; trainer:
  trains, never draws, sends policy snapshots). The single-worker mode (role "single",
  `?workers=1`, the App drawing the training population) was removed 2026-11-19: phones
  are not handled differently. On phones the
  left panel scrolls as a whole (the tab row sticky), nothing pinned below the tabs.
- Simulation helpers (CEM / ES only, `Algo.parallel`): more workers (role "helper",
  core packages only) run a share of the population (`webgym/helpers.py`: the trainer's
  `Pool` posts "evaluate" requests through the page, the helper answers "evaluated";
  `rollout` in `algos/cem.py` runs the local share, then yields "waiting" until the
  answers are in; no answer in 60 s: those candidates count as the worst). Slider
  "Simulation workers" (0 .. cores - 2, default half, remembered; `?helpers=N`); started
  once the trainer is ready. Measured: Walker, 20 s: 20 iterations alone, 72 with 3.
- Query params: `env`, `algo`, `population`, `seed`, `train=0` (no autostart; picking another world or a
  setting in the page always starts training, after restoring its autosave),
  `arch`, `dpr`, `tour=0`, `welcome=1`, env settings (`poles=4`), `checkpoint=NAME` (one
  parameter for all kinds alike, decided 2026-11-20: own saves of the env key first,
  "autosave" too, any algo, the newest; then pretrained runs by their name, the algo
  without "_torch": `?checkpoint=ppo`, kept short on purpose): restored at page
  load instead of the autosave, as by the user (`checkpoints.onControls` `pick`; the
  automatic training start is cancelled: it watches). A restore notes the run's env +
  algo first, so its algo switch doesn't restore that algo's autosave over it. All of
  them are documented for users in help.html#url ("URL options"): keep that list
  complete when adding one. help.html#pretrained links the best pretrained run of each
  non-trivial world (`?env=…&checkpoint=<algo>`; the user-facing text says "a
  pretrained run", not "best"): update it after retraining.
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