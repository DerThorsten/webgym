"""The shared app: envs, training, drawing, interaction; identical natively and in
the browser.

The browser runs two of them (host_web.py): the display worker's draws what you watch
(`frame(width, height)` once per animation frame, pointer input (and R) via
`event(...)`), the trainer's trains (`_train_steps`, never drawn) and sends its policy
to the display (`load_policy`). Both render `controls()` as widgets (values back via
`command(name, value)`) and plot `history`.

* watch: the env (one agent) driven by the current policy (zero actions before any
  training); drag its bodies with the pointer
* train: the trainer's population (one batched env, n agents in one world). Stop keeps
  the trainer's state: Train continues, "New run" starts over.
"""

import importlib.util
import json
import sys
import time

import webgym.gl  # noqa: F401 (first: switches PyOpenGL's per-call glGetError off)
from OpenGL import GL as gl

import numpy as np
import pyb2d3 as b2d

from webgym import checkpoints, toys
from webgym.algos import ALGOS, HELP
from webgym.core import ENVS, MAX_SEPARATE, SENSOR_COLOR, env_key, make
from webgym.gl.debug_draw_gl import Camera, GLDebugDraw
from webgym.policies import from_sizes

import webgym.envs  # noqa: F401 (registers the built-in envs)

TRAIN_STEPS = 256  # env steps per _train_steps call at most (training: always full speed)
REPLAY_SPEEDS = [0.25, 0.5, 1, 2, 4, 8]  # watch mode: x real time (slow motion .. fast)
END_SECONDS = 1.0  # watch mode: the world goes on this long after an episode ended
MAX_POPULATION = 256  # (one world per population: no box2d world limit)
FOLLOW = 0.08  # "Camera follows": the share of the way to the target per frame (smooth)


class NeedPackage(Exception):
    """the algorithm needs a package that is not loaded (pytorch): the browser
    loads it and trains then"""

    def __init__(self, package):
        super().__init__(f"{package} is needed (load it first)")
        self.package = package


class App:
    """see the module docstring"""

    def __init__(self, env="CartPole", algo=None, population=64, seed=0, draws=True, helpers=None, settings=None):
        # the algorithm: as given (e.g. the URL's), else the env's suggestion for the time
        # budget (Env.suggested; "quick" / "medium" / "long")
        self.algo = algo or "cem"
        self.budget = "quick"
        self._keep_algo = algo is not None  # (for the first env only)
        # simulation helpers (webgym.helpers.Pool, the training worker; None: none): run a
        # share of a CEM / ES population in more workers
        self.helpers = helpers
        self.waiting = False  # the trainer waits for the helpers' answers
        # decoration of the watched env (flags, thrust particles: looks only, never in
        # training): only where something is drawn (not in the training worker)
        self.draws = draws
        self.hp = {name: a.defaults() for name, a in ALGOS.items()}  # parameters per algorithm
        self.population = min(population, MAX_POPULATION)
        self.seed = seed
        self.watched = 0  # watched episodes so far (each gets its own seed)
        # the Signals tab: per watched step, agent 0's observations + action + reward (the
        # rows since the last state(), which sends and clears them)
        self.trace = []
        # agent-steps simulated by training here (+ the helpers': Pool.steps), ever growing;
        # the page shows the rate (steps/s) next to the iteration
        self.sim_steps = 0
        # watch mode: env steps per frame (x real time); `_replay_due`: steps owed
        self.replay_speed = 1
        self._replay_due = 0.0
        self.paused = False  # the watched world stands still (training goes on)
        self.run = 0  # +1 per new run / restore: the display starts a new episode then
        self.shown_run = None  # (display) the run of the trainer's last snapshot
        self._pending = None  # (display) its newest policy, for the next episode: (tensors, meta)
        # watch mode: a new episode when one ends; off: the env goes on after the end
        # (see the agent recover), "reset" / R starts a new one
        self.auto_reset = True
        self._auto_reset_before = True  # (playing yourself turns it off; restored after)
        # the camera follows the point the env suggests (e.g. the walker); off: fixed view
        self.follow = True
        # watch mode: what a click into the world does ("grab": drag bodies; "box" /
        # "ball": drop one; "explosion"), and rain (webgym/toys.py; None: no rain)
        self.toy = "grab"
        self.rain = None
        # toy mode (the Toys tab open, or a toy used): no scores (no time limit applies)
        self.toys_on = False
        self.gravity_shown = False  # the gravity arrow (the page: while its Gravity fold is open)
        self.toy_params = toys.defaults()  # (their sliders: size, weight, rain rate, ...)
        self.blasts = []  # explosions still drawn (toys.draw_blasts)
        self.camera = Camera()
        self.debug_draw = None  # created on the first frame (needs the GL context)
        self.panning = None  # world point under the pointer while panning
        self.frame_ms = 0.0
        self.mouse_drag = None
        self.env = None  # the watched env (n = 1)
        self.pop_env = None  # the population while training (n = population)
        self.pop_key = None  # (env key, population) it was made for
        self.trainer = None
        self.env_settings = {}  # the settings chosen per env (name -> {setting: value})
        self.set_env(env, settings)  # (settings: e.g. the URL's ?poles=3)

    @property
    def settings(self):
        """the current env's settings (all of them, see Env.settings)"""
        return ENVS[self.env_name].chosen(self.env_settings.get(self.env_name))

    @property
    def env_key(self):
        """the env with its settings, e.g. "WaterCannon · balls 2": runs and checkpoints
        are kept per key"""
        return env_key(self.env_name, self.settings)

    @property
    def decorated(self):
        """is the watched env decorated (this app draws)?"""
        return self.draws

    # -- modes ------------------------------------------------------------------

    def set_env(self, name, settings=None):
        """switch the env (or its `settings`): a new run (no policy, no history), watch
        mode"""
        self.stop()
        self._close_population()
        if self.env is not None:
            self.mouse_drag.end()
            self.env.close()
        another_env = name != getattr(self, "env_name", None)
        self.env_name = name
        if settings is not None:
            self.env_settings[name] = ENVS[name].chosen(settings)
        # (watch() makes it with the agents it shows)
        self.env = make(name, decorated=self.decorated, **self.settings)
        if another_env and not self._keep_algo:
            self.suggest()
        self._keep_algo = False
        if not ALGOS[self.algo].supports(self.env):
            self.algo = "cem"  # (e.g. DQN, but this env has no discrete actions)
        self.camera.center = b2d.Vec2(*self.env.view_center)
        self.camera.zoom = self.env.view_zoom
        self.camera._matrix = None
        self.new_run()

    def suggestions(self):
        """the env's suggestions for the time budget that can train it: {algo: params}"""
        listed = self.env.suggested.get(self.budget, {})  # (may depend on the settings)
        return {name: params for name, params in listed.items() if name in ALGOS and ALGOS[name].supports(self.env)}

    def suggest(self, algo=None):
        """take the env's suggestion: its best algorithm for the time budget (or `algo`,
        if it is suggested) with the suggested parameters"""
        suggested = self.suggestions()
        if algo is None:
            algo = next(iter(suggested), self.algo)
        if algo in suggested:
            defaults = ALGOS[algo].defaults()
            self.hp[algo] = {**defaults, **{k: v for k, v in suggested[algo].items() if k in defaults}}
        self.algo = algo

    def new_run(self):
        """forget the policy, the trainer's state and the history"""
        self.stop()
        self.run += 1
        self._pending = None
        # (discrete: Q-values of the env's discrete actions, acting with argmax)
        self.policy = ALGOS[self.algo].policy(self.env, self.hp[self.algo])
        self.params = None  # the watched policy's parameters (None: zero actions)
        self.policy_info = "none yet (zero actions)"
        self.policy_iteration = 0  # the training iteration of the watched policy (0: untrained)
        self.scores = []  # returns of the finished watch-mode episodes (newest last)
        self.new_best = False  # the last finished episode beat all earlier ones
        self.train_state = None  # resumes the trainer (stats["state"])
        self.history = []  # stats per training iteration (for the plots)
        self.watch()

    def watch(self):
        """watch mode: a new episode of the watched env (one agent), driven by the current
        policy (the display: the trainer's newest policy takes over now)"""
        if self.mouse_drag is not None:
            self.mouse_drag.end()
        if self._pending is not None:
            self._take_policy(*self._pending)
            self._pending = None
        if self.env.n != 1 or self.env.decorated != self.decorated:
            self.env.close()
            self.env = make(self.env_name, n=1, decorated=self.decorated, **self.settings)
        self.env.freeze_done = False  # (it goes on after its end: with auto reset off, you see it recover)
        # (a new seed each watched episode: another start, as in training)
        self.watched += 1
        self.obs = self.env.reset(seed=self.seed + self.watched)
        self.trace = []
        self.episode_return = 0.0
        self.end_timer = None  # seconds until the next episode (after all ended)
        self.ended = False  # all agents' episodes ended
        self.scored = False  # the current policy's (agent 0) episode ended: its score is in
        self.episodes = 1
        self.mouse_drag = b2d.MouseDrag(self.env.world)

    @property
    def max_population(self):
        """the most agents this env trains with: a non-batchable env needs a box2d world
        per agent (MAX_SEPARATE of them), a batchable one has all in one world"""
        return MAX_POPULATION if self.env.batchable else MAX_SEPARATE

    @property
    def used_population(self):
        """the population training uses: the chosen one (kept for other envs), at most
        max_population"""
        return min(self.population, self.max_population)

    def train(self):
        """train mode: continue the trainer from its last state (or start); while
        training: restart the trainer from its state (e.g. a new population size)"""
        if self.trainer is not None:
            self.trainer.close()
        self.mouse_drag.end()
        algo = ALGOS[self.algo]
        # simulation helpers (more workers, CEM / ES): this env runs a share of the
        # population, the helpers the rest (webgym/helpers.py)
        helpers = self.helpers if self.helpers is not None and self.helpers.count and algo.parallel else None
        population = self.used_population
        local = -(-population // (helpers.count + 1)) if helpers else population
        if self.pop_env is None or self.pop_key != (self.env_key, population, local):
            self._close_population()
            self.pop_env = make(self.env_name, n=local, **self.settings)
            self.pop_key = (self.env_key, population, local)
            if helpers:
                self.pop_env.helpers, self.pop_env.population = helpers, population
        if helpers:
            helpers.spec = {"env": self.env_name, "settings": self.settings}
        if not algo.available():
            raise NeedPackage(algo.requires)
        params = {k: v for k, v in self.hp[self.algo].items() if k not in algo.new_run}
        seed = self.seed + len(self.history)  # a continued run samples new candidates
        self.trainer = algo.train(self.pop_env, self.policy, seed=seed, state=self.train_state, **params)

    def stop(self):
        """back to watch mode; the trainer's state is kept (Train continues)"""
        if self.trainer is not None:
            self.trainer.close()
            self.trainer = None
            self.watch()

    def _close_population(self):
        """free the population's world"""
        if self.pop_env is not None:
            self.pop_env.close()
            self.pop_env = None

    @property
    def training(self):
        return self.trainer is not None

    # -- checkpoints ---------------------------------------------------------------

    def checkpoint(self):
        """the current run as .safetensors bytes: policy weights, the trainer's state
        (resumable), env / algo / population, the history"""
        tensors = {}
        if self.params is not None:
            tensors["policy"] = self.params
        state = {}
        for key, value in (self.train_state or {}).items():
            if isinstance(value, np.ndarray):
                tensors[f"state.{key}"] = value
            else:
                state[key] = value
        metadata = {
            "format": 1,
            "env": self.env_name,
            "settings": self.settings,
            "env_key": self.env_key,
            "algo": self.algo,
            "params": self.hp[self.algo],
            "population": self.population,
            "policy_sizes": self.policy.sizes,
            "state": state if self.train_state is not None else None,
            "history": self.history,
            "policy_info": self.policy_info,
            "run": self.run,
            "iteration": self.history[-1]["iteration"] if self.history else 0,
            "mean_return": self.history[-1]["mean_return"] if self.history else None,
        }
        return checkpoints.dumps(tensors, metadata)

    def load_policy(self, data):
        """the display side of two workers: the trainer's latest policy (a checkpoint).
        The same run, trained on: it takes over with the next episode (an episode is
        played by one policy, of one iteration). Another run (a new run, another
        algorithm or network, a restore) or env: a new episode at once, with the new
        (maybe untrained) policy"""
        tensors, meta = checkpoints.loads(data)
        self._pending = None
        if meta.get("run") == self.shown_run and meta["env"] == self.env_name and \
                ENVS[meta["env"]].chosen(meta.get("settings")) == self.settings:
            self._pending = (tensors, meta)  # (see watch)
            return
        self._env_of(meta)
        self._check_fits(meta)
        self._take_policy(tensors, meta)
        self.shown_run = meta.get("run")
        self.scores, self.new_best = [], False  # (the old run's)
        self.watch()

    def _take_policy(self, tensors, meta):
        """the policy of a checkpoint drives the watched agent"""
        self.algo = meta["algo"]
        sizes, algo = meta["policy_sizes"], ALGOS[self.algo]
        if sizes != self.policy.sizes or self.policy.tanh_out == algo.discrete or self.policy.kind != algo.network:
            self.policy = from_sizes(algo.network, sizes, tanh_out=not algo.discrete)
        self.params = tensors.get("policy")
        self.policy_info = meta["policy_info"]
        self.policy_iteration = meta.get("iteration", 0)

    def _env_of(self, meta):
        """switch to the env (and settings) of a checkpoint, if it is another one"""
        if meta["env"] not in ENVS:  # (an env that was renamed or removed)
            raise ValueError(f"this checkpoint is from the env {meta['env']!r}, which webgym no longer has")
        settings = ENVS[meta["env"]].chosen(meta.get("settings"))  # (older checkpoints: the defaults)
        if meta["env"] != self.env_name or settings != self.settings:
            self.set_env(meta["env"], settings)

    def _check_fits(self, meta):
        """a checkpoint's policy must fit the env as it is now (an env whose code changed
        sees / does other things than in an old checkpoint): else an error, nothing loaded"""
        sizes, env = meta["policy_sizes"], self.env
        discrete = ALGOS[meta["algo"]].discrete if meta["algo"] in ALGOS else False
        outputs = len(env.discrete_actions) if discrete else len(env.action_names)
        if sizes[0] != len(env.obs_names) or sizes[-1] != outputs:
            raise ValueError(
                f"this {meta['env']} checkpoint is from another version of the env (its policy sees "
                f"{sizes[0]} values and has {sizes[-1]} outputs, the env now {len(env.obs_names)} and "
                f"{outputs}): not loaded")

    def restore(self, data):
        """continue a run saved with checkpoint() (watch mode; Train continues it)"""
        tensors, meta = checkpoints.loads(data)
        if meta.get("format") != 1:
            raise ValueError("not a webgym checkpoint")
        if meta["algo"] not in ALGOS:  # (e.g. a run of the removed jax algorithms)
            raise ValueError(f"this checkpoint is from the algorithm {meta['algo']!r}, which webgym no longer has")
        self._env_of(meta)
        self._check_fits(meta)
        self.stop()
        self.run += 1
        self._pending = None
        self.algo = meta["algo"]
        self.hp[self.algo].update(meta.get("params", {}))
        self.population = int(meta["population"])
        sizes, algo = meta["policy_sizes"], ALGOS[self.algo]
        # the network sliders show the restored network (also for checkpoints from before
        # the "layers" parameter)
        if algo.network == "graph":
            self.hp[self.algo].update(max_hidden=sizes[1])
        else:
            self.hp[self.algo].update(layers=len(sizes) - 2, hidden=sizes[1])
        self.policy = from_sizes(algo.network, sizes, tanh_out=not algo.discrete)
        self.params = tensors.get("policy")
        state = meta["state"]
        if state is not None:
            state = dict(state, **{k[len("state.") :]: v for k, v in tensors.items() if k.startswith("state.")})
        self.train_state = state
        self.history = meta["history"]
        self.policy_info = meta["policy_info"]
        self.policy_iteration = meta.get("iteration", 0)
        self.watch()

    # -- host API ----------------------------------------------------------------

    def controls(self):
        """the widgets the hosts show (generic: info / select / slider / checkbox / button),
        each in a section: "bar" (the page's icons above the tabs: Train / Stop, New run,
        reset episode, the switches, how many policies, replay speed),
        "environment"
        (the world: the env, its settings, what is watched) or "training\""""
        resume = self.train_state is not None and not self.training
        env = [
            # ("key": the env with its settings: the page keeps checkpoints per key)
            {"type": "select", "name": "env", "label": "Environment", "options": sorted(ENVS, key=lambda e: (e.startswith("My"), e)), "value": self.env_name,
             "key": self.env_key},
            *({"type": "select", "name": f"setting.{name}", "label": name, "options": list(choices),
               "value": self.settings[name], "help": f"{explanation} A change starts a new run."}
              for name, (_, choices, explanation) in ENVS[self.env_name].settings.items()),
            {"type": "info", "name": "description", "text": self.env.description()},
            {"type": "checkbox", "name": "auto_reset", "label": "Auto reset",
             "value": self.auto_reset, "inline": True,
             "help": "Start a new watched episode when one ends. Off: the episode goes on, see "
                     "the agent recover; R starts a new one. (Training resets its own episodes "
                     "either way.)"},
            {"type": "slider", "name": "replay_speed", "label": "Replay speed", "min": 0,
             "max": len(REPLAY_SPEEDS) - 1, "step": 1, "value": REPLAY_SPEEDS.index(self.replay_speed),
             "labels": [f"{s}x" for s in REPLAY_SPEEDS],
             "help": "How fast the watched policy runs: 1x real time, below slow motion, "
                     "above fast forward. (Training always runs as fast as it can.)"},
            {"type": "checkbox", "name": "follow", "label": "Camera follows", "value": self.follow,
             "disabled": not self.env.follows, "inline": True,
             "help": "The camera follows the agent the environment suggests (e.g. the walker; "
                     "the current policy). Off: a fixed view (pan with the "
                     "mouse, zoom with the wheel). Environments with a fixed scene offer none."},
            {"type": "button", "name": "reset", "label": "Reset episode (R)"},
            {"type": "checkbox", "name": "paused", "label": "Pause", "value": self.paused,
             "help": "Stops the world you watch (look closely, drag the view); training goes "
                     "on. Space toggles it."},
        ]
        suggested = self.suggestions()
        training = [
            {"type": "select", "name": "budget", "label": "Suggest an algorithm for this time budget",
             "options": [["quick", "Quick (~1 min)"], ["medium", "Medium (~5 min)"], ["long", "Long (~20 min)"]],
             "value": self.budget,
             "help": "How long you want to train: picks the algorithm (and its settings) that "
                     "this environment suggests for that time (★ in the list). Rough guesses."},
            {"type": "select", "name": "algo", "label": "Algorithm",
             "options": [[name, ("★ " if name in suggested else "") + a.label]
                         for name, a in ALGOS.items() if a.supports(self.env)],
             "value": self.algo,
             "help": "How the agent learns. ★: suggested for this environment and time budget. "
                     "Discrete-action algorithms (DQN) are offered for envs that also have a "
                     "few discrete actions."},
            *self._param_controls(),
            {"type": "hint", "name": "network", "text": self._network_hint()},
            {"type": "slider", "name": "population", "label": "Population", "min": 8, "max": self.max_population,
             "step": 8, "value": self.used_population,
             "help": "How many agents learn at once (in one world, in the training worker): candidates "
                     "for CEM / ES, parallel episodes for PPO / DQN."},
            {"type": "button", "name": "stop" if self.training else "train",
             "label": "Stop" if self.training else ("Continue training" if resume else "Train")},
            {"type": "button", "name": "new_run", "label": "New run"},
        ]
        # the Toys tab: things to add to the world you watch (webgym/toys.py)
        toy_controls = [
            {"type": "hint", "name": "toys_hint",
             "text": "Pick a tool below, then click into the world (a click on a body still grabs "
                     "it). Click the tool again to put it away. This tab turns auto reset off; R "
                     "resets (and clears the toys)."},
        ]
        for group in toys.GROUPS:  # (each a fold of the tab: its tools as buttons, its sliders)
            tools = [[name, label] for name, (label, _, in_group) in toys.TOOLS.items() if in_group == group]
            if tools:  # (a click into the world uses the chosen one; again: back to grabbing)
                toy_controls.append({"type": "tools", "name": "toy", "value": self.toy, "options": tools,
                                     "group": group, "fold": True})
            if group == "Weather":
                toy_controls.append(
                    {"type": "checkbox", "name": "rain", "label": "Rain", "value": self.rain is not None,
                     "inline": True, "group": group, "fold": True,
                     "help": "Drops fall from above the view onto the agent and the world."})
            toy_controls += [
                {"type": "slider", "name": f"toy.{name}", "label": name, "min": low, "max": high, "step": step,
                 "value": self.toy_params[name], "help": explanation, "group": group, "fold": True,
                 "live": group in toys.LIVE}  # (the watched world follows while it is dragged)
                for name, (_, low, high, step, in_group, explanation) in toys.PARAMS.items() if in_group == group]
            if group == "Gravity":
                toy_controls.append({"type": "button", "name": "gravity_reset", "label": "Reset gravity",
                                     "group": group, "fold": True})
        # always in view (the page's icons above the tabs): the training's buttons, reset
        # episode, auto reset, camera follows; how many policies, replay speed
        bar = lambda c: c["name"] in ("auto_reset", "train", "stop", "new_run", "reset", "paused",  # noqa: E731
                                      "follow", "replay_speed")
        return ([{**c, "section": "bar" if bar(c) else "environment"} for c in env]
                + [{**c, "section": "bar" if bar(c) else "training"} for c in training]
                + [{**c, "section": "toys"} for c in toy_controls])

    def _network_hint(self):
        """the policy network's size, in words"""
        sizes = self.policy.sizes
        if self.policy.kind == "graph":
            return (f"Policy network {sizes[0]} inputs → up to {sizes[1]} hidden neurons (grown by "
                    f"NEAT) → {sizes[-1]} outputs")
        return f"Policy network {' → '.join(map(str, sizes))} · {self.policy.num_params:,} weights"

    def code(self):
        """the python source of the current env and algorithm: {kind: {"path", "text"}}"""
        paths = {
            "env": sys.modules[type(self.env).__module__].__file__,
            "algo": importlib.util.find_spec(f"webgym.algos.{ALGOS[self.algo].module}").origin,
        }
        return {kind: {"path": path, "text": open(path).read()} for kind, path in paths.items()}

    def set_code(self, kind, text):
        """edited source of the env / algorithm: written to its file and reloaded; the
        env is rebuilt, a running training goes on with the new trainer"""
        if kind == "env":
            module = sys.modules[type(self.env).__module__]
        else:
            name = f"webgym.algos.{ALGOS[self.algo].module}"
            module = sys.modules.get(name) or importlib.import_module(name)
        compile(text, module.__file__, "exec")  # (a syntax error: the file stays as it was)
        with open(module.__file__, "w") as f:
            f.write(text)
        importlib.reload(module)  # (an env re-registers itself)
        if kind == "env":
            self.set_env(self.env_name)
        elif self.training:
            self.train()

    def _param_controls(self):
        """sliders for the current algorithm's parameters (collapsible group)"""
        algo = ALGOS[self.algo]
        controls = []
        for key, spec in algo.params.items():
            label = key + (" (new run)" if key in algo.new_run else "")
            common = {"name": f"hp.{key}", "group": "Algorithm parameters", "label": label,
                      "value": self.hp[self.algo][key], "help": HELP.get(key, "")}
            if isinstance(spec[1], list):  # (default, [choices])
                controls.append({"type": "select", "options": spec[1], **common})
            else:  # (default, min, max, step)
                controls.append({"type": "slider", "min": spec[1], "max": spec[2], "step": spec[3], **common})
        return controls

    def command(self, name, value=None):
        """a widget changed / a button was pressed"""
        if name.startswith("hp."):
            key = name[3:]
            default = ALGOS[self.algo].params[key][0]
            self.hp[self.algo][key] = value if isinstance(default, str) else type(default)(value)
            if key in ALGOS[self.algo].new_run:
                self.new_run()  # (e.g. another network size)
            elif self.training:
                self.train()  # goes on from its state with the new value
            return
        if name == "code":  # {"kind": "env" | "algo", "text": source} as JSON
            self.set_code(**json.loads(str(value)))
        elif name == "env":
            self.set_env(value)
        elif name.startswith("setting."):  # an env setting (the page sends text): a new run
            setting = name[len("setting.") :]
            choice = next(c for c in ENVS[self.env_name].settings[setting][1] if str(c) == str(value))
            self.set_env(self.env_name, {**self.settings, setting: choice})
        elif name in ("algo", "budget"):
            if name == "algo" and not ALGOS[value].supports(self.env):
                raise ValueError(f"{value} does not support {self.env_name}")
            training = self.training
            if name == "budget":  # the env's suggestion for that time
                self.budget = value
                self.suggest()
            else:  # (a suggested one: with the suggested parameters)
                self.suggest(value)
            value = self.algo
            self.new_run()  # another algorithm: its own state
            # it was training: the new algorithm trains right away (unless it needs a
            # package that isn't loaded yet: Train then waits for it as usual)
            if training and ALGOS[value].available():
                self.train()
        elif name.startswith("toy."):  # a toy's setting (toys.PARAMS)
            self.toy_params[name[len("toy."):]] = float(value)
        elif name == "gravity_shown":  # (the page: the Toys tab's Gravity fold is open)
            self.gravity_shown = bool(value)
        elif name == "gravity_reset":  # (the Toys tab: the world's own gravity again)
            defaults = toys.defaults()
            for key in ("gravity x", "gravity y"):
                self.toy_params[key] = defaults[key]
        elif name == "toy_mode":  # (the Toys tab was opened; False: left)
            if value is False:
                # (left: a new episode, the world as it was: no toys, rain or other gravity,
                # auto reset on again; their settings stay for the next visit of the tab)
                if self.toys_on:
                    self.toys_on = False
                    self.auto_reset = True
                    self.env.world.gravity = self.env.gravity
                    self.env.world.sleeping_enabled = True
                    self.watch()
            else:
                self._toy_mode()
        elif name in ("toy", "rain"):  # (the Toys tab)
            if name == "toy":
                self.toy = value
            else:
                if self.rain is not None and not value:  # (stopped: its drops go too)
                    self.rain.clear(self.env.world)
                self.rain = toys.Rain() if value else None
            if self.toy != "grab" or self.rain is not None:
                self._toy_mode()
        elif name == "population":
            # the trainer's state does not depend on the population size: training goes
            # on with the new size (from the next iteration on)
            self.population = min(int(value), MAX_POPULATION)
            if self.training:
                self.train()
        elif name == "helpers":  # (the page: this many simulation helpers are ready)
            if self.helpers is not None:
                self.helpers.count = int(value)
                if self.training:
                    self.train()  # (a new split of the population, from the next iteration on)
        elif name == "train":
            self.train()
        elif name == "stop":
            self.stop()
        elif name == "new_run":  # (the button: a fresh network, trained right away)
            self.new_run()
            self.train()
        elif name == "reset":
            self.watch()
        elif name == "paused":
            self.paused = bool(value)
        elif name == "replay_speed":
            self.replay_speed = REPLAY_SPEEDS[int(value)]
        elif name == "follow":
            self.follow = bool(value)
            if not self.follow:  # (back to the env's fixed view)
                self.camera.center = b2d.Vec2(*self.env.view_center)
                self.camera._matrix = None
        elif name == "auto_reset":
            self.auto_reset = bool(value)
            self.env.freeze_done = self.env.n > 1 and self.auto_reset
            if self.auto_reset and self.ended and self.end_timer is None:
                self.end_timer = END_SECONDS
        else:
            raise ValueError(f"unknown command {name!r}")

    def state(self):
        """what the hosts show besides the canvas: mode, episode, last stats"""
        last = self.history[-1] if self.history else {}
        return {
            "mode": "train" if self.training else "watch",
            "env": self.env_name,
            "policy": self.policy_info,
            "policy_iteration": self.policy_iteration,  # (0: untrained)
            "max_return": self.env.max_return,  # (the best possible, None: unknown)
            "episode": self.episodes,
            "episode_over": self.ended,
            "auto_reset": self.auto_reset,
            "episode_return": self.episode_return,
            # the finished episodes' scores (watch mode): the last few and the best
            "toys": self.toys_on,  # (the page shows no scores / return then)
            "scores": [round(s, 1) for s in self.scores[-5:]],
            "best_score": round(max(self.scores), 1) if self.scores else None,
            "new_best": self.new_best,
            "network": None if self.training else self.network(),
            "trace": self._take_trace(),
            "sim_steps": self.sim_steps + (self.helpers.steps if self.helpers is not None else 0),            "iteration": last.get("iteration", 0),
            "frame_ms": round(self.frame_ms, 2),
        }

    def _take_trace(self):
        """the Signals tab: the watched steps since the last call (one row each: the
        observations, the action, the reward), then forgotten"""
        rows, self.trace = self.trace, []
        return {"episode": self.watched, "dt": self.env.dt, "obs": list(self.env.obs_names),
                "actions": list(self.env.action_names), "rows": rows}

    def network(self):
        """the watched policy's network for the view: layer sizes, weights (rounded),
        the activations of every layer for agent 0's current observation, the input /
        output names; None without a policy"""
        if self.params is None:
            return None
        x = np.asarray(self.obs[0], dtype=np.float64)
        if self.policy.kind == "graph":  # (NEAT: nodes and connections of any shape)
            return {**self.policy.view(self.params, x), "inputs": list(self.env.obs_names),
                    "outputs": list(self.env.action_names), "argmax": False}
        activations, weights = [x], []
        layers = self.policy.unflatten(np.asarray(self.params)[None])
        for k, (w, b) in enumerate(layers):
            x = x @ w[0] + b[0]
            if self.policy.tanh_out or k < len(layers) - 1:
                x = np.tanh(x)
            activations.append(x)
            weights.append(np.round(w[0], 3).tolist())
        discrete = not self.policy.tanh_out
        return {
            "sizes": self.policy.sizes,
            "weights": weights,
            "activations": [np.round(a, 3).tolist() for a in activations],
            "inputs": list(self.env.obs_names),
            "outputs": list(self.env.discrete_names if discrete else self.env.action_names),
            "argmax": discrete,  # (Q-values: the chosen action is the largest)
        }

    def frame(self, width, height):
        """one animation frame into the current GL context of size (width, height) px"""
        t0 = time.perf_counter()
        # replay speed: steps per frame (0.25x: one step every 4th frame)
        self._replay_due += 0 if self.paused else self.replay_speed
        while self._replay_due >= 1:
            self._watch_step()
            self._replay_due -= 1
        self._draw(width, height)
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * 1e3 * (time.perf_counter() - t0)

    def event(self, kind, x=0.0, y=0.0, delta=0.0, key=""):
        """input: pointer in drawing-buffer pixels (origin top left), kind "down" /
        "move" / "up" / "leave" / "wheel" (delta: wheel px, + = zoom out) / "pan" (like
        "down" on the background: the page's two-finger pinch); "keydown" with `key` "r":
        a new episode"""
        if kind in ("keydown", "keyup"):
            if kind == "keydown" and key == "r":
                self.command("reset")
            return
        pos = self.camera.convert_screen_to_world(b2d.Vec2(x, y))
        if kind == "pan":  # (two fingers: pan from here, never grab a body)
            self.panning = pos
        elif kind == "down":
            # a dynamic body under the pointer: always grabbed (dragged); else the chosen
            # toy (toys.py) is used there; without one: pan the view
            body = self.mouse_drag.begin(pos)
            if body is not None:
                body.awake = True
            elif self.toy != "grab" and self.toys_on and toys.empty(self.env.world, pos):
                blast = toys.TOOLS[self.toy][1](self.env.world, pos, self.toy_params)
                if blast is not None:  # (an explosion: drawn for a moment)
                    self.blasts.append(blast)
            else:
                self.panning = pos
        elif kind == "move":
            if self.mouse_drag.active:
                self.mouse_drag.update(pos)
            elif self.panning is not None:
                # keep the grabbed world point under the pointer; the user's view wins
                self.camera.center = self.camera.center + (self.panning - pos)
                self.camera._matrix = None
        elif kind in ("up", "leave"):
            self.mouse_drag.end()
            self.panning = None
        elif kind == "wheel":
            # zoom about the pointer: the world point under it stays there
            self.camera.zoom = min(max(self.camera.zoom * 1.0015**delta, 0.2), 500)
            self.camera._matrix = None
            after = self.camera.convert_screen_to_world(b2d.Vec2(x, y))
            self.camera.center = self.camera.center + (pos - after)

    # -- internals ----------------------------------------------------------------

    def _train_steps(self, budget_s=None, max_steps=TRAIN_STEPS):
        """advance the trainer by up to `max_steps` env steps, but at most for `budget_s`
        seconds (the frame rate stays smooth); new stats -> history, the watched policy
        and the state to continue from"""
        deadline = None if budget_s is None else time.perf_counter() + budget_s
        self.waiting = False
        for _ in range(max_steps):
            if deadline is not None and time.perf_counter() > deadline:
                return
            item = next(self.trainer, "finished")
            if item == "finished":
                self.stop()
                return
            if item == "waiting":  # (for the simulation helpers' answers: the host takes them)
                self.waiting = True
                return
            if item is None:  # (an env step: the steps/s count the agents still going)
                self.sim_steps += int(self.pop_env.n - self.pop_env.done.sum())
            else:
                entry = {k: item[k] for k in ("iteration", "mean_return", "best_return")}
                entry.update(item.get("losses", {}))
                self.history.append(entry)
                self.params = item["mean_params"]
                self.train_state = item["state"]
                self.policy_info = f"{self.algo}, iteration {item['iteration']}"
                self.policy_iteration = item["iteration"]

    def _watch_step(self):
        """one env step with the current policy; when the episode ends, the
        world goes on (no actions) for END_SECONDS before the next episode (auto reset)
        or the agent keeps control (no auto reset)"""
        if self.end_timer is not None:
            self.mouse_drag.step(self.env.dt)
            self.env.world.step(self.env.dt, self.env.substeps)
            if self.env.decorated:
                self.env.decorate()
            self.end_timer -= self.env.dt
            if self.end_timer <= 0 and not self.mouse_drag.active:
                episodes = self.episodes + 1
                self.watch()
                self.episodes = episodes
            return
        if self.params is None:
            action = np.zeros((1, *self.env.action_space.shape), dtype=np.float32)
        elif not self.policy.tanh_out:  # Q-values: the best discrete action
            action = self.env.discrete_actions[np.argmax(self.policy(self.params[None], self.obs), axis=1)]
        else:
            action = self.policy(self.params[None], self.obs)
        self.mouse_drag.step(self.env.dt)
        if self.toys_on:  # (the world's gravity: only while the Toys tab is open)
            toys.gravity(self.env.world, self.env.gravity, self.toy_params)
        if self.rain is not None and self.toys_on:  # (drops from just above the view, across its width)
            corner = self.camera.convert_screen_to_world(b2d.Vec2(0, 0))
            other = self.camera.convert_screen_to_world(b2d.Vec2(self.camera.width, 0))
            self.rain.step(self.env.world, corner[0], other[0], corner[1] + 0.5, self.env.dt, self.toy_params)
        seen = self.obs[0]  # (what the policy saw when it chose the action)
        self.obs, reward, terminated, truncated = self.env.step(action)
        if len(self.trace) < 1000:  # (the Signals tab; capped: no state() reads them)
            self.trace.append(np.round(np.concatenate([seen, action[0], reward[:1]]), 3).tolist())
        if self.env.decorated:  # (looks only: the watched env, never a training one)
            self.env.decorate()
        self.episode_return = float(self.env.episode_return[0])
        # the score: the current policy's (none with toys: no time limit, no fair score)
        if self.env.done[0] and not self.scored and not self.toys_on:
            self.scored = True
            # a new best: better than every earlier score (not a tie, not the first)
            self.new_best = bool(self.scores) and self.episode_return > max(self.scores)
            self.scores.append(self.episode_return)
        if self.env.done[0] and not self.ended:
            self.ended = True
            if self.auto_reset:
                self.end_timer = END_SECONDS

    def _toy_mode(self):
        """toys: no auto reset (the episode goes on, see how the agent copes); no scores"""
        self.toys_on = True
        self.auto_reset = False
        self.env.freeze_done = False
        self.end_timer = None  # (an ended episode: no countdown to the next one)

    def _draw(self, width, height):
        """the watched env"""
        if self.debug_draw is None:
            self.debug_draw = GLDebugDraw(self.camera)
        self.camera.set_size(width, height)
        self._update_view()
        gl.glViewport(0, 0, width, height)
        self.debug_draw.begin_draw()
        # (not drawn: the sensors; the rain's drops, drawn as water below)
        self.debug_draw.draw_world(self.env.world, hide=(SENSOR_COLOR, toys.RAIN_COLOR))
        if self.rain is not None:  # (its drops as water)
            self.rain.draw(self.debug_draw, self.toy_params)
        self.debug_draw.flush()  # (the world drawn: the env's own drawing lies on top of it)
        self.env.draw(self.debug_draw)  # (the env's own lines, discs, water, if any)
        self.env.draw_hinges(self.debug_draw)  # (a disc on each joint of the current policy)
        if self.blasts:  # (the Toys tab's explosions)
            self.blasts = toys.draw_blasts(self.debug_draw, self.blasts)
        if self.toys_on and self.gravity_shown:  # (the Gravity fold open: the gravity vector, an arrow)
            toys.draw_gravity(self.debug_draw, self.camera, self.toy_params)
        self.debug_draw.end_draw()

    def _update_view(self):
        """with "Camera follows" the camera glides to the point the env suggests for it
        (env.camera_target)"""
        if self.follow:
            target = self.env.camera_target(0)
            if target is not None:
                c = self.camera.center
                self.camera.center = b2d.Vec2(c[0] + FOLLOW * (target[0] - c[0]), c[1] + FOLLOW * (target[1] - c[1]))
                self.camera._matrix = None
