"""The shared app: envs, training, drawing, interaction; identical natively and in
the browser.

Hosts (host_native.py, host_web.py) only provide a current GL context, call
`frame(width, height)` once per animation frame, forward pointer / key input to
`event(...)`, render `controls()` as widgets (values back via `command(name, value)`)
and plot `history`.

Two modes:
* watch: one env (n = 1) driven by the current policy (zero actions before any
  training) or by a human (keyboard); drag its bodies with the pointer
* train: the trainer's population (one batched env, n agents in one world) drawn as
  translucent ghosts, the leader (best alive candidate) solid; the pointer only pans /
  zooms. Stop keeps the trainer's state: Train continues, "New run" starts over.
"""

import importlib.util
import json
import sys
import time

import webgym.gl  # noqa: F401 (first: switches PyOpenGL's per-call glGetError off)
from OpenGL import GL as gl

import numpy as np
import pyb2d3 as b2d

from webgym import checkpoints
from webgym.algos import ALGOS, HELP
from webgym.core import ENVS, LEADER_COLOR, PARTICLE_COLOR, PROP_COLOR, SCENERY_COLOR, env_key, make
from webgym.gl.debug_draw_gl import Camera, GLDebugDraw
from webgym.policies import MLP, policy_for

import webgym.envs  # noqa: F401 (registers the built-in envs)

GHOST_COLOR = 0x4FA3FF
GHOST_ALPHA = 70
TRAIN_STEPS = 256  # env steps per _train_steps call at most (training: always full speed)
REPLAY_SPEEDS = [0.25, 0.5, 1, 2, 4, 8]  # watch mode: x real time (slow motion .. fast)
FRAME_BUDGET_S = 0.012  # training time per frame: the rest of a 60 Hz frame draws
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

    def __init__(self, env="CartPole", algo="cem", population=64, seed=0, draws=True, helpers=None):
        self.algo = algo
        # simulation helpers (webgym.helpers.Pool, the training worker; None: none): run a
        # share of a CEM / ES population in more workers
        self.helpers = helpers
        self.waiting = False  # the trainer waits for the helpers' answers
        # decoration of the watched env (flags, thrust particles: looks only, never in
        # training): wanted by the user (the checkbox) and only where something is drawn
        # (not in the training worker, whose controls the page shows: hence two flags)
        self.draws = draws
        self.show_deco = True
        self.hp = {name: a.defaults() for name, a in ALGOS.items()}  # parameters per algorithm
        self.population = min(population, MAX_POPULATION)
        self.seed = seed
        # watch mode: env steps per frame (x real time); `_replay_due`: steps owed
        self.replay_speed = 1
        self._replay_due = 0.0
        self.controller = "agent"  # watch mode: "agent" (the policy) or "human" (keyboard)
        self.keys = set()  # keys held down (human controller)
        # watch mode: a new episode when one ends; off: the env goes on after the end
        # (see the agent / yourself recover), "reset" / R starts a new one
        self.auto_reset = True
        self._auto_reset_before = True  # (playing yourself turns it off; restored after)
        # watch mode: how many policies run at once: the current one (solid) + the best
        # candidates (CEM / ES) or the most recent policies (PPO / DQN) as ghosts
        self.show_n = 1  # (more: the "Show policies" slider)
        # the camera follows the point the env suggests (e.g. the walker); off: fixed view
        self.follow = True
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
        self.set_env(env)

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
        """is the watched env decorated (the user wants it, and this app draws)?"""
        return self.show_deco and self.draws

    # -- modes ------------------------------------------------------------------

    def set_env(self, name, settings=None):
        """switch the env (or its `settings`): a new run (no policy, no history), watch
        mode"""
        self.stop()
        self._close_population()
        if self.env is not None:
            self.mouse_drag.end()
            self.env.close()
        self.env_name = name
        if settings is not None:
            self.env_settings[name] = ENVS[name].chosen(settings)
        # (watch() makes it with the agents it shows)
        self.env = make(name, decorated=self.decorated, **self.settings)
        if not ALGOS[self.algo].supports(self.env):
            self.algo = "cem"  # (e.g. DQN, but this env has no discrete actions)
        self.camera.center = b2d.Vec2(*self.env.view_center)
        self.camera.zoom = self.env.view_zoom
        self.camera._matrix = None
        self.new_run()

    def new_run(self):
        """forget the policy, the trainer's state and the history"""
        self.stop()
        # (discrete: Q-values of the env's discrete actions, acting with argmax)
        hp = self.hp[self.algo]
        self.policy = policy_for(self.env, ALGOS[self.algo].discrete, hp.get("hidden", 16), hp.get("layers", 1))
        self.params = None  # the watched policy's parameters (None: zero actions)
        self.top_params = None  # (k, num_params): more policies to show, best / newest first
        self.recent = []  # the last policies (for algorithms without top_params)
        self.policy_info = "none yet (zero actions)"
        self.scores = []  # returns of the finished watch-mode episodes (newest last)
        self.new_best = False  # the last finished episode beat all earlier ones
        self.train_state = None  # resumes the trainer (stats["state"])
        self.history = []  # stats per training iteration (for the plots)
        self.watch()

    def watch(self):
        """watch mode: the current policy and show_n - 1 more (or a human: alone) drive
        the agents of the watched env"""
        if self.mouse_drag is not None:
            self.mouse_drag.end()
        n = 1 if self.controller == "human" or not ENVS[self.env_name].batchable else self.show_n
        if self.env.n != n or self.env.decorated != self.decorated:
            self.env.close()
            self.env = make(self.env_name, n=n, decorated=self.decorated, **self.settings)
        # one agent: it goes on after its end (no auto reset); several: finished ghosts
        # disappear (with auto reset)
        self.env.freeze_done = n > 1 and self.auto_reset
        self.env.same_start = True  # (all you watch start alike: a fair comparison)
        self.obs = self.env.reset(seed=self.seed)
        self.episode_return = 0.0
        self.end_timer = None  # seconds until the next episode (after all ended)
        self.ended = False  # all agents' episodes ended
        self.scored = False  # the current policy's (agent 0) episode ended: its score is in
        self.episodes = 1
        self.mouse_drag = b2d.MouseDrag(self.env.world)

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
        local = -(-self.population // (helpers.count + 1)) if helpers else self.population
        if self.pop_env is None or self.pop_key != (self.env_key, self.population, local):
            self._close_population()
            self.pop_env = make(self.env_name, n=local, **self.settings)  # (non-batchable: fewer)
            self.pop_key = (self.env_key, self.population, local)
            if helpers:
                self.pop_env.helpers, self.pop_env.population = helpers, self.population
        if helpers:
            helpers.spec = {"env": self.env_name, "settings": self.settings}
        if not algo.available():
            raise NeedPackage(algo.requires)
        params = {k: v for k, v in self.hp[self.algo].items() if k not in algo.new_run}
        seed = self.seed + len(self.history)  # a continued run samples new candidates
        self.trainer = algo.train(self.pop_env, self.policy, seed=seed, state=self.train_state, **params)
        self.leader = None

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
        if self.top_params is not None and len(self.top_params):
            tensors["top"] = self.top_params  # (more policies for the display)
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
            "iteration": self.history[-1]["iteration"] if self.history else 0,
            "mean_return": self.history[-1]["mean_return"] if self.history else None,
        }
        return checkpoints.dumps(tensors, metadata)

    def load_policy(self, data):
        """the display side of two workers: the trainer's latest policy (a checkpoint)
        drives the watched env from now on; the episode goes on (no reset), unless the
        env changed"""
        tensors, meta = checkpoints.loads(data)
        self._env_of(meta)
        self._check_fits(meta)
        self.algo = meta["algo"]
        sizes = meta["policy_sizes"]
        if sizes != self.policy.sizes or self.policy.tanh_out == ALGOS[self.algo].discrete:
            self.policy = MLP(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]), tanh_out=not ALGOS[self.algo].discrete)
        self.params = tensors.get("policy")
        self.top_params = tensors.get("top")
        self.policy_info = meta["policy_info"]

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
        self.algo = meta["algo"]
        self.hp[self.algo].update(meta.get("params", {}))
        self.population = int(meta["population"])
        sizes = meta["policy_sizes"]
        # the network sliders show the restored network (also for checkpoints from before
        # the "layers" parameter)
        self.hp[self.algo].update(layers=len(sizes) - 2, hidden=sizes[1])
        self.policy = MLP(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]), tanh_out=not ALGOS[self.algo].discrete)
        self.params = tensors.get("policy")
        self.top_params = tensors.get("top")
        state = meta["state"]
        if state is not None:
            state = dict(state, **{k[len("state.") :]: v for k, v in tensors.items() if k.startswith("state.")})
        self.train_state = state
        self.history = meta["history"]
        self.policy_info = meta["policy_info"]
        self.watch()

    # -- host API ----------------------------------------------------------------

    def controls(self):
        """the widgets the hosts show (generic: info / select / slider / checkbox / button),
        each in a section: "bar" (the page's top bar: play yourself, auto reset, Train /
        New run), "environment"
        (the world: the env, its settings, what is watched) or "training\""""
        resume = self.train_state is not None and not self.training
        env = [
            # ("key": the env with its settings: the page keeps checkpoints per key)
            {"type": "select", "name": "env", "label": "Environment", "options": sorted(ENVS), "value": self.env_name,
             "key": self.env_key},
            *({"type": "select", "name": f"setting.{name}", "label": name, "options": list(choices),
               "value": self.settings[name], "help": f"{explanation} A change starts a new run."}
              for name, (_, choices, explanation) in ENVS[self.env_name].settings.items()),
            {"type": "info", "name": "description", "text": self.env.description()},
            {"type": "heading", "name": "watching", "text": "Watching"},
            {"type": "checkbox", "name": "controller", "label": "Play yourself (keyboard)",
             "value": self.controller == "human", "inline": True,
             "help": "You control the env with the keyboard instead of the trained policy; "
                     "training (if it runs) goes on in the background. Turns auto reset off "
                     "while you play (back on when you stop). (On phones, with one worker, "
                     "the view shows the training while it runs: play when it is stopped.)"},
            {"type": "checkbox", "name": "auto_reset", "label": "Auto reset",
             "value": self.auto_reset, "inline": True,
             "help": "Start a new watched episode when one ends. Off: the episode goes on, see "
                     "the agent (or yourself) recover; R starts a new one. (Training resets its "
                     "own episodes either way.)"},
            {"type": "hint", "name": "keys", "active": self.controller == "human",
             "text": f"Keys: {self.env.keys} · R reset"},
            {"type": "slider", "name": "show_n", "label": "Show policies", "min": 1, "max": 16, "step": 1,
             "value": self.show_n, "disabled": not self.env.batchable,
             "help": "How many policies run at once in the view: the current one (solid) and as "
                     "ghosts the best candidates of the last iteration (CEM / ES) or the most "
                     "recent policies (PPO / DQN). With the human controller only you."},
            {"type": "slider", "name": "replay_speed", "label": "Replay speed", "min": 0,
             "max": len(REPLAY_SPEEDS) - 1, "step": 1, "value": REPLAY_SPEEDS.index(self.replay_speed),
             "labels": [f"{s}x" for s in REPLAY_SPEEDS],
             "help": "How fast the watched policies run: 1x real time, below slow motion, "
                     "above fast forward. (Training always runs as fast as it can.)"},
            {"type": "checkbox", "name": "follow", "label": "Camera follows", "value": self.follow,
             "disabled": not self.env.follows, "inline": True,
             "help": "The camera follows the agent the environment suggests (e.g. the walker; "
                     "while training the highlighted leader). Off: a fixed view (pan with the "
                     "mouse, zoom with the wheel). Environments with a fixed scene offer none."},
            {"type": "checkbox", "name": "decorate", "label": "Decoration", "value": self.show_deco,
             "inline": True,
             "help": "Looks only, in the env you watch (never in training): e.g. the lander's "
                     "flags and thrust particles. Off: just the physics."},
            {"type": "button", "name": "reset", "label": "Reset episode (R)"},
        ]
        training = [
            {"type": "select", "name": "algo", "label": "Algorithm",
             "options": [[name, a.label] for name, a in ALGOS.items() if a.supports(self.env)],
             "value": self.algo,
             "help": "How the agent learns. Discrete-action algorithms (DQN) are offered for "
                     "envs that also have a few discrete actions."},
            *self._param_controls(),
            {"type": "hint", "name": "network",
             "text": f"Policy network {' → '.join(map(str, self.policy.sizes))} · "
                     f"{self.policy.num_params:,} weights"},
            {"type": "slider", "name": "population", "label": "Population", "min": 8, "max": MAX_POPULATION,
             "step": 8, "value": self.population, "disabled": not self.env.batchable,
             "help": "How many agents learn at once (one world, drawn as ghosts): candidates "
                     "for CEM / ES, parallel episodes for PPO / DQN."},
            {"type": "button", "name": "stop" if self.training else "train",
             "label": "Stop" if self.training else ("Continue training" if resume else "Train")},
            {"type": "button", "name": "new_run", "label": "New run"},
        ]
        # the top bar (always in view): who plays (you / the agent), auto reset, the keys
        # (while you play), the training's buttons
        bar = lambda c: c["name"] in ("controller", "auto_reset", "keys", "train", "stop", "new_run")  # noqa: E731
        return ([{**c, "section": "bar" if bar(c) else "environment"} for c in env]
                + [{**c, "section": "bar" if bar(c) else "training"} for c in training])

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
        elif name == "controller":  # (the checkbox: True = you play; or "human" / "agent")
            human = value in (True, "human")
            if human and self.controller != "human":  # playing: no auto reset (restored after)
                self._auto_reset_before, self.auto_reset = self.auto_reset, False
            elif not human and self.controller == "human":
                self.auto_reset = self._auto_reset_before
            self.controller = "human" if human else "agent"
            self.scores = []  # (scores of another player)
            self.new_best = False
            self.watch()
        elif name == "algo":
            if not ALGOS[value].supports(self.env):
                raise ValueError(f"{value} does not support {self.env_name}")
            training = self.training
            self.algo = value
            self.new_run()  # another algorithm: its own state
            # it was training: the new algorithm trains right away (unless it needs a
            # package that isn't loaded yet: Train then waits for it as usual)
            if training and ALGOS[value].available():
                self.train()
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
            if not self.training:
                self.watch()
        elif name == "replay_speed":
            self.replay_speed = REPLAY_SPEEDS[int(value)]
        elif name == "show_n":
            self.show_n = int(value)
            self.watch()
        elif name == "decorate":
            self.show_deco = bool(value)
            if self.env.decorated != self.decorated and not self.training:
                self.watch()  # (builds the watched env anew, with / without decoration)
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
            "policy": "you (keyboard)" if self.controller == "human" else self.policy_info,
            "episode": self.episodes,
            "episode_over": self.ended,
            "auto_reset": self.auto_reset,
            "episode_return": self.episode_return,
            # the finished episodes' scores (watch mode): the last few and the best
            "scores": [round(s, 1) for s in self.scores[-5:]],
            "best_score": round(max(self.scores), 1) if self.scores else None,
            "new_best": self.new_best,
            "network": None if self.training else self.network(),
            "iteration": last.get("iteration", 0),
            "frame_ms": round(self.frame_ms, 2),
        }

    def network(self):
        """the watched policy's network for the view: layer sizes, weights (rounded),
        the activations of every layer for agent 0's current observation, the input /
        output names; None without a policy"""
        if self.params is None or self.controller == "human":
            return None
        x = np.asarray(self.obs[0], dtype=np.float64)
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
        if self.training:
            self._train_steps(FRAME_BUDGET_S)
        else:
            # replay speed: steps per frame (0.25x: one step every 4th frame)
            self._replay_due += self.replay_speed
            while self._replay_due >= 1:
                self._watch_step()
                self._replay_due -= 1
        self._draw(width, height)
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * 1e3 * (time.perf_counter() - t0)

    def event(self, kind, x=0.0, y=0.0, delta=0.0, key=""):
        """input: pointer in drawing-buffer pixels (origin top left), kind "down" /
        "move" / "up" / "leave" / "wheel" (delta: wheel px, + = zoom out); keys:
        "keydown" / "keyup" with `key` ("left", "right", "up", "down", "space", letters)"""
        if kind == "keydown":
            self.keys.add(key)
            if key == "r":
                self.command("reset")
            return
        if kind == "keyup":
            self.keys.discard(key)
            return
        pos = self.camera.convert_screen_to_world(b2d.Vec2(x, y))
        if kind == "down":
            body = None if self.training else self.mouse_drag.begin(pos)
            if body is not None:
                body.awake = True
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
            if item is not None:
                entry = {k: item[k] for k in ("iteration", "mean_return", "best_return")}
                entry.update(item.get("losses", {}))
                self.history.append(entry)
                self.params = item["mean_params"]
                top = item.get("top_params")
                if top is None:  # (PPO / DQN: the most recent policies, newest first)
                    self.recent = [*self.recent[-15:], self.params]
                    top = np.array(self.recent[-2::-1]) if len(self.recent) > 1 else None
                self.top_params = top
                self.train_state = item["state"]
                self.policy_info = f"{self.algo}, iteration {item['iteration']}"
                self.leader = None

    def _watch_step(self):
        """one env step with the current policy / the human; when the episode ends, the
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
        n = self.env.n
        if self.controller == "human":
            action = self.env.human_action(self.keys)[None]
        elif self.params is None:
            action = np.zeros((n, *self.env.action_space.shape), dtype=np.float32)
        elif not self.policy.tanh_out:  # Q-values: the best discrete action
            action = self.env.discrete_actions[np.argmax(self.policy(self._watch_params(), self.obs), axis=1)]
        else:
            action = self.policy(self._watch_params(), self.obs)
        self.mouse_drag.step(self.env.dt)
        self.obs, reward, terminated, truncated = self.env.step(action)
        if self.env.decorated:  # (looks only: the watched env, never a training one)
            self.env.decorate()
        self.episode_return = float(self.env.episode_return[0])
        if self.env.done[0] and not self.scored:  # the score: the current policy's
            self.scored = True
            # a new best: better than every earlier score (not a tie, not the first)
            self.new_best = bool(self.scores) and self.episode_return > max(self.scores)
            self.scores.append(self.episode_return)
        if self.env.done.all() and not self.ended:
            self.ended = True
            if self.auto_reset:
                self.end_timer = END_SECONDS

    def _watch_params(self):
        """(n, num_params): agent 0 the current policy, then the best candidates (CEM /
        ES) or the most recent policies (PPO / DQN), repeated if there are fewer"""
        rows = [self.params, *(self.top_params if self.top_params is not None else [])]
        return np.stack([rows[i % len(rows)] for i in range(self.env.n)])

    def _draw(self, width, height):
        """the env (watch) or the population as ghosts with the leader solid (train)"""
        if self.debug_draw is None:
            self.debug_draw = GLDebugDraw(self.camera)
        self.camera.set_size(width, height)
        self._update_view()
        gl.glViewport(0, 0, width, height)
        self.debug_draw.begin_draw()
        env = self.pop_env if self.training else self.env
        if env.n > 1:  # ghosts, the leader solid
            self.debug_draw.draw_world(
                env.world, color=GHOST_COLOR, alpha=GHOST_ALPHA, keep=(LEADER_COLOR, SCENERY_COLOR, PROP_COLOR, PARTICLE_COLOR)
            )
        else:
            self.debug_draw.draw_world(env.world)
        self.debug_draw.end_draw()

    def _update_view(self):
        """the leader: while training the best alive candidate, watching several
        policies the current one (agent 0); with "Camera follows" the camera glides to
        the point the env suggests for it (env.camera_target)"""
        if not self.training and self.env.n > 1:
            self.env.set_leader(0)
        if self.training:
            env = self.pop_env
            # the leader stays until its episode ends (no flicker), then the best alive
            if self.leader is None or env.done[self.leader]:
                alive = np.flatnonzero(~env.done)
                self.leader = int(alive[np.argmax(env.episode_return[alive])]) if len(alive) else None
            env.set_leader(self.leader)
        if self.follow:
            env, agent = (self.pop_env, self.leader) if self.training else (self.env, 0)
            target = env.camera_target(agent) if agent is not None else None
            if target is not None:
                c = self.camera.center
                self.camera.center = b2d.Vec2(c[0] + FOLLOW * (target[0] - c[0]), c[1] + FOLLOW * (target[1] - c[1]))
                self.camera._matrix = None
