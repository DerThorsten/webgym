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

import time

import webgym.gl  # noqa: F401 (first: switches PyOpenGL's per-call glGetError off)
from OpenGL import GL as gl

import numpy as np
import pyb2d3 as b2d

from webgym import checkpoints
from webgym.algos import ALGOS
from webgym.core import ENVS, LEADER_COLOR, SCENERY_COLOR, make
from webgym.gl.debug_draw_gl import Camera, GLDebugDraw
from webgym.policies import MLP

import webgym.envs  # noqa: F401 (registers the built-in envs)

GHOST_COLOR = 0x4FA3FF
GHOST_ALPHA = 70
SPEEDS = [1, 2, 4, 8, 16, 64, 256]  # env steps per frame while training (at most)
FRAME_BUDGET_S = 0.012  # training time per frame: the rest of a 60 Hz frame draws
END_SECONDS = 1.0  # watch mode: the world goes on this long after an episode ended
MAX_POPULATION = 256  # (one world per population: no box2d world limit)


class NeedPackage(Exception):
    """the algorithm needs a package that is not loaded (pytorch, jax): the browser
    loads it and trains then"""

    def __init__(self, package):
        super().__init__(f"{package} is needed (load it first)")
        self.package = package


class App:
    """see the module docstring"""

    def __init__(self, env="CartPole", algo="cem", population=64, seed=0, speed=SPEEDS[-1]):
        self.algo = algo
        self.hp = {name: a.defaults() for name, a in ALGOS.items()}  # parameters per algorithm
        self.population = min(population, MAX_POPULATION)
        self.seed = seed
        # env steps per frame (at most: the frame budget keeps the page smooth)
        self.speed = speed if speed in SPEEDS else SPEEDS[-1]
        self.controller = "agent"  # watch mode: "agent" (the policy) or "human" (keyboard)
        self.keys = set()  # keys held down (human controller)
        # watch mode: a new episode when one ends; off: the env goes on after the end
        # (see the agent / yourself recover), "reset" / R starts a new one
        self.auto_reset = True
        self.camera = Camera()
        self.debug_draw = None  # created on the first frame (needs the GL context)
        self.panning = None  # world point under the pointer while panning
        self.frame_ms = 0.0
        self.mouse_drag = None
        self.env = None  # the watched env (n = 1)
        self.pop_env = None  # the population while training (n = population)
        self.trainer = None
        self.set_env(env)

    # -- modes ------------------------------------------------------------------

    def set_env(self, name):
        """switch the env: a new run (no policy, no history), watch mode"""
        self.stop()
        self._close_population()
        if self.env is not None:
            self.mouse_drag.end()
            self.env.close()
        self.env_name = name
        self.env = make(name)
        self.env.freeze_done = False  # watch: the env goes on after the end (no auto reset)
        self.camera.center = b2d.Vec2(*self.env.view_center)
        self.camera.zoom = self.env.view_zoom
        self.camera._matrix = None
        self.new_run()

    def new_run(self):
        """forget the policy, the trainer's state and the history"""
        self.stop()
        hidden = int(self.hp[self.algo].get("hidden", 16))
        self.policy = MLP(self.env.observation_space.shape[0], self.env.action_space.shape[0], hidden=(hidden,))
        self.params = None  # the watched policy's parameters (None: zero actions)
        self.policy_info = "none yet (zero actions)"
        self.scores = []  # returns of the finished watch-mode episodes (newest last)
        self.new_best = False  # the last finished episode beat all earlier ones
        self.train_state = None  # resumes the trainer (stats["state"])
        self.history = []  # stats per training iteration (for the plots)
        self.watch()

    def watch(self):
        """watch mode: the current policy (or a human) drives one env"""
        if self.mouse_drag is not None:
            self.mouse_drag.end()
        self.obs = self.env.reset(seed=self.seed)
        self.episode_return = 0.0
        self.end_timer = None  # seconds until the next episode (after one ended)
        self.ended = False
        self.episodes = 1
        self.mouse_drag = b2d.MouseDrag(self.env.world)

    def train(self):
        """train mode: continue the trainer from its last state (or start); while
        training: restart the trainer from its state (e.g. a new population size)"""
        if self.trainer is not None:
            self.trainer.close()
        self.mouse_drag.end()
        if self.pop_env is None or self.pop_env.n != self.population or not isinstance(self.pop_env, type(self.env)):
            self._close_population()
            self.pop_env = make(self.env_name, n=self.population)
        algo = ALGOS[self.algo]
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
        state = {}
        for key, value in (self.train_state or {}).items():
            if isinstance(value, np.ndarray):
                tensors[f"state.{key}"] = value
            else:
                state[key] = value
        metadata = {
            "format": 1,
            "env": self.env_name,
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

    def restore(self, data):
        """continue a run saved with checkpoint() (watch mode; Train continues it)"""
        tensors, meta = checkpoints.loads(data)
        if meta.get("format") != 1:
            raise ValueError("not a webgym checkpoint")
        if meta["env"] != self.env_name:
            self.set_env(meta["env"])
        self.stop()
        self.algo = meta["algo"]
        self.hp[self.algo].update(meta.get("params", {}))
        self.population = int(meta["population"])
        self.policy = MLP(meta["policy_sizes"][0], meta["policy_sizes"][-1], hidden=tuple(meta["policy_sizes"][1:-1]))
        self.params = tensors.get("policy")
        state = meta["state"]
        if state is not None:
            state = dict(state, **{k[len("state.") :]: v for k, v in tensors.items() if k.startswith("state.")})
        self.train_state = state
        self.history = meta["history"]
        self.policy_info = meta["policy_info"]
        self.watch()

    # -- host API ----------------------------------------------------------------

    def controls(self):
        """the widgets the hosts show (generic: info / select / slider / checkbox / button)"""
        resume = self.train_state is not None and not self.training
        return [
            {"type": "select", "name": "env", "label": "Environment", "options": sorted(ENVS), "value": self.env_name},
            {"type": "info", "name": "description", "text": self.env.description()},
            {"type": "select", "name": "controller", "label": "Controller (when not training)",
             "options": ["agent", "human"], "value": self.controller},
            {"type": "hint", "name": "keys", "active": self.controller == "human",
             "text": f"Keys: {self.env.keys} · R reset" if self.controller == "human"
             else f"Play yourself: Controller = human ({self.env.keys})"},
            {"type": "select", "name": "algo", "label": "Algorithm",
             "options": [[name, a.label] for name, a in ALGOS.items()], "value": self.algo},
            *self._param_controls(),
            {"type": "slider", "name": "population", "label": "Population", "min": 8, "max": MAX_POPULATION,
             "step": 8, "value": self.population},
            {"type": "checkbox", "name": "auto_reset", "label": "Auto reset (when not training)",
             "value": self.auto_reset},
            {"type": "slider", "name": "speed", "label": "Training speed (max, x real time)", "min": 0,
             "max": len(SPEEDS) - 1, "step": 1, "value": SPEEDS.index(self.speed),
             "labels": [f"{s}x" for s in SPEEDS]},
            {"type": "button", "name": "stop" if self.training else "train",
             "label": "Stop" if self.training else ("Continue training" if resume else "Train")},
            {"type": "button", "name": "new_run", "label": "New run"},
            {"type": "button", "name": "reset", "label": "Reset episode (R)"},
        ]

    def _param_controls(self):
        """sliders for the current algorithm's parameters (collapsible group)"""
        algo = ALGOS[self.algo]
        controls = []
        for key, spec in algo.params.items():
            label = key + (" (new run)" if key in algo.new_run else "")
            common = {"name": f"hp.{key}", "group": "Algorithm parameters", "label": label,
                      "value": self.hp[self.algo][key]}
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
        if name == "env":
            self.set_env(value)
        elif name == "controller":
            self.controller = value
            self.scores = []  # (scores of another player)
            self.new_best = False
            self.watch()
        elif name == "algo":
            self.algo = value
            self.new_run()  # another algorithm: its own state
        elif name == "population":
            # the trainer's state does not depend on the population size: training goes
            # on with the new size (from the next iteration on)
            self.population = min(int(value), MAX_POPULATION)
            if self.training:
                self.train()
        elif name == "train":
            self.train()
        elif name == "stop":
            self.stop()
        elif name == "new_run":
            self.new_run()
        elif name == "reset":
            if not self.training:
                self.watch()
        elif name == "speed":
            self.speed = SPEEDS[int(value)]
        elif name == "auto_reset":
            self.auto_reset = bool(value)
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
            "iteration": last.get("iteration", 0),
            "frame_ms": round(self.frame_ms, 2),
        }

    def frame(self, width, height):
        """one animation frame into the current GL context of size (width, height) px"""
        t0 = time.perf_counter()
        if self.training:
            self._train_steps(FRAME_BUDGET_S)
        else:
            self._watch_step()
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

    def _train_steps(self, budget_s=None):
        """advance the trainer by up to `speed` env steps, but at most for `budget_s`
        seconds (the frame rate stays smooth at any speed); new stats -> history, the
        watched policy and the state to continue from"""
        deadline = None if budget_s is None else time.perf_counter() + budget_s
        for _ in range(self.speed):
            if deadline is not None and time.perf_counter() > deadline:
                return
            item = next(self.trainer, "finished")
            if item == "finished":
                self.stop()
                return
            if item is not None:
                entry = {k: item[k] for k in ("iteration", "mean_return", "best_return")}
                entry.update(item.get("losses", {}))
                self.history.append(entry)
                self.params = item["mean_params"]
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
            self.end_timer -= self.env.dt
            if self.end_timer <= 0 and not self.mouse_drag.active:
                episodes = self.episodes + 1
                self.watch()
                self.episodes = episodes
            return
        if self.controller == "human":
            action = self.env.human_action(self.keys)[None]
        elif self.params is None:
            action = np.zeros((1, *self.env.action_space.shape), dtype=np.float32)
        else:
            action = self.policy(self.params, self.obs)
        self.mouse_drag.step(self.env.dt)
        self.obs, reward, terminated, truncated = self.env.step(action)
        self.episode_return = float(self.env.episode_return[0])
        if (terminated[0] or truncated[0]) and not self.ended:
            self.ended = True
            # a new best: better than every earlier score (not a tie, not the first)
            self.new_best = bool(self.scores) and self.episode_return > max(self.scores)
            self.scores.append(self.episode_return)
            if self.auto_reset:
                self.end_timer = END_SECONDS

    def _draw(self, width, height):
        """the env (watch) or the population as ghosts with the leader solid (train)"""
        if self.debug_draw is None:
            self.debug_draw = GLDebugDraw(self.camera)
        self.camera.set_size(width, height)
        self._update_view()
        gl.glViewport(0, 0, width, height)
        self.debug_draw.begin_draw()
        if self.training:
            self.debug_draw.draw_world(
                self.pop_env.world, color=GHOST_COLOR, alpha=GHOST_ALPHA, keep=(LEADER_COLOR, SCENERY_COLOR)
            )
        else:
            self.debug_draw.draw_world(self.env.world)
        self.debug_draw.end_draw()

    def _update_view(self):
        """the leader while training (the camera stays where the user put it)"""
        if self.training:
            env = self.pop_env
            # the leader stays until its episode ends (no flicker), then the best alive
            if self.leader is None or env.done[self.leader]:
                alive = np.flatnonzero(~env.done)
                self.leader = int(alive[np.argmax(env.episode_return[alive])]) if len(alive) else None
            env.set_leader(self.leader)
