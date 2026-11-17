"""The shared app: envs, training, drawing, interaction; identical natively and in
the browser.

Hosts (host_native.py, host_web.py) only provide a current GL context, call
`frame(width, height)` once per animation frame, forward pointer input to
`event(...)`, render `controls()` as widgets (values back via `command(name, value)`)
and plot `history`.

Two modes:
* watch: one env driven by the current policy (zero actions before any training);
  drag its bodies with the pointer (the env is yours to mess with)
* train: the trainer's population of envs drawn overlaid as translucent ghosts,
  advanced `speed` population steps per frame; the pointer only pans / zooms
"""

import time

import webgym.gl  # noqa: F401 (first: switches PyOpenGL's per-call glGetError off)
from OpenGL import GL as gl

import numpy as np
import pyb2d3 as b2d

from webgym.algos import ALGOS
from webgym.core import ENVS, make
from webgym.gl.debug_draw_gl import Camera, GLDebugDraw
from webgym.policies import MLP

import webgym.envs  # noqa: F401 (registers the built-in envs)

GHOST_COLOR = 0x4FA3FF
GHOST_ALPHA = 70
SPEEDS = [1, 2, 4, 8, 16, 64, 256]  # population steps per frame while training
END_SECONDS = 1.0  # watch mode: the world goes on this long after an episode ended


class App:
    """see the module docstring"""

    def __init__(self, env="CartPole", algo="cem", population=32, seed=0):
        self.algo = algo
        self.population = population
        self.seed = seed
        self.speed = 4
        self.controller = "agent"  # watch mode: "agent" (the policy) or "human" (keyboard)
        self.keys = set()  # keys held down (human controller)
        # watch mode: a new episode when one ends; off: the env goes on after the end
        # (see the agent / yourself recover), "reset" starts a new one
        self.auto_reset = True
        self.camera = Camera()
        self.debug_draw = None  # created on the first frame (needs the GL context)
        self.panning = None  # world point under the pointer while panning
        self.frame_ms = 0.0
        self.set_env(env)

    # -- modes ------------------------------------------------------------------

    def set_env(self, name):
        """switch the env: forget the policy and the history, watch mode"""
        self.env_name = name
        self.env = make(name)
        self.policy = MLP(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.params = None  # the watched policy's parameters (None: zero actions)
        self.policy_info = "none yet (zero actions)"
        self.history = []  # stats per training iteration (for the plots)
        self.camera.center = b2d.Vec2(*self.env.view_center)
        self.camera.zoom = self.env.view_zoom
        self.camera._matrix = None
        self.watch()

    def watch(self):
        """watch mode: the current policy drives one env, the pointer drags bodies"""
        self.trainer = None
        self.envs = None
        self.obs, _ = self.env.reset(seed=self.seed)
        self.episode_return = 0.0
        self.end_timer = None  # seconds until the next episode (after one ended)
        self.ended = False  # the episode is over (without auto reset: it goes on anyway)
        self.episodes = 1
        self.mouse_drag = b2d.MouseDrag(self.env.world)

    def train(self):
        """train mode: a new trainer on a population of envs (continues from nothing:
        a fresh run, the history is cleared)"""
        self.mouse_drag.end()
        self.envs = [make(self.env_name) for _ in range(self.population)]
        self.trainer = ALGOS[self.algo](self.envs, self.policy, seed=self.seed)
        self.history = []

    @property
    def training(self):
        return self.trainer is not None

    # -- host API ----------------------------------------------------------------

    def controls(self):
        """the widgets the hosts show (generic: select / button / slider)"""
        return [
            {"type": "select", "name": "env", "label": "Environment", "options": sorted(ENVS), "value": self.env_name},
            {"type": "info", "name": "description", "text": self.env.description()},
            {"type": "select", "name": "controller", "label": "Controller (when not training)",
             "options": ["agent", "human"], "value": self.controller},
            {"type": "select", "name": "algo", "label": "Algorithm", "options": sorted(ALGOS), "value": self.algo},
            {"type": "slider", "name": "population", "label": "Population", "min": 4, "max": 128, "step": 4,
             "value": self.population},
            {"type": "button", "name": "stop" if self.training else "train", "label": "Stop" if self.training else "Train"},
            {"type": "button", "name": "reset", "label": "Reset episode"},
            {"type": "checkbox", "name": "auto_reset", "label": "Auto reset (when not training)",
             "value": self.auto_reset},
            {"type": "slider", "name": "speed", "label": "Training speed (x real time)", "min": 0,
             "max": len(SPEEDS) - 1, "step": 1, "value": SPEEDS.index(self.speed),
             "labels": [f"{s}x" for s in SPEEDS]},
        ]

    def command(self, name, value=None):
        """a widget changed / a button was pressed"""
        if name == "env":
            self.set_env(value)
        elif name == "controller":
            self.controller = value
            self.watch()
        elif name == "algo":
            self.algo = value
        elif name == "population":
            self.population = int(value)
        elif name == "train":
            self.train()
        elif name == "stop":
            self.watch()
        elif name == "reset":
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
            "iteration": last.get("iteration", 0),
            "frame_ms": round(self.frame_ms, 2),
        }

    def frame(self, width, height):
        """one animation frame into the current GL context of size (width, height) px"""
        t0 = time.perf_counter()
        if self.training:
            self._train_steps()
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
                # keep the grabbed world point under the pointer
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

    def _train_steps(self):
        """advance the trainer by `speed` population steps; new stats -> history and
        the watched policy"""
        for _ in range(self.speed):
            item = next(self.trainer, "finished")
            if item == "finished":
                self.watch()
                return
            if item is not None:
                self.history.append({k: v for k, v in item.items() if not k.endswith("_params")})
                self.params = item["mean_params"]
                self.policy_info = f"{self.algo}, iteration {item['iteration']}"

    def _watch_step(self):
        """one env step with the current policy; when the episode ends, the world goes
        on (no actions) for END_SECONDS before the next episode: you see what happened"""
        if self.end_timer is not None:
            self.mouse_drag.step(self.env.dt)
            self.env.world.step(self.env.dt, self.env.substeps)
            self.end_timer -= self.env.dt
            if self.end_timer <= 0 and not self.mouse_drag.active:
                self.end_timer = None
                self.ended = False
                self.episodes += 1
                self.obs, _ = self.env.reset()
                self.episode_return = 0.0
            return
        if self.controller == "human":
            action = self.env.human_action(self.keys)
        elif self.params is None:
            action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        else:
            action = self.policy(self.params, self.obs[None])[0]
        self.mouse_drag.step(self.env.dt)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        if not self.ended:
            self.episode_return += reward
        if (terminated or truncated) and not self.ended:
            self.ended = True
            if self.auto_reset:
                self.end_timer = END_SECONDS

    def _draw(self, width, height):
        """the env (watch) or the live population as ghosts (train)"""
        if self.debug_draw is None:
            self.debug_draw = GLDebugDraw(self.camera)
        self.camera.set_size(width, height)
        gl.glViewport(0, 0, width, height)
        if self.training:
            alive = [env for env in self.envs if not env.done]
            # the leader: the alive candidate with the highest return so far
            leader = max(alive, key=lambda env: env.episode_return) if alive else None
        else:
            alive, leader = [], self.env
        self._follow(leader)
        self.debug_draw.begin_draw()
        for env in alive:
            if env is not leader:
                self.debug_draw.draw_world(env.world, color=GHOST_COLOR, alpha=GHOST_ALPHA)
        if leader is not None:
            self.debug_draw.draw_world(leader.world)  # on top, own colors
        self.debug_draw.end_draw()

    def _follow(self, env):
        """the camera follows env.focus() horizontally (smoothly), unless panning"""
        point = env.focus() if env is not None else None
        if point is None or self.panning is not None:
            return
        x = self.camera.center[0] + 0.1 * (point[0] - self.camera.center[0])
        self.camera.center = b2d.Vec2(x, self.camera.center[1])
        self.camera._matrix = None
