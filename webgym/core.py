"""Minimal gymnasium-style env API (no gymnasium dependency) + env registry.

An env holds `n` agents in ONE box2d world: they share the scenery (ground) but don't
interact (collision filters). The API is batched (numpy, one row per agent), which is
much faster than one world per agent (one world.step, batch reads / writes; in wasm
~8x, see webgym/bench.py). n = 1 is a classic single env.
"""

import sys

import numpy as np
import pyb2d3 as b2d

# collision categories: agents collide with the scenery and with category 1 (the
# default category of queries: the pointer can grab them), not with each other
SCENERY = 4
AGENT = 2
AGENT_FILTER = b2d.make_filter(category_bits=AGENT, mask_bits=1 | SCENERY)
SCENERY_FILTER = b2d.make_filter(category_bits=SCENERY)
SCENERY_COLOR = 0x4E8A68  # custom debug draw colors (the view keeps these opaque)
LEADER_COLOR = 0xFF8C42


class Box:
    """continuous space: arrays of `shape` in [low, high]"""

    def __init__(self, low, high, shape):
        self.shape = tuple(shape)
        self.low = np.broadcast_to(np.float32(low), self.shape)
        self.high = np.broadcast_to(np.float32(high), self.shape)

    def sample(self, rng, n=None):
        size = self.shape if n is None else (n, *self.shape)
        return rng.uniform(self.low, self.high, size).astype(np.float32)


def new_world(**kwargs):
    """b2d.World(**kwargs), checked: box2d hands out an invalid world (no error) when
    all b2d.MAX_WORLDS (128) are in use, and using it corrupts memory"""
    world = b2d.World(**kwargs)
    if not world.is_valid:
        raise RuntimeError(f"too many box2d worlds (at most {b2d.MAX_WORLDS}): close() unused envs")
    return world


class Env:
    """n agents in one box2d world, stepped with a fixed dt per env step.

    Subclasses implement `build()` (scenery via `scenery_shape`, agents via `agent_body`),
    `_reset(rng)`, `_apply(actions)`, `_observe()` -> (n, obs_dim) and `_outcome()` ->
    (reward (n,), terminated (n,)), all batched. Spaces are per agent. The module
    docstring of an env is its description in the UI.
    """

    observation_space = None
    action_space = None
    dt = 1 / 60
    substeps = 4
    max_steps = 1000
    gravity = (0, -10)
    # initial camera: world point at the view center, half height in world units
    view_center = (0.0, 0.0)
    view_zoom = 10.0
    keys = ""  # the keys of the human controller (shown in the UI, see human_action)

    def __init__(self, n=1):
        self.n = n
        self.world = new_world(gravity=self.gravity)
        self._bodies = []  # all agent bodies
        self._owner = []  # agent index of each
        self._shapes = [[] for _ in range(n)]  # shapes per agent (leader color)
        self.build()
        self._bodies = b2d.Bodies(self._bodies)
        self._owner = np.array(self._owner)
        self._shapes = [b2d.Shapes(s) for s in self._shapes]
        self.start = self.world.get_snapshot()
        self.steps = 0
        self.done = np.zeros(n, dtype=bool)  # the agent's episode ended
        self.episode_return = np.zeros(n)
        # finished agents are disabled (not simulated, not drawn) until the next reset;
        # False: they go on (watch mode without auto reset)
        self.freeze_done = True
        self._leader = None

    # -- building ----------------------------------------------------------------

    def build(self):
        """create the scenery and the n agents"""
        raise NotImplementedError

    def scenery_shape(self, body, geometry, **material):
        """a shape of the static scenery (ground, flags, ...)"""
        mat = b2d.surface_material(custom_color=SCENERY_COLOR, **material)
        return body.create_shape(b2d.shape_def(filter=SCENERY_FILTER, material=mat), geometry)

    def agent_body(self, i, position, shapes, **body_kwargs):
        """a dynamic body of agent i with `shapes`: [(geometry, density), ...]"""
        body = self.world.create_dynamic_body(position=position, **body_kwargs)
        material = b2d.surface_material(friction=0.8)
        for geometry, density in shapes:
            shape_def = b2d.shape_def(density=density, filter=AGENT_FILTER, material=material)
            self._shapes[i].append(body.create_shape(shape_def, geometry))
        self._bodies.append(body)
        self._owner.append(i)
        return body

    # -- the API -----------------------------------------------------------------

    @classmethod
    def description(cls):
        """what the env is about (its module's docstring)"""
        return (sys.modules[cls.__module__].__doc__ or "").strip()

    def reset(self, seed=None):
        """a new episode for all agents -> obs (n, obs_dim)"""
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        self.world.restore_from_snapshot(self.start)  # (re-enables every body)
        self.steps = 0
        self.done[:] = False
        self.episode_return[:] = 0.0
        self._leader = None
        self._reset(self.rng)
        return self._observe()

    def step(self, actions):
        """apply `actions` (n, act_dim) for one env step ->
        (obs (n, obs_dim), reward (n,), terminated (n,), truncated (n,)); agents whose
        episode already ended get reward 0 and no new end"""
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)
        if self.freeze_done and self.done.any():
            actions = np.where(self.done[:, None], 0, actions)
        self._apply(actions)
        self.world.step(self.dt, self.substeps)
        self.steps += 1
        reward, terminated = self._outcome()
        reward = np.where(self.done, 0.0, reward)
        terminated = terminated & ~self.done
        truncated = np.full(self.n, self.steps >= self.max_steps) & ~self.done
        ended = terminated | truncated
        self.episode_return += reward
        if ended.any():
            self.done |= ended
            if self.freeze_done:
                self._bodies[np.isin(self._owner, np.flatnonzero(ended))].enabled = False
        return self._observe(), reward, terminated, truncated

    def set_leader(self, i):
        """draw agent i (None: none) in LEADER_COLOR"""
        if i == self._leader:
            return
        if self._leader is not None:
            self._shapes[self._leader].custom_color = 0
        if i is not None:
            self._shapes[i].custom_color = LEADER_COLOR
        self._leader = i

    def focus(self, i=0):
        """a world point of agent i the camera follows (None: fixed camera)"""
        return None

    def human_action(self, keys):
        """the action (act_dim,) of a human pressing `keys` (a set of names like "left",
        "up", "space"): default zeros (no keyboard play)"""
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def close(self):
        """free the box2d world now (box2d has at most b2d.MAX_WORLDS = 128 worlds at a
        time; don't wait for the garbage collector)"""
        self.world.destroy()


ENVS = {}


def register_env(name):
    """class decorator: make the env available as make(name)"""

    def register(cls):
        ENVS[name] = cls
        return cls

    return register


def make(name, n=1, **kwargs):
    """a new env with n agents"""
    if name not in ENVS:
        import webgym.envs  # noqa: F401 (registers the built-in envs)
    return ENVS[name](n=n, **kwargs)
