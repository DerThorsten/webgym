"""Minimal gymnasium-style env API (no gymnasium dependency) + env registry."""

import sys

import numpy as np
import pyb2d3 as b2d


class Box:
    """continuous space: arrays of `shape` in [low, high]"""

    def __init__(self, low, high, shape):
        self.shape = tuple(shape)
        self.low = np.broadcast_to(np.float32(low), self.shape)
        self.high = np.broadcast_to(np.float32(high), self.shape)

    def sample(self, rng):
        return rng.uniform(self.low, self.high).astype(np.float32)


class Discrete:
    """integers 0 .. n-1"""

    def __init__(self, n):
        self.n = n
        self.shape = ()

    def sample(self, rng):
        return int(rng.integers(self.n))


class Env:
    """base class: a box2d world stepped with a fixed dt per env step.

    Subclasses build `self.world` in `__init__` and implement `_reset(rng)`,
    `_apply(action)`, `_observe()` and `_outcome()` -> (reward, terminated).
    The module docstring of an env is its description in the UI.
    """

    observation_space = None
    action_space = None
    dt = 1 / 60
    substeps = 4
    max_steps = 1000
    # initial camera: world point at the view center, half height in world units
    view_center = (0.0, 0.0)
    view_zoom = 10.0

    @classmethod
    def description(cls):
        """what the env is about (its module's docstring)"""
        return (sys.modules[cls.__module__].__doc__ or "").strip()

    def focus(self):
        """a world point the camera follows (None: fixed camera)"""
        return None

    def human_action(self, keys):
        """the action for a human pressing `keys` (a set of names like "left", "up",
        "space"): default zeros (no keyboard play)"""
        return np.zeros(self.action_space.shape, dtype=np.float32)

    def close(self):
        """free the box2d world now (box2d has at most b2d.MAX_WORLDS = 128 worlds at a
        time; don't wait for the garbage collector)"""
        self.world.destroy()

    def reset(self, seed=None):
        """start a new episode -> (obs, info)"""
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.episode_return = 0.0
        self.done = False  # the episode ended (the view hides finished envs)
        self._reset(self.rng)
        return self._observe(), {}

    def step(self, action):
        """apply `action` for one env step -> (obs, reward, terminated, truncated, info)"""
        self._apply(action)
        self.world.step(self.dt, self.substeps)
        self.steps += 1
        reward, terminated = self._outcome()
        truncated = self.steps >= self.max_steps
        self.done = terminated or truncated
        self.episode_return += reward
        return self._observe(), reward, terminated, truncated, {}


ENVS = {}


def register_env(name):
    """class decorator: make the env available as make(name)"""

    def register(cls):
        ENVS[name] = cls
        return cls

    return register


def make(name, **kwargs):
    """a new env instance"""
    if name not in ENVS:
        import webgym.envs  # noqa: F401 (registers the built-in envs)
    return ENVS[name](**kwargs)


def new_world(**kwargs):
    """b2d.World(**kwargs) for an env, checked: box2d hands out an invalid world (no
    error) when all b2d.MAX_WORLDS (128) are in use, and using it corrupts memory"""
    world = b2d.World(**kwargs)
    if not world.is_valid:
        raise RuntimeError(f"too many box2d worlds (at most {b2d.MAX_WORLDS}): close() unused envs")
    return world
