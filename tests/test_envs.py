"""Every env: API shapes, random actions, determinism with a seed."""

import numpy as np
import pytest

from webgym.core import ENVS, make

import webgym.envs  # noqa: F401 (registers the envs)


@pytest.mark.parametrize("name", sorted(ENVS))
def test_random_actions(name):
    """reset / step shapes and types over a few episodes of random actions"""
    env = make(name)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample(rng))
        assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
        assert isinstance(reward, float)
        if terminated or truncated:
            obs, info = env.reset()


@pytest.mark.parametrize("name", sorted(ENVS))
def test_seed_determinism(name):
    """same seed + actions -> same observations (snapshot reset restores everything)"""

    def run(env):
        env.reset(seed=3)
        return np.stack([env.step(np.full(env.action_space.shape, 0.3))[0] for _ in range(30)])

    env = make(name)
    first = run(env)
    assert np.array_equal(first, run(env))  # after a reset of the same env
    assert np.array_equal(first, run(make(name)))  # a fresh env
