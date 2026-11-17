"""Every env: batched API shapes, random actions, determinism, agents independent,
finished agents frozen."""

import numpy as np
import pytest

from webgym.core import ENVS, make

import webgym.envs  # noqa: F401 (registers the envs)

NAMES = sorted(ENVS)


@pytest.mark.parametrize("name", NAMES)
def test_random_actions(name):
    """reset / step shapes and types over a few episodes of random actions (8 agents)"""
    env = make(name, n=8)
    rng = np.random.default_rng(0)
    obs = env.reset(seed=0)
    assert obs.shape == (8, *env.observation_space.shape) and obs.dtype == np.float32
    for _ in range(300):
        obs, reward, terminated, truncated = env.step(env.action_space.sample(rng, 8))
        assert obs.shape == (8, *env.observation_space.shape) and np.isfinite(obs).all()
        assert reward.shape == terminated.shape == truncated.shape == (8,)
        if env.done.all():
            obs = env.reset()
    env.close()


@pytest.mark.parametrize("name", NAMES)
def test_seed_determinism(name):
    """same seed + actions -> same observations (snapshot reset restores everything)"""

    def run(env):
        env.reset(seed=3)
        return np.stack([env.step(np.full((env.n, *env.action_space.shape), 0.3))[0] for _ in range(30)])

    env = make(name, n=4)
    first = run(env)
    assert np.array_equal(first, run(env))  # after a reset of the same env
    assert np.array_equal(first, run(make(name, n=4)))  # a fresh env


@pytest.mark.parametrize("name", NAMES)
def test_agents_are_independent(name):
    """the agents of one world don't interact: other agents' actions don't change what
    happens to agent 0"""
    env = make(name, n=5)

    def run(others):
        env.reset(seed=7)
        out = []
        for _ in range(60):
            actions = np.zeros((5, *env.action_space.shape), dtype=np.float32)
            actions[1:] = others.uniform(-1, 1, (4, *env.action_space.shape))
            out.append(env.step(actions)[0][0])
        return np.stack(out)

    assert np.allclose(run(np.random.default_rng(1)), run(np.random.default_rng(99)), atol=1e-5)


def test_finished_agents_are_frozen():
    """an agent whose episode ended gets no more reward and no new end"""
    env = make("CartPole", n=3)
    env.reset(seed=0)
    while not env.done.any():
        actions = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)  # agent 0 pushes: falls first
        env.step(actions)
    first = np.flatnonzero(env.done)
    returns = env.episode_return[first].copy()
    _, reward, terminated, truncated = env.step(np.zeros((3, 1), np.float32))
    assert (reward[first] == 0).all() and not terminated[first].any() and not truncated[first].any()
    assert np.array_equal(env.episode_return[first], returns)
