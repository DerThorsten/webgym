"""Trainers: every algorithm learns CartPole a bit, resumes, exports numpy weights.
(torch / jax ones run only where those are installed: pixi run -e ml test)"""

import numpy as np
import pytest

from webgym.algos import ALGOS
from webgym.algos.cem import train as cem
from webgym.cli import train
from webgym.core import make
from webgym.policies import MLP


def test_cem_solves_cartpole():
    """the population's mean return gets close to the maximum (500)"""
    stats = train("CartPole", "cem", population=64, iterations=25, seed=0, log=lambda _: None)
    assert stats["mean_return"] > 300


def test_cem_resumes():
    """a run continued from stats["state"] goes on with the iteration count"""
    env = make("CartPole", n=16)
    policy = MLP(4, 1)
    stats = [s for s in cem(env, policy, iterations=3) if s is not None]
    more = [s for s in cem(env, policy, iterations=2, state=stats[-1]["state"], seed=1) if s is not None]
    assert [s["iteration"] for s in more] == [4, 5]


@pytest.mark.parametrize("name", sorted(ALGOS))
def test_every_algo_learns_and_resumes(name):
    """each algorithm improves CartPole within a few iterations, its numpy weights fit
    the MLP, and it resumes from its state"""
    algo = ALGOS[name]
    if not algo.available():
        pytest.skip(f"{algo.requires} not installed")
    params = algo.defaults()
    hidden = int(params.pop("hidden"))
    env = make("CartPole", n=32)
    policy = MLP(4, 1, hidden=(hidden,))
    stats = [s for s in algo.train(env, policy, iterations=12, seed=0, **params) if s is not None]
    assert stats[-1]["mean_return"] > stats[0]["mean_return"]
    assert stats[-1]["mean_params"].shape == (policy.num_params,)
    assert np.isfinite(policy(stats[-1]["mean_params"], np.zeros((1, 4), np.float32))).all()
    more = [s for s in algo.train(env, policy, iterations=1, seed=1, state=stats[-1]["state"], **params) if s]
    assert more[-1]["iteration"] == 13
