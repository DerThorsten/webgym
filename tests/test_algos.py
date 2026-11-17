"""Trainers: CEM solves CartPole quickly and can be resumed."""

from webgym.algos import cem
from webgym.cli import train
from webgym.core import make
from webgym.policies import MLP


def test_cem_solves_cartpole():
    """the population's mean return gets close to the maximum (500)"""
    stats = train("CartPole", "cem", population=64, iterations=25, seed=0, log=lambda _: None)
    assert stats["mean_return"] > 300


def test_cem_resumes():
    """a run continued from stats["state"] goes on with the iteration count and the
    distribution where it stopped"""
    env = make("CartPole", n=16)
    policy = MLP(4, 1)
    stats = [s for s in cem(env, policy, iterations=3) if s is not None]
    state = stats[-1]["state"]
    more = [s for s in cem(env, policy, iterations=2, state=state, seed=1) if s is not None]
    assert [s["iteration"] for s in more] == [4, 5]
