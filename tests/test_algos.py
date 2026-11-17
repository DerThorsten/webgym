"""Trainers: CEM solves CartPole quickly."""

from webgym.cli import train


def test_cem_solves_cartpole():
    """the population's mean return gets close to the maximum (500)"""
    stats = train("CartPole", "cem", population=64, iterations=25, seed=0, log=lambda _: None)
    assert stats["mean_return"] > 300
