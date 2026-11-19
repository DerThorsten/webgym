"""Trainers: every algorithm learns CartPole a bit, resumes, exports numpy weights.
(the torch ones run only where torch is installed: pixi run -e ml test)"""

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
    env = make("CartPole", n=32)
    policy = algo.policy(env, algo.defaults())
    params = {k: v for k, v in algo.defaults().items() if k not in algo.new_run}
    stats = [s for s in algo.train(env, policy, iterations=12, seed=0, **params) if s is not None]
    assert max(s["mean_return"] for s in stats[-3:]) > stats[0]["mean_return"]
    assert stats[-1]["mean_params"].shape == (policy.num_params,)
    assert np.isfinite(policy(stats[-1]["mean_params"], np.zeros((1, 4), np.float32))).all()
    more = [s for s in algo.train(env, policy, iterations=1, seed=1, state=stats[-1]["state"], **params) if s]
    assert more[-1]["iteration"] == 13


def test_sac_exports_the_squashed_mean():
    """SAC acts with tanh(mean + noise); the watcher's MLP (tanh output) computes
    tanh(mean) from the exported weights"""
    torch = pytest.importorskip("torch")
    from webgym.algos.ppo_torch import from_flat, mlp

    env = make("CartPole", n=8)
    policy = ALGOS["sac_torch"].policy(env, {"hidden": 8, "layers": 2})
    stats = [s for s in ALGOS["sac_torch"].train(env, policy, iterations=1, batch=32) if s is not None]
    mean = mlp(policy.sizes, tanh_out=False)
    from_flat(mean, stats[-1]["state"]["actor"])
    obs = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    expected = torch.tanh(mean(torch.as_tensor(obs))).detach().numpy()
    assert np.allclose(policy(stats[-1]["mean_params"], obs), expected, atol=1e-5)


def test_neat_grows_networks():
    """NEAT: the genomes get neurons; the Graph evaluates every genome as written
    (agrees with a plain node-by-node evaluation); its state survives JSON (checkpoints)"""
    import json

    from webgym.algos import neat
    from webgym.policies import Graph

    env = make("CartPole", n=32)
    policy = Graph(4, 1, max_hidden=8)
    stats = [s for s in neat.train(env, policy, iterations=15, add_node=0.3, seed=0) if s is not None]
    state = json.loads(json.dumps(stats[-1]["state"]))
    s = neat.load(state)
    assert max(len(g["nodes"]) for g in s["genomes"]) > 1  # (more than the output: neurons)
    assert all(len(g["nodes"]) - 1 <= 8 for g in s["genomes"])
    obs = np.random.default_rng(0).normal(size=(len(s["genomes"]), 4))
    params = np.stack([neat.encode(g, policy, s["order"]) for g in s["genomes"]])
    for g, o, a in zip(s["genomes"], obs, policy(params, obs)):
        value = dict(enumerate(o))
        for k in sorted(g["nodes"], key=lambda k: s["order"][k]):  # (by order: sources first)
            total = g["nodes"][k] + sum(w * value[i] for (i, j), (w, on) in g["conns"].items() if j == k and on)
            value[k] = np.tanh(total)
        assert np.isclose(a[0], value[4])
    more = [s for s in neat.train(env, policy, iterations=1, seed=1, state=state) if s]
    assert more[-1]["iteration"] == 16
