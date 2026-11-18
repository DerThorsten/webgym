"""Trainers: every algorithm learns CartPole a bit, resumes, exports numpy weights.
(torch / jax ones run only where those are installed: pixi run -e ml test)"""

import numpy as np
import pytest

from webgym.algos import ALGOS
from webgym.algos.cem import train as cem
from webgym.cli import train
from webgym.core import make
from webgym.policies import MLP, policy_for


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


def test_ppo_jax_update_through_iree():
    """the PPO update that goes to the GPU: compiled by IREE (here for the CPU, the
    browser compiles the same StableHLO for WebGPU) it computes what jax computes"""
    pytest.importorskip("jax")
    pytest.importorskip("iree.compiler")
    import iree.runtime as rt

    from webgym import gpu
    from webgym.algos import ppo_jax

    actor, critic = [4, 8, 1], [4, 8, 1]
    n = ppo_jax.count(actor) + ppo_jax.count(critic) + 1
    rng = np.random.default_rng(0)
    theta = (0.3 * rng.standard_normal(n)).astype(np.float32)
    zeros = np.zeros(n, np.float32)
    obs, act, lp, adv, ret, _ = (rng.standard_normal(x.shape).astype(np.float32) for x in ppo_jax.batch_arrays(64, 4, 1))
    mask = np.ones(64, np.float32)
    args = (theta, zeros, zeros, np.float32(0), obs, act, lp, adv, ret, mask, np.float32(3e-3), np.float32(0.2))
    update = ppo_jax.make_update(actor, critic)
    import jax

    expected = jax.jit(update)(*args)
    binary = gpu.compile_module(gpu.stablehlo(update, *args), "cpu")
    config = rt.Config("local-task")
    context = rt.SystemContext(config=config)
    module = rt.VmModule.copy_buffer(config.vm_instance, binary)
    context.add_vm_module(module)
    got = getattr(context.modules, module.name).main(*args)
    for e, g in zip(expected, got):
        np.testing.assert_allclose(np.asarray(g.to_host()), np.asarray(e), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", sorted(ALGOS))
def test_every_algo_learns_and_resumes(name):
    """each algorithm improves CartPole within a few iterations, its numpy weights fit
    the MLP, and it resumes from its state"""
    algo = ALGOS[name]
    if not algo.available():
        pytest.skip(f"{algo.requires} not installed")
    params = algo.defaults()
    env = make("CartPole", n=32)
    policy = policy_for(env, algo.discrete, params.pop("hidden"))
    stats = [s for s in algo.train(env, policy, iterations=12, seed=0, **params) if s is not None]
    assert max(s["mean_return"] for s in stats[-3:]) > stats[0]["mean_return"]
    assert stats[-1]["mean_params"].shape == (policy.num_params,)
    assert np.isfinite(policy(stats[-1]["mean_params"], np.zeros((1, 4), np.float32))).all()
    more = [s for s in algo.train(env, policy, iterations=1, seed=1, state=stats[-1]["state"], **params) if s]
    assert more[-1]["iteration"] == 13
