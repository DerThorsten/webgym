"""Throughput benchmark: env steps per second, n single-agent envs (own worlds) vs one
env with n agents (one world, batch API). Runs natively and in a web worker
(web/bench.html).

    python -m webgym.bench
"""

import json
import time

import numpy as np

from webgym.core import make
from webgym.policies import MLP


def _run(envs, n, steps):
    """step `envs` (each with its share of the n agents) with a random MLP -> steps / s"""
    policy = MLP(envs[0].observation_space.shape[0], envs[0].action_space.shape[0])
    params = np.random.default_rng(0).standard_normal((n, policy.num_params)) * 0.1
    obs = np.concatenate([env.reset(seed=i) for i, env in enumerate(envs)])
    start = time.perf_counter()
    for _ in range(steps):
        actions = policy(params, obs)
        k = 0
        for env in envs:
            obs[k : k + env.n] = env.step(actions[k : k + env.n])[0]
            k += env.n
            if env.done.all():
                obs[k - env.n : k] = env.reset()
    seconds = time.perf_counter() - start
    for env in envs:
        env.close()
    return n * steps / seconds


def run(n=64, steps=300, kinds=("separate", "one_world"), env="CartPole"):
    """{kind: env steps per second} as JSON (for the web worker)"""
    make_envs = {
        "separate": lambda: [make(env) for _ in range(n)],
        "one_world": lambda: [make(env, n=n)],
    }
    return json.dumps({k: round(_run(make_envs[k](), n, steps)) for k in kinds})


if __name__ == "__main__":
    print(run())
