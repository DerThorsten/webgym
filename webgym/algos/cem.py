"""Cross-entropy method: sample a population of policy parameters around a mean, run
one episode each (agent i <-> candidate i of a batched env), refit mean / std to the
best ones."""

import numpy as np


def rollout(env, policy, params, seed):
    """one episode for all agents of `env` (agent i acts with params[i]); yields None
    per env step, returns the episode returns"""
    obs = env.reset(seed=seed)
    while not env.done.all():
        obs = env.step(policy(params, obs))[0]
        yield None
    return env.episode_return.copy()


def train(env, policy, iterations=1000, elite_frac=0.2, init_std=0.5, min_std=0.02, seed=0, state=None):
    """CEM over the flat parameters of `policy`; population size = env.n.
    Yields None per env step and after each iteration {"iteration", "mean_return",
    "best_return", "mean_params", "state"}; `state` (from such stats) resumes a run."""
    rng = np.random.default_rng(seed)
    if state is None:
        state = {"iteration": 0, "mean": np.zeros(policy.num_params), "std": np.full(policy.num_params, init_std)}
    mean, std, start = state["mean"], state["std"], state["iteration"]
    n_elite = max(2, int(env.n * elite_frac))
    for iteration in range(start, start + iterations):
        params = mean + std * rng.standard_normal((env.n, policy.num_params))
        returns = yield from rollout(env, policy, params, seed=int(rng.integers(1 << 30)))
        elite = params[np.argsort(returns)[-n_elite:]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), min_std)
        yield {
            "iteration": iteration + 1,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": mean,
            "state": {"iteration": iteration + 1, "mean": mean, "std": std},
        }
