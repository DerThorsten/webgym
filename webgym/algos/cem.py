"""Cross-entropy method: sample a population of policy parameters around a mean,
run one episode each (env i <-> candidate i), refit mean / std to the best ones."""

import numpy as np


def rollout(envs, policy, params, seed):
    """one episode per env with params[i] -> yields None per step, returns the returns.
    (Finished envs are not stepped any more.)"""
    obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    returns = np.zeros(len(envs))
    alive = np.ones(len(envs), dtype=bool)
    while alive.any():
        idx = np.flatnonzero(alive)
        actions = policy(params[idx], obs[idx])
        for j, i in enumerate(idx):
            obs[i], reward, terminated, truncated, _ = envs[i].step(actions[j])
            returns[i] += reward
            alive[i] = not (terminated or truncated)
        yield None
    return returns


def train(envs, policy, iterations=1000, elite_frac=0.2, init_std=0.5, min_std=0.02, seed=0):
    """CEM over the flat parameters of `policy`; population size = len(envs).
    Yields None per env step and {"iteration", "mean_return", "best_return", "best_params"}
    after each iteration."""
    rng = np.random.default_rng(seed)
    mean = np.zeros(policy.num_params)
    std = np.full(policy.num_params, init_std)
    n_elite = max(2, int(len(envs) * elite_frac))
    for iteration in range(iterations):
        params = mean + std * rng.standard_normal((len(envs), policy.num_params))
        returns = yield from rollout(envs, policy, params, seed=int(rng.integers(1 << 30)))
        elite = params[np.argsort(returns)[-n_elite:]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), min_std)
        yield {
            "iteration": iteration + 1,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "best_params": params[np.argmax(returns)],
            "mean_params": mean,
        }
