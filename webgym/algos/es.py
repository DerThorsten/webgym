"""Evolution strategies: perturb the policy weights with mirrored Gaussian noise
(+eps / -eps), run one episode per candidate (agent i <-> candidate i), step the
weights along the return-weighted noise (rank-shaped returns, Adam)."""

import numpy as np

from webgym.algos.cem import rollout


def ranks(returns):
    """returns -> centered ranks in [-0.5, 0.5] (robust to the scale of the returns)"""
    r = np.empty(len(returns))
    r[np.argsort(returns)] = np.arange(len(returns))
    return r / max(len(returns) - 1, 1) - 0.5


def train(env, policy, iterations=1000, sigma=0.1, lr=0.03, seed=0, state=None, forward=None):
    """ES over the flat parameters of `policy`; population = env.n (pairs of mirrored
    candidates, an odd last agent runs the current weights). `forward(params, obs)`:
    the batched policy (default: the numpy MLP; es_jax passes a jitted one).
    Yields None per env step, stats per iteration; `state` resumes."""
    forward = forward or policy
    rng = np.random.default_rng(seed)
    p = policy.num_params
    if state is None:
        state = {"iteration": 0, "theta": np.zeros(p), "m": np.zeros(p), "v": np.zeros(p)}
    theta, m, v, start = state["theta"], state["m"], state["v"], state["iteration"]
    half = env.n // 2
    for iteration in range(start, start + iterations):
        eps = rng.standard_normal((half, p))
        noise = np.concatenate([eps, -eps])
        params = np.concatenate([theta + sigma * noise, theta[None].repeat(env.n - 2 * half, 0)])
        returns = yield from rollout(env, forward, params, seed=int(rng.integers(1 << 30)))
        grad = ranks(returns[: 2 * half]) @ noise / (2 * half * sigma)
        # Adam (ascent)
        t = iteration + 1
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        theta = theta + lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": theta,
            "state": {"iteration": t, "theta": theta, "m": m, "v": v},
        }
