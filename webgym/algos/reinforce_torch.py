"""REINFORCE (the textbook policy gradient) with pytorch: a Gaussian policy (the mean
is an MLP with the layout of webgym.policies.MLP), one episode per agent per
iteration, then one gradient step that makes the actions taken more likely the more
reward followed them. Baseline: the agents' mean return-to-go at the same time step;
the advantages are normalized."""

import sys

import numpy as np
import torch
from torch import nn

from webgym.algos.ppo_torch import from_flat, mlp, to_flat

if sys.platform == "emscripten":
    torch.autograd.set_multithreading_enabled(False)  # (no threads in wasm)


def train(env, policy, iterations=1000, lr=3e-3, gamma=0.99, seed=0, state=None):
    """REINFORCE on the batched env. Yields None per env step, stats per iteration;
    `state` resumes (the weights; the optimizer starts fresh)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = mlp(policy.sizes, tanh_out=True)
    log_std = nn.Parameter(torch.full((policy.sizes[-1],), -0.5))
    start = 0
    if state is not None:
        from_flat(actor, state["actor"])
        log_std.data = torch.as_tensor(state["log_std"], dtype=torch.float32)
        start = state["iteration"]
    optimizer = torch.optim.Adam([*actor.parameters(), log_std], lr=lr)

    for iteration in range(start, start + iterations):
        # one episode per agent; keep the log-probabilities of the actions taken (with
        # their gradient graph: the update needs it)
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        LP, R, M = [], [], []
        while not env.done.all():
            alive = ~env.done
            dist = torch.distributions.Normal(actor(torch.as_tensor(obs)), log_std.exp())
            a = dist.sample()
            obs, reward, _, _ = env.step(a.numpy())
            LP.append(dist.log_prob(a).sum(-1))
            R.append(torch.as_tensor(reward, dtype=torch.float32))
            M.append(torch.as_tensor(alive))
            yield None
        LP, R, M = torch.stack(LP), torch.stack(R), torch.stack(M)  # (T, n)

        # return-to-go: the discounted reward from each step to the episode's end
        ret = torch.zeros_like(R)
        G = torch.zeros(env.n)
        for t in reversed(range(len(R))):
            G = R[t] + gamma * G
            ret[t] = G
        # baseline: the mean return-to-go of the agents still alive at that step
        alive = M.float()
        baseline = (ret * alive).sum(1) / alive.sum(1).clamp(min=1)
        adv = (ret - baseline[:, None])[M]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss = -(LP[M] * adv).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        returns = env.episode_return
        t = iteration + 1
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": to_flat(actor),
            "losses": {"policy_loss": float(loss.detach()), "std": float(log_std.detach().exp().mean())},
            "state": {"iteration": t, "actor": to_flat(actor), "log_std": log_std.detach().double().numpy()},
        }
