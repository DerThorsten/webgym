"""A2C (synchronous advantage actor-critic) with pytorch: a Gaussian policy (the mean
is an MLP with the layout of webgym.policies.MLP) and a value network. All agents act
together for a few steps (`steps`), then one update: the value network learns the
n-step returns, the policy makes actions more likely whose return beat the value
estimate (the advantage). Many small updates per episode: learns while it plays."""

import sys

import numpy as np
import torch
from torch import nn

from webgym.algos.ppo_torch import RewardScale, from_flat, mlp, to_flat

if sys.platform == "emscripten":
    torch.autograd.set_multithreading_enabled(False)  # (no threads in wasm)


def train(env, policy, iterations=1000, lr=3e-3, gamma=0.99, steps=8, normalize=True, scale_rewards=True, seed=0,
          state=None):
    """A2C on the batched env (an iteration: one episode per agent, an update every
    `steps` env steps). Yields None per env step, stats per iteration; `state`
    resumes (the weights; the optimizers start fresh)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sizes = policy.sizes
    actor = mlp(sizes, tanh_out=True)
    critic = mlp([sizes[0], *sizes[1:-1], 1], tanh_out=False)
    log_std = nn.Parameter(torch.full((sizes[-1],), -0.5))
    start = 0
    if state is not None:
        from_flat(actor, state["actor"])
        from_flat(critic, state["critic"])
        log_std.data = torch.as_tensor(state["log_std"], dtype=torch.float32)
        start = state["iteration"]
    scale = RewardScale(gamma)  # (rewards of any size: values the critic can learn, see ppo_torch)
    actor_params = [*actor.parameters(), log_std]
    actor_optimizer = torch.optim.Adam(actor_params, lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    for iteration in range(start, start + iterations):
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        losses = {"policy_loss": 0.0, "value_loss": 0.0}
        count = 0
        while not env.done.all():
            # a short rollout of all agents
            O, A, R, D, M, O2 = [], [], [], [], [], []
            for _ in range(int(steps)):
                if env.done.all():
                    break
                alive = ~env.done
                o = torch.as_tensor(obs)
                with torch.no_grad():
                    a = actor(o) + log_std.exp() * torch.randn(env.n, sizes[-1])
                obs, reward, terminated, _ = env.step(a.numpy())
                if scale_rewards:  # (for learning only: the returns reported stay the env's)
                    reward = scale(reward, alive)
                O.append(o), A.append(a), M.append(torch.as_tensor(alive)), O2.append(torch.as_tensor(obs))
                R.append(torch.as_tensor(reward, dtype=torch.float32))
                D.append(torch.as_tensor(terminated, dtype=torch.float32))
                yield None
            O, A, R, D, M, O2 = (torch.stack(x) for x in (O, A, R, D, M, O2))  # (T, n, ...)

            # n-step returns, from the value of the last observation backwards; where
            # an episode ended early by a time limit, from the value of its last one
            with torch.no_grad():
                v_next = critic(O2).squeeze(-1)
            ret = torch.zeros_like(R)
            G = v_next[-1]
            for t in reversed(range(len(R))):
                if t + 1 < len(R):
                    G = torch.where(M[t + 1], G, v_next[t])
                G = R[t] + gamma * (1 - D[t]) * G
                ret[t] = G

            # one update on the steps where the agents were alive
            o, a, ret = O[M], A[M], ret[M]
            v = critic(o).squeeze(-1)
            adv = (ret - v).detach()
            if normalize and len(adv) > 1:  # (any reward scale; an option in Stable-Baselines3)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            log_prob = torch.distributions.Normal(actor(o), log_std.exp()).log_prob(a).sum(-1)
            policy_loss = -(log_prob * adv).mean()
            value_loss = (v - ret).pow(2).mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor_params, 0.5)
            actor_optimizer.step()
            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            losses["policy_loss"] += float(policy_loss.detach())
            losses["value_loss"] += float(value_loss.detach())
            count += 1
        returns = env.episode_return
        t = iteration + 1
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": to_flat(actor),
            "losses": {**{k: v / max(count, 1) for k, v in losses.items()},
                       "std": float(log_std.detach().exp().mean())},
            "state": {"iteration": t, "actor": to_flat(actor), "critic": to_flat(critic),
                      "log_std": log_std.detach().double().numpy()},
        }
