"""PPO (proximal policy optimization) with pytorch: a Gaussian policy (the mean is an
MLP with the layout of webgym.policies.MLP), a value network, one episode per agent
per iteration, advantages with GAE, clipped policy updates (Adam). Yields between
minibatches too: the frame loop stays responsive during the updates."""

import sys

import numpy as np
import torch
from torch import nn

if sys.platform == "emscripten":
    torch.autograd.set_multithreading_enabled(False)  # (no threads in wasm)


def mlp(sizes, tanh_out):
    """Linear + tanh layers; tanh after the last one too if `tanh_out`"""
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if tanh_out or i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def to_flat(net):
    """the weights of `net` as one vector in the MLP layout (W (in, out), bias)"""
    parts = []
    for layer in net:
        if isinstance(layer, nn.Linear):
            parts += [layer.weight.detach().T.reshape(-1), layer.bias.detach()]
    return torch.cat(parts).double().numpy()


def from_flat(net, flat):
    """load weights from to_flat's layout"""
    i = 0
    for layer in net:
        if isinstance(layer, nn.Linear):
            out_dim, in_dim = layer.weight.shape
            layer.weight.data = torch.as_tensor(flat[i : i + in_dim * out_dim], dtype=torch.float32).reshape(in_dim, out_dim).T.contiguous()
            i += in_dim * out_dim
            layer.bias.data = torch.as_tensor(flat[i : i + out_dim], dtype=torch.float32)
            i += out_dim


def train(env, policy, iterations=1000, lr=3e-3, gamma=0.99, clip=0.2, epochs=4, seed=0, state=None,
          lam=0.95, minibatch=2048):
    """PPO on the batched env; `policy`: the numpy MLP whose layout the actor uses.
    Yields None per env step / minibatch, stats per iteration (with losses); `state`
    resumes (the weights; the optimizer starts fresh)."""
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
    # two optimizers: the critic's loss (squared returns, large when rewards add up over
    # long episodes) must not drown the actor's gradient in a shared clip
    actor_params = [*actor.parameters(), log_std]
    actor_optimizer = torch.optim.Adam(actor_params, lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    for iteration in range(start, start + iterations):
        # one episode per agent
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        O, A, LP, R, V, M = [], [], [], [], [], []
        while not env.done.all():
            alive = ~env.done
            with torch.no_grad():
                o = torch.as_tensor(obs)
                mean, std = actor(o), log_std.exp()
                a = mean + std * torch.randn_like(mean)
                lp = torch.distributions.Normal(mean, std).log_prob(a).sum(-1)
                v = critic(o).squeeze(-1)
            next_obs, reward, _, _ = env.step(a.numpy())
            O.append(o), A.append(a), LP.append(lp), V.append(v)
            R.append(torch.as_tensor(reward, dtype=torch.float32)), M.append(torch.as_tensor(alive))
            obs = next_obs
            yield None
        O, A, LP, R, V, M = (torch.stack(x) for x in (O, A, LP, R, V, M))  # (T, n, ...)

        # advantages (GAE); an agent's episode ends where it is not alive any more
        T = len(R)
        adv = torch.zeros_like(R)
        last = torch.zeros(env.n)
        for t in reversed(range(T)):
            next_alive = M[t + 1].float() if t + 1 < T else torch.zeros(env.n)
            next_v = V[t + 1] if t + 1 < T else torch.zeros(env.n)
            delta = R[t] + gamma * next_v * next_alive - V[t]
            last = delta + gamma * lam * next_alive * last
            adv[t] = last
        ret = adv + V
        valid = M.reshape(-1)
        o, a, lp_old = O.reshape(-1, O.shape[-1])[valid], A.reshape(-1, A.shape[-1])[valid], LP.reshape(-1)[valid]
        adv, ret = adv.reshape(-1)[valid], ret.reshape(-1)[valid]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # clipped policy updates
        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        count = 0
        for _ in range(int(epochs)):
            for idx in torch.randperm(len(o)).split(minibatch):
                dist = torch.distributions.Normal(actor(o[idx]), log_std.exp())
                ratio = (dist.log_prob(a[idx]).sum(-1) - lp_old[idx]).exp()
                policy_loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                value_loss = (critic(o[idx]).squeeze(-1) - ret[idx]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
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
                losses["entropy"] += float(entropy.detach())
                count += 1
                yield None
        returns = env.episode_return
        t = iteration + 1
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": to_flat(actor),
            "losses": {k: v / max(count, 1) for k, v in losses.items()},
            "state": {"iteration": t, "actor": to_flat(actor), "critic": to_flat(critic),
                      "log_std": log_std.detach().double().numpy()},
        }
