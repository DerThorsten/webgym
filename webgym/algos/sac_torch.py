"""SAC (soft actor-critic) with pytorch, for continuous actions: the policy samples
actions = tanh(mean + std * noise) (the mean is an MLP with the layout of
webgym.policies.MLP; watching acts with tanh(mean)), twin Q-networks learn from a
replay buffer, and the policy maximizes Q + entropy (it stays random where that
costs little). The weight of the entropy (alpha) tunes itself towards a target
randomness."""

import copy
import sys

import numpy as np
import torch

from webgym.algos.offpolicy import Replay, TwinQ, soft_update
from webgym.algos.ppo_torch import from_flat, mlp, to_flat

if sys.platform == "emscripten":
    torch.autograd.set_multithreading_enabled(False)  # (no threads in wasm)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # (as CleanRL)


def train(env, policy, iterations=1000, lr=3e-4, q_lr=1e-3, gamma=0.99, batch=256, updates=2, tau=0.005,
          buffer=200_000, warmup=5000, policy_delay=2, seed=0, state=None):
    """SAC on the batched env (one episode per agent per iteration, `updates` gradient
    steps per env step; random actions for the first `warmup` experiences; the policy
    and alpha learn every `policy_delay` critic steps, as in CleanRL). Yields None per
    env step, stats per iteration; `state` resumes (networks, alpha; the replay buffer
    starts empty)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sizes = policy.sizes
    actor = mlp(sizes, tanh_out=False)  # the mean before the tanh
    # the spread of the actions, per state: a second network (its tanh output scaled
    # to log std in [LOG_STD_MIN, LOG_STD_MAX]); watching only needs the mean
    spread = mlp(sizes, tanh_out=True)
    critic = TwinQ(sizes)
    log_alpha = torch.zeros(1, requires_grad=True)
    start = 0
    if state is not None:
        from_flat(actor, state["actor"])
        from_flat(spread, state["spread"])
        critic.from_flat(state["critic"])
        log_alpha.data = torch.as_tensor(state["log_alpha"], dtype=torch.float32)
        start = state["iteration"]
    target = copy.deepcopy(critic)
    target_entropy = -float(sizes[-1])
    actor_optimizer = torch.optim.Adam([*actor.parameters(), *spread.parameters()], lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=q_lr)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=q_lr)
    replay = Replay(buffer, sizes[0], sizes[-1])
    gradient_steps = 0

    def sample(o):
        """random actions in [-1, 1] and their log-probabilities"""
        mean = actor(o)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (spread(o) + 1)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        # the tanh squashes the Gaussian: correct its density
        log_prob = torch.distributions.Normal(mean, std).log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1)

    for iteration in range(start, start + iterations):
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        losses = {"q_loss": 0.0, "policy_loss": 0.0}
        count = 0
        while not env.done.all():
            alive = ~env.done
            if replay.count < warmup:  # (first random actions: a varied buffer to start from)
                a = rng.uniform(-1, 1, (env.n, sizes[-1])).astype(np.float32)
            else:
                with torch.no_grad():
                    a = sample(torch.as_tensor(obs))[0].numpy()
            next_obs, reward, terminated, _ = env.step(a)
            replay.add(o=obs[alive], a=a[alive], r=reward[alive], o2=next_obs[alive], d=terminated[alive])
            obs = next_obs
            for _ in range(int(updates) if replay.count >= batch else 0):
                o, a_b, r, o2, d = replay.sample(rng, batch)
                alpha = log_alpha.exp().detach()
                # the critics learn reward + gamma * (soft value of the next state)
                with torch.no_grad():
                    a2, log_prob2 = sample(o2)
                    y = r + gamma * (1 - d) * (torch.min(*target(o2, a2)) - alpha * log_prob2)
                q1, q2 = critic(o, a_b)
                q_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
                critic_optimizer.zero_grad()
                q_loss.backward()
                critic_optimizer.step()
                gradient_steps += 1
                if gradient_steps % policy_delay == 0:
                    # the policy: high Q and high entropy
                    a_new, log_prob = sample(o)
                    policy_loss = (alpha * log_prob - torch.min(*critic(o, a_new))).mean()
                    actor_optimizer.zero_grad()
                    policy_loss.backward()
                    actor_optimizer.step()
                    # alpha: up when the policy is less random than the target, else down
                    alpha_loss = -(log_alpha.exp() * (log_prob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    losses["policy_loss"] += float(policy_loss.detach()) * policy_delay
                soft_update(target, critic, tau)
                losses["q_loss"] += float(q_loss.detach())
                count += 1
            yield None
        returns = env.episode_return
        t = iteration + 1
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": to_flat(actor),  # watching: tanh(mean), the MLP with tanh output
            "losses": {**{k: v / max(count, 1) for k, v in losses.items()}, "alpha": float(log_alpha.detach().exp())},
            "state": {"iteration": t, "actor": to_flat(actor), "spread": to_flat(spread),
                      "critic": critic.to_flat(), "log_alpha": log_alpha.detach().double().numpy()},
        }
