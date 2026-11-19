"""TD3 (twin delayed DDPG) with pytorch, for continuous actions: a deterministic
policy (an MLP with the layout of webgym.policies.MLP) acts with added noise, twin
Q-networks learn from a replay buffer, the policy follows the gradient of Q. The
"twin" and "delayed" parts make it stable: the smaller of two Q estimates, smoothed
target actions, and policy updates only every second critic update."""

import copy
import sys

import numpy as np
import torch

from webgym.algos.offpolicy import Replay, TwinQ, soft_update
from webgym.algos.ppo_torch import from_flat, mlp, to_flat

if sys.platform == "emscripten":
    torch.autograd.set_multithreading_enabled(False)  # (no threads in wasm)


def train(env, policy, iterations=1000, lr=3e-4, gamma=0.99, noise=0.1, batch=256, updates=2, tau=0.005,
          buffer=200_000, policy_delay=2, warmup=25_000, seed=0, state=None):
    """TD3 on the batched env (one episode per agent per iteration, `updates` gradient
    steps per env step; random actions for the first `warmup` experiences). Yields None per env step, stats per iteration; `state`
    resumes (the networks; the replay buffer starts empty)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sizes = policy.sizes
    actor = mlp(sizes, tanh_out=True)
    critic = TwinQ(sizes)
    start = 0
    if state is not None:
        from_flat(actor, state["actor"])
        critic.from_flat(state["critic"])
        start = state["iteration"]
    actor_target, critic_target = copy.deepcopy(actor), copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    replay = Replay(buffer, sizes[0], sizes[-1])
    gradient_steps = 0

    for iteration in range(start, start + iterations):
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        losses = {"q_loss": 0.0, "policy_loss": 0.0}
        count = 0
        while not env.done.all():
            alive = ~env.done
            # explore: random actions for the first experiences, then the policy's
            # action plus Gaussian noise
            if replay.count < warmup:
                a = rng.uniform(-1, 1, (env.n, sizes[-1])).astype(np.float32)
            else:
                with torch.no_grad():
                    a = actor(torch.as_tensor(obs))
                    a = (a + noise * torch.randn_like(a)).clamp(-1, 1).numpy()
            next_obs, reward, terminated, _ = env.step(a)
            replay.add(o=obs[alive], a=a[alive], r=reward[alive], o2=next_obs[alive], d=terminated[alive])
            obs = next_obs
            for _ in range(int(updates) if replay.count >= batch else 0):
                o, a_b, r, o2, d = replay.sample(rng, batch)
                # the critics learn reward + gamma * min Q of the targets at a smoothed
                # (noisy) next action
                with torch.no_grad():
                    a2 = actor_target(o2)
                    a2 = (a2 + (0.2 * torch.randn_like(a2)).clamp(-0.5, 0.5)).clamp(-1, 1)
                    y = r + gamma * (1 - d) * torch.min(*critic_target(o2, a2))
                q1, q2 = critic(o, a_b)
                q_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
                critic_optimizer.zero_grad()
                q_loss.backward()
                critic_optimizer.step()
                losses["q_loss"] += float(q_loss.detach())
                count += 1
                gradient_steps += 1
                # delayed: the policy (and the targets) only every policy_delay steps
                if gradient_steps % policy_delay == 0:
                    policy_loss = -critic(o, actor(o))[0].mean()
                    actor_optimizer.zero_grad()
                    policy_loss.backward()
                    actor_optimizer.step()
                    soft_update(actor_target, actor, tau)
                    soft_update(critic_target, critic, tau)
                    losses["policy_loss"] += float(policy_loss.detach()) * policy_delay
            yield None
        returns = env.episode_return
        t = iteration + 1
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": to_flat(actor),
            "losses": {k: v / max(count, 1) for k, v in losses.items()},
            "state": {"iteration": t, "actor": to_flat(actor), "critic": critic.to_flat()},
        }
