"""Deep Q-learning (DQN) with pytorch, for envs with discrete actions
(env.discrete_actions): a Q-network estimates the long-run reward of every action, the
agents act epsilon-greedy (mostly the best action, sometimes a random one),
experiences go to a replay buffer, minibatches from it train the network towards
reward + gamma * max Q of a slowly updated target network.

`policy`: a webgym MLP with a linear output (one Q-value per action): its weights
are exported in that layout, watching acts with argmax (no torch needed)."""

import sys

import numpy as np
import torch
from torch import nn

from webgym.algos.ppo_torch import from_flat, mlp, to_flat

if sys.platform == "emscripten":
    torch.autograd.set_multithreading_enabled(False)  # (no threads in wasm)


def train(env, policy, iterations=1000, lr=1e-3, gamma=0.99, eps_decay=0.9, eps_min=0.05, batch=256,
          updates=4, buffer=50_000, target_every=200, seed=0, state=None):
    """DQN on the batched env (one episode per agent per iteration); epsilon starts at
    1 and is multiplied by eps_decay after every iteration. Yields None per env step,
    stats per iteration (losses: td_loss, epsilon); `state` resumes (network,
    epsilon; the replay buffer starts empty)."""
    actions = torch.as_tensor(env.discrete_actions)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = mlp(policy.sizes, tanh_out=False)
    target = mlp(policy.sizes, tanh_out=False)
    start, epsilon = 0, 1.0
    if state is not None:
        from_flat(q, state["q"])
        start, epsilon = state["iteration"], float(np.ravel(state["epsilon"])[0])
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    obs_dim = policy.sizes[0]
    B = {"o": np.zeros((buffer, obs_dim), np.float32), "a": np.zeros(buffer, np.int64),
         "r": np.zeros(buffer, np.float32), "o2": np.zeros((buffer, obs_dim), np.float32),
         "d": np.zeros(buffer, np.float32)}
    size, pos, gradient_steps = 0, 0, 0

    for iteration in range(start, start + iterations):
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        td_sum, td_count = 0.0, 0
        while not env.done.all():
            alive = ~env.done
            with torch.no_grad():
                greedy = q(torch.as_tensor(obs)).argmax(-1).numpy()
            explore = rng.random(env.n) < epsilon
            a = np.where(explore, rng.integers(len(actions), size=env.n), greedy)
            next_obs, reward, terminated, truncated = env.step(actions[a].numpy())
            # the alive agents' transitions -> the replay buffer (ring)
            idx = np.flatnonzero(alive)
            for name, value in (("o", obs[idx]), ("a", a[idx]), ("r", reward[idx]), ("o2", next_obs[idx]),
                                ("d", terminated[idx].astype(np.float32))):
                at = (pos + np.arange(len(idx))) % buffer
                B[name][at] = value
            pos, size = (pos + len(idx)) % buffer, min(size + len(idx), buffer)
            obs = next_obs
            # `updates` gradient steps on replayed minibatches (the agents add env.n
            # experiences per step)
            for _ in range(int(updates) if size >= batch else 0):
                j = rng.integers(size, size=batch)
                o, a_b, r = (torch.as_tensor(B[k][j]) for k in ("o", "a", "r"))
                o2, d = torch.as_tensor(B["o2"][j]), torch.as_tensor(B["d"][j])
                with torch.no_grad():
                    y = r + gamma * (1 - d) * target(o2).max(-1).values
                prediction = q(o).gather(1, a_b[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(prediction, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                optimizer.step()
                td_sum += float(loss.detach())
                td_count += 1
                gradient_steps += 1
                if gradient_steps % target_every == 0:
                    target.load_state_dict(q.state_dict())
            yield None
        epsilon = max(eps_min, epsilon * eps_decay)
        returns = env.episode_return
        t = iteration + 1
        yield {
            "iteration": t,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": to_flat(q),
            "losses": {"td_loss": td_sum / max(td_count, 1), "epsilon": epsilon},
            "state": {"iteration": t, "q": to_flat(q), "epsilon": np.array(epsilon)},
        }
