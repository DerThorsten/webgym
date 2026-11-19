"""Shared by SAC and TD3 (off-policy actor-critics for continuous actions): the
replay buffer, the twin Q-networks and the slowly following target copies."""

import numpy as np
import torch
from torch import nn

from webgym.algos.ppo_torch import from_flat, mlp, to_flat


class Replay:
    """a ring buffer of transitions: obs, action, reward, next obs, terminated"""

    def __init__(self, size, obs_dim, act_dim):
        self.size, self.count, self.pos = size, 0, 0
        self.data = {"o": np.zeros((size, obs_dim), np.float32), "a": np.zeros((size, act_dim), np.float32),
                     "r": np.zeros(size, np.float32), "o2": np.zeros((size, obs_dim), np.float32),
                     "d": np.zeros(size, np.float32)}

    def add(self, **columns):
        at = (self.pos + np.arange(len(columns["o"]))) % self.size
        for name, value in columns.items():
            self.data[name][at] = value
        self.pos = (self.pos + len(at)) % self.size
        self.count = min(self.count + len(at), self.size)

    def sample(self, rng, batch):
        """a random minibatch as tensors (o, a, r, o2, d)"""
        j = rng.integers(self.count, size=batch)
        return [torch.as_tensor(self.data[name][j]) for name in ("o", "a", "r", "o2", "d")]


class TwinQ(nn.Module):
    """two Q-networks Q(obs, action) -> the long-run reward; the smaller of the two
    estimates counters the overestimation a single one tends to"""

    def __init__(self, sizes):
        super().__init__()
        q_sizes = [sizes[0] + sizes[-1], *sizes[1:-1], 1]
        self.q1, self.q2 = mlp(q_sizes, tanh_out=False), mlp(q_sizes, tanh_out=False)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def to_flat(self):
        return np.concatenate([to_flat(self.q1), to_flat(self.q2)])

    def from_flat(self, flat):
        half = len(flat) // 2
        from_flat(self.q1, flat[:half])
        from_flat(self.q2, flat[half:])


def soft_update(target, net, tau):
    """move the target network's weights a little (tau) towards the network's"""
    with torch.no_grad():
        for t, p in zip(target.parameters(), net.parameters()):
            t.mul_(1 - tau).add_(tau * p)
