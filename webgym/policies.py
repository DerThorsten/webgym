"""Small numpy MLP policies, evaluated for a whole population at once."""

import numpy as np


def policy_for(env, discrete=False, hidden=16, layers=1):
    """the MLP an algorithm trains for `env`: `layers` hidden layers of `hidden` neurons;
    actions (tanh output) or, `discrete`, one Q-value per discrete action (linear output)"""
    obs_dim = env.observation_space.shape[0]
    sizes = (int(hidden),) * int(layers)
    if discrete:
        return MLP(obs_dim, len(env.discrete_actions), hidden=sizes, tanh_out=False)
    return MLP(obs_dim, env.action_space.shape[0], hidden=sizes)


class MLP:
    """obs -> tanh hidden layers -> tanh output in [-1, 1] (actions), or a linear
    output (`tanh_out=False`: Q-values of discrete actions); all weights in one flat
    vector (what evolution strategies optimize)"""

    def __init__(self, obs_dim, act_dim, hidden=(16,), tanh_out=True):
        self.tanh_out = tanh_out
        self.sizes = [obs_dim, *hidden, act_dim]
        self.shapes = [(a, b) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.num_params = sum(a * b + b for a, b in self.shapes)

    def unflatten(self, params):
        """(P, num_params) -> list of (W (P, a, b), bias (P, b)) per layer"""
        layers, i = [], 0
        for a, b in self.shapes:
            w = params[:, i : i + a * b].reshape(-1, a, b)
            i += a * b
            layers.append((w, params[:, i : i + b]))
            i += b
        return layers

    def __call__(self, params, obs):
        """actions (P, act_dim) for P parameter vectors (P, num_params) and obs (P, obs_dim)"""
        x = obs
        layers = self.unflatten(np.atleast_2d(params))
        for i, (w, b) in enumerate(layers):
            x = np.einsum("pa,pab->pb", x, w) + b
            if self.tanh_out or i < len(layers) - 1:
                x = np.tanh(x)
        return x
