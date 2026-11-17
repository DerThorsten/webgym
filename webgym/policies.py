"""Small numpy MLP policies, evaluated for a whole population at once."""

import numpy as np


class MLP:
    """obs -> tanh hidden layers -> tanh output in [-1, 1]; all weights in one flat
    vector (what evolution strategies optimize)"""

    def __init__(self, obs_dim, act_dim, hidden=(16,)):
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
        for w, b in self.unflatten(np.atleast_2d(params)):
            x = np.tanh(np.einsum("pa,pab->pb", x, w) + b)
        return x
