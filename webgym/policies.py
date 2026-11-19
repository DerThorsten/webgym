"""Small numpy policies (an MLP, or a Graph of any shape for NEAT), evaluated for a
whole population at once: all weights of a policy are one flat vector."""

import numpy as np


def policy_for(env, discrete=False, hidden=16, layers=1, kind="mlp"):
    """the policy an algorithm trains for `env`: an MLP with `layers` hidden layers of
    `hidden` neurons, actions (tanh output) or, `discrete`, one Q-value per discrete
    action (linear output); kind "graph": a Graph with up to `hidden` hidden neurons"""
    obs_dim = env.observation_space.shape[0]
    if kind == "graph":
        return Graph(obs_dim, env.action_space.shape[0], int(hidden))
    sizes = (int(hidden),) * int(layers)
    if discrete:
        return MLP(obs_dim, len(env.discrete_actions), hidden=sizes, tanh_out=False)
    return MLP(obs_dim, env.action_space.shape[0], hidden=sizes)


def from_sizes(kind, sizes, tanh_out=True):
    """the policy with these `sizes` (of a checkpoint, a helper's message)"""
    if kind == "graph":
        return Graph(sizes[0], sizes[-1], sizes[1])
    return MLP(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]), tanh_out=tanh_out)


class MLP:
    """obs -> tanh hidden layers -> tanh output in [-1, 1] (actions), or a linear
    output (`tanh_out=False`: Q-values of discrete actions); all weights in one flat
    vector (what evolution strategies optimize)"""

    kind = "mlp"

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


class Graph:
    """a feed-forward network of any shape (NEAT, webgym/algos/neat.py): n_in inputs,
    up to `max_hidden` hidden neurons, n_out outputs, all tanh. Node index: inputs
    0 .. n_in - 1, hidden neurons n_in .. n_in + max_hidden - 1 (the used ones first),
    outputs after them; a connection goes from a lower to a higher index. The flat
    vector: weights (sources = inputs + hidden) x (targets = hidden + outputs), 0 = no
    connection; the targets' biases; the hidden neurons' ids (0: unused)."""

    kind = "graph"
    tanh_out = True

    def __init__(self, n_in, n_out, max_hidden=16):
        self.n_in, self.n_out, self.max_hidden = n_in, n_out, max_hidden
        self.sizes = [n_in, max_hidden, n_out]
        self.n_src, self.n_dst = n_in + max_hidden, max_hidden + n_out
        self.num_params = self.n_src * self.n_dst + self.n_dst + max_hidden

    def unflatten(self, params):
        """(P, num_params) -> weights (P, sources, targets), biases (P, targets), ids (P, max_hidden)"""
        k = self.n_src * self.n_dst
        weights = params[:, :k].reshape(len(params), self.n_src, self.n_dst)
        return weights, params[:, k : k + self.n_dst], params[:, k + self.n_dst :]

    def pack(self, edges, biases, ids):
        """one flat vector: `edges` [(from, to, weight)] and `biases` {node: bias} by node
        index, `ids` of the used hidden neurons (in index order)"""
        params = np.zeros((1, self.num_params))
        w, b, h = self.unflatten(params)
        for a, t, weight in edges:
            w[0, a, t - self.n_in] = weight
        for t, bias in biases.items():
            b[0, t - self.n_in] = bias
        h[0, : len(ids)] = ids
        return params[0]

    def forward(self, params, obs):
        """every node's value (P, n_in + max_hidden + n_out), node by node"""
        w, b, ids = self.unflatten(np.atleast_2d(params))
        used = int((ids > 0).sum(1).max()) if self.max_hidden else 0  # (unused neurons: skipped)
        x = np.zeros((len(w), self.n_src + self.n_out))
        x[:, : self.n_in] = obs
        for j in range(used):
            t = self.n_in + j  # (its sources all come before it)
            x[:, t] = np.tanh(np.einsum("ps,ps->p", x[:, :t], w[:, :t, j]) + b[:, j])
        h = self.max_hidden  # the outputs at once (no connections between them)
        x[:, self.n_src :] = np.tanh(np.einsum("ps,pso->po", x[:, : self.n_src], w[:, :, h:]) + b[:, h:])
        return x

    def __call__(self, params, obs):
        """actions (P, n_out) for P parameter vectors (P, num_params) and obs (P, n_in)"""
        return self.forward(params, obs)[:, self.n_src :]

    def view(self, params, obs):
        """the network of one vector for the page's view: nodes {"key" (stable: "i3",
        "h17" (the neuron's id), "o0"), "column" (inputs 0, hidden neurons 1 + the
        largest column of their sources, outputs last), "value" (for `obs`)} and
        connections [from, to (positions in nodes), weight]"""
        params = np.asarray(params, dtype=np.float64)
        w, _, ids = (a[0] for a in self.unflatten(params[None]))
        x = self.forward(params[None], np.asarray(obs)[None])[0]
        hidden = [self.n_in + j for j in range(int((ids > 0).sum()))]
        outputs = list(range(self.n_src, self.n_src + self.n_out))
        nodes = [*range(self.n_in), *hidden, *outputs]
        target = lambda t: t - self.n_in if t < self.n_src else t - self.n_src + self.max_hidden  # noqa: E731
        column = {i: 0 for i in range(self.n_in)}
        for t in hidden:
            column[t] = 1 + max([column[s] for s in range(t) if w[s, target(t)] != 0], default=0)
        last = 1 + max([column[t] for t in hidden], default=0)
        key = {**{i: f"i{i}" for i in range(self.n_in)}, **{o: f"o{o - self.n_src}" for o in outputs},
               **{t: f"h{int(ids[t - self.n_in])}" for t in hidden}}
        position = {node: k for k, node in enumerate(nodes)}
        edges = [[position[s], position[t], round(float(w[s, target(t)]), 3)]
                 for t in [*hidden, *outputs] for s in [*range(self.n_in), *hidden] if w[s, target(t)] != 0]
        return {
            "graph": True,
            "nodes": [{"key": key[i], "column": column.get(i, last), "value": round(float(x[i]), 3)} for i in nodes],
            "edges": edges,
        }
