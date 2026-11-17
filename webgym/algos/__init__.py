"""Trainers. Each is a generator `train(env, policy, state=None, seed=0, **params)` on
a batched env (env.n agents): it yields None after every env step (so a frame loop
can stop it any time) and a stats dict at the end of every iteration
({"iteration", "mean_return", "best_return", "mean_params", "state", optional
"losses": {name: value}}); stats["state"] passed back as `state` resumes training.
"mean_params" are the flat weights of a webgym.policies.MLP (numpy), whatever
framework trained them: watching a policy never needs torch / jax.

Each algorithm declares its parameters (shown in the UI): {name: (default, min, max,
step)}; `new_run`: the ones that only apply to a new run (the network size);
`requires`: the package it needs ("pytorch", "jax": loaded on demand in the browser).
"""

import importlib
import importlib.util
from dataclasses import dataclass, field


@dataclass
class Algo:
    """a trainer and its description"""

    module: str  # webgym.algos.<module>.train
    label: str
    params: dict = field(default_factory=dict)
    new_run: tuple = ("hidden",)
    requires: str = None  # conda package name (its python module: IMPORTS)

    @property
    def train(self):
        """the trainer (imported on first use: torch / jax only when needed)"""
        return importlib.import_module(f"webgym.algos.{self.module}").train

    def available(self):
        """is the package it needs importable?"""
        return self.requires is None or importlib.util.find_spec(IMPORTS[self.requires]) is not None

    def defaults(self):
        return {name: spec[0] for name, spec in self.params.items()}


IMPORTS = {"pytorch": "torch", "jax": "jax"}
HIDDEN = (16, 4, 128, 4)  # hidden layer size of the policy MLP
ES_PARAMS = {"sigma": (0.1, 0.01, 0.5, 0.01), "lr": (0.03, 0.001, 0.2, 0.001), "hidden": HIDDEN}

ALGOS = {
    "cem": Algo(
        "cem", "CEM (numpy)",
        {"elite_frac": (0.2, 0.05, 0.5, 0.05), "init_std": (0.5, 0.05, 2.0, 0.05),
         "min_std": (0.02, 0.0, 0.5, 0.01), "hidden": HIDDEN},
    ),
    "es": Algo("es", "Evolution strategies (numpy)", ES_PARAMS),
    "es_jax": Algo("es_jax", "Evolution strategies (jax, jit)", ES_PARAMS, requires="jax"),
    "ppo_torch": Algo(
        "ppo_torch", "PPO (pytorch)",
        {"lr": (0.003, 0.0001, 0.01, 0.0001), "gamma": (0.99, 0.9, 0.999, 0.001),
         "clip": (0.2, 0.05, 0.5, 0.05), "epochs": (4, 1, 10, 1), "hidden": (32, 4, 128, 4)},
        requires="pytorch",
    ),
}
