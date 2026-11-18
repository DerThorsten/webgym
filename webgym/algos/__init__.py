"""Trainers. Each is a generator `train(env, policy, state=None, seed=0, **params)` on
a batched env (env.n agents): it yields None after every env step (so a frame loop
can stop it any time) and a stats dict at the end of every iteration
({"iteration", "mean_return", "best_return", "mean_params", "state", optional
"losses": {name: value}, optional "top_params": the best candidates, best first, for
the display}); stats["state"] passed back as `state` resumes training.
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
    # True: acts with the env's discrete actions (env.discrete_actions), the policy MLP
    # outputs one value per action (argmax); only for envs that advertise them
    discrete: bool = False
    # True: can train on the GPU (WebGPU): train(..., device="webgpu" | "cpu")
    gpu: bool = False

    def supports(self, env):
        """can it train this env (class or instance)?"""
        return not self.discrete or env.discrete_actions is not None

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

# what the parameters mean (tooltips in the UI)
HELP = {
    "hidden": "Neurons in the hidden layer of the network. More: can learn more complex "
              "behaviour, but slower and needs more data. Starts a new run.",
    "elite_frac": "Fraction of the best candidates the next generation is sampled around. "
                  "Small: greedy and fast, but can get stuck.",
    "init_std": "How widely the first generation's weights are spread. Large: explores "
                "more at the start.",
    "min_std": "The spread never gets smaller than this: keeps exploring a little "
               "(0: may converge too early).",
    "sigma": "Size of the random weight changes that are tried. Large: explores more, "
             "but the gradient estimate gets noisier.",
    "lr": "Learning rate: the step size of each update. Too large: unstable; too small: "
          "slow.",
    "gamma": "Discount: how much future rewards count compared to immediate ones "
             "(0.99: looks ~100 steps ahead).",
    "clip": "PPO clip range: how far one update may move the policy from the one that "
            "collected the data. Small: careful, stable updates.",
    "epochs": "How often each batch of experience is reused for updates. More: learns "
              "more from the data, but may overfit it.",
    "eps_decay": "Exploration: the chance of a random action is multiplied by this after "
                 "every iteration (starts at 1: all random).",
    "eps_min": "Exploration never drops below this chance of a random action.",
    "batch": "Experiences per gradient step (sampled from the replay buffer).",
    "updates": "Gradient steps per environment step. The agents add many experiences "
               "per step: more updates learn faster from them, but cost time.",
}
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
    "ppo_jax": Algo(
        "ppo_jax", "PPO (jax, CPU or GPU)",
        {"lr": (0.003, 0.0001, 0.01, 0.0001), "gamma": (0.99, 0.9, 0.999, 0.001),
         "clip": (0.2, 0.05, 0.5, 0.05), "epochs": (4, 1, 10, 1), "hidden": (32, 4, 256, 4)},
        requires="jax",
        gpu=True,
    ),
    "dqn_torch": Algo(
        "dqn_torch", "DQN (pytorch, discrete actions)",
        {"lr": (0.001, 0.0001, 0.01, 0.0001), "gamma": (0.99, 0.9, 0.999, 0.001),
         "eps_decay": (0.9, 0.5, 0.99, 0.01), "eps_min": (0.05, 0.0, 0.5, 0.01),
         "batch": (256, 32, 1024, 32), "updates": (4, 1, 16, 1), "hidden": (64, 8, 256, 8)},
        requires="pytorch",
        discrete=True,
    ),
}
