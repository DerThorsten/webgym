"""Headless training: python -m webgym.cli CartPole --algo cem"""

import argparse
import time

from webgym.algos import ALGOS
from webgym.core import make
from webgym.policies import MLP


def train(env_name, algo="cem", population=64, iterations=30, seed=0, log=print):
    """run a trainer to the end without drawing; returns the last stats"""
    env = make(env_name, n=population)
    spec = ALGOS[algo]
    params = spec.defaults()
    hidden = int(params.pop("hidden", 16))
    policy = MLP(env.observation_space.shape[0], env.action_space.shape[0], hidden=(hidden,))
    stats, start = None, time.perf_counter()
    for item in spec.train(env, policy, iterations=iterations, seed=seed, **params):
        if item is None:
            continue
        stats = item
        losses = "  ".join(f"{k} {v:.3f}" for k, v in stats.get("losses", {}).items())
        log(
            f"iter {stats['iteration']:3d}  mean {stats['mean_return']:7.1f}  "
            f"best {stats['best_return']:7.1f}  {losses}  ({time.perf_counter() - start:.1f} s)"
        )
    env.close()
    return stats


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", nargs="?", default="CartPole")
    parser.add_argument("--algo", default="cem", choices=sorted(ALGOS))
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.env, args.algo, args.population, args.iterations, args.seed)


if __name__ == "__main__":
    main()
