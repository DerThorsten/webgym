"""Headless training: python -m webgym.cli CartPole --algo cem"""

import argparse
import time

from webgym.algos import ALGOS
from webgym.core import make
from webgym.policies import MLP


def train(env_name, algo="cem", population=64, iterations=30, seed=0, log=print):
    """run a trainer to the end without drawing; returns the last stats"""
    envs = [make(env_name) for _ in range(population)]
    space = envs[0].observation_space, envs[0].action_space
    policy = MLP(space[0].shape[0], space[1].shape[0])
    stats, steps, start = None, 0, time.perf_counter()
    for item in ALGOS[algo](envs, policy, iterations=iterations, seed=seed):
        if item is None:
            steps += 1
            continue
        stats = item
        log(
            f"iter {stats['iteration']:3d}  mean {stats['mean_return']:7.1f}  "
            f"best {stats['best_return']:7.1f}  ({time.perf_counter() - start:.1f} s)"
        )
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
