"""Headless training: python -m webgym.cli CartPole --algo cem
(env settings: python -m webgym.cli WaterCannon --set balls=2;
 a pretrained run for the page: python -m webgym.cli Walker --iterations 300
 --save web/public/pretrained/Walker-cem.safetensors)"""

import argparse
import time

from webgym.algos import ALGOS
from webgym.core import make


def train(env_name, algo="cem", population=64, iterations=30, seed=0, log=print, settings=None):
    """run a trainer to the end without drawing; returns the last stats"""
    env = make(env_name, n=population, **(settings or {}))
    spec = ALGOS[algo]
    policy = spec.policy(env, spec.defaults())
    params = {k: v for k, v in spec.defaults().items() if k not in spec.new_run}
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


def train_and_save(path, env_name, algo="cem", population=64, iterations=30, seed=0, settings=None, log=print,
                   minutes=None, hp=None):
    """train with the page's App (headless) and write its checkpoint to `path`: the
    page restores it like its own (e.g. web/public/pretrained/, see build_page.py). The
    algorithm's parameters: as the env suggests them (Env.suggested, any time budget),
    else its defaults. Stops after `iterations` or `minutes`; the checkpoint written is
    the one of the iteration with the best mean return (some algorithms fall back
    after their peak). `hp`: more parameters of the algorithm ({name: value})."""
    from pathlib import Path

    from webgym.app import App

    app = App(env=env_name, algo=algo, population=population, seed=seed, draws=False, settings=settings)
    for budget in ("quick", "medium", "long"):  # (the env's suggested parameters for it)
        app.budget = budget
        if algo in app.suggestions():
            app.suggest(algo)
            app.new_run()
            break
    for name, value in (hp or {}).items():
        app.command(f"hp.{name}", value)
    app.command("train")
    start, shown = time.perf_counter(), 0
    best, best_data = None, None
    while len(app.history) < iterations and (minutes is None or time.perf_counter() - start < 60 * minutes):
        if not app.training:  # (the trainer ended by itself, e.g. after 1000 iterations: go on)
            app.command("train")
        app._train_steps()
        for entry in app.history[shown:]:
            log(f"iter {entry['iteration']:3d}  mean {entry['mean_return']:7.1f}  "
                f"best {entry['best_return']:7.1f}  ({time.perf_counter() - start:.1f} s)")
        if len(app.history) > shown and (best is None or app.history[-1]["mean_return"] > best["mean_return"]):
            best, best_data = app.history[-1], app.checkpoint()
        shown = len(app.history)
    app.command("stop")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(best_data if best_data is not None else app.checkpoint())
    log(f"written: {path} ({app.env_key}, {algo}, iteration {best['iteration'] if best else 0} of "
        f"{len(app.history)}, mean {best['mean_return'] if best else float('nan'):.1f})")


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", nargs="?", default="CartPole")
    parser.add_argument("--algo", default="cem", choices=sorted(ALGOS))
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="an env setting")
    parser.add_argument("--save", metavar="FILE", help="write the run as a page checkpoint (.safetensors)")
    parser.add_argument("--minutes", type=float, help="(with --save) stop after this long, too")
    parser.add_argument("--hp", action="append", default=[], metavar="NAME=VALUE",
                        help="(with --save) a parameter of the algorithm, e.g. min_std=0.1")
    args = parser.parse_args()
    # (the values as the env's choices: "2" -> 2)
    from webgym.core import ENVS
    import webgym.envs  # noqa: F401

    settings = {}
    for item in args.set:
        name, value = item.split("=", 1)
        settings[name] = next(c for c in ENVS[args.env].settings[name][1] if str(c) == value)
    if args.save:
        hp = dict(item.split("=", 1) for item in args.hp)  # (App.command converts: "0.1" -> 0.1)
        train_and_save(args.save, args.env, args.algo, args.population, args.iterations, args.seed, settings,
                       minutes=args.minutes, hp=hp)
    else:
        train(args.env, args.algo, args.population, args.iterations, args.seed, settings=settings)


if __name__ == "__main__":
    main()
