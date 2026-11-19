"""The pretrained runs that ship with the page (web/public/pretrained/): for every env, one
run of each algorithm it suggests (Env.suggested, any time budget), trained natively with
the suggested parameters (webgym.cli --save: the checkpoint of the best iteration).

    pixi run -e ml python web/pretrain.py              # all (needs pytorch: the ml env)
    pixi run -e ml python web/pretrain.py Walker cem   # one env (and algorithm)
    pixi run -e ml python web/pretrain.py Walker,Crawler   # some envs

Several runs at once (--jobs, one thread each); logs in build/pretrain/. Then build the
page (build_page.py lists them in public/pretrained/index.json).

They are not in git: they are the assets of the GitHub release "pretrained-<version>"
(pyproject.toml's version), which the Pages workflow downloads before it builds:

    python web/pretrain.py --upload      # the files of public/pretrained/ -> the release (gh)
    python web/pretrain.py --download    # the release -> public/pretrained/ (gh; also for local work)
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "pretrained"
LOGS = HERE.parent / "build" / "pretrain"
# per run: at most this many iterations / minutes (numpy algorithms are fast per
# iteration, the pytorch ones slow)
LIMITS = {"numpy": (5000, 6), "pytorch": (1000, 12)}  # (the minutes decide: Walker / CEM did 400 in 1.2 min)


def runs(only_env=None, only_algo=None):
    """(env, algo) of every suggestion"""
    import webgym.envs  # noqa: F401
    from webgym.core import ENVS

    for name, cls in sorted(ENVS.items()):
        algos = []
        for budget in cls.suggested.values():
            algos += [a for a in budget if a not in algos]
        for algo in algos:
            if (only_env is None or name in only_env.split(",")) and (only_algo in (None, algo)):
                yield name, algo


def train(env, algo):
    """one run -> (env, algo, seconds, the last log line)"""
    from webgym.algos import ALGOS

    iterations, minutes = LIMITS["pytorch" if ALGOS[algo].requires else "numpy"]
    log = LOGS / f"{env}-{algo}.log"
    command = [sys.executable, "-u", "-m", "webgym.cli", env, "--algo", algo, "--iterations", str(iterations),
               "--minutes", str(minutes), "--save", str(OUT / f"{env}-{algo}.safetensors")]
    start = time.perf_counter()
    with open(log, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"})
    lines = log.read_text().strip().splitlines()
    return env, algo, time.perf_counter() - start, lines[-1] if lines else "(no output)"


def release():
    """the GitHub release that holds the pretrained runs of this webgym release"""
    import tomllib

    return f"pretrained-{tomllib.loads((HERE.parent / 'pyproject.toml').read_text())['project']['version']}"


def upload():
    """public/pretrained/*.safetensors -> the release's assets (made if missing; replaced)"""
    tag, files = release(), sorted(str(f) for f in OUT.glob("*.safetensors"))
    if subprocess.run(["gh", "release", "view", tag], capture_output=True).returncode != 0:
        notes = ("The pretrained runs of the webgym page (one per environment and suggested algorithm), "
                 "made with web/pretrain.py. The Pages workflow downloads them.")
        subprocess.run(["gh", "release", "create", tag, "--title", f"Pretrained runs ({tag})", "--notes", notes],
                       check=True)
    subprocess.run(["gh", "release", "upload", tag, "--clobber", *files], check=True)
    print(f"{len(files)} files -> release {tag}")


def download():
    """the release's assets -> public/pretrained/"""
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", release(), "--dir", str(OUT), "--pattern", "*.safetensors",
                    "--clobber"], check=True)
    print(f"{len(list(OUT.glob('*.safetensors')))} pretrained runs in {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", nargs="?")
    parser.add_argument("algo", nargs="?")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--upload", action="store_true", help="upload public/pretrained/ to the release")
    parser.add_argument("--download", action="store_true", help="download the release into public/pretrained/")
    args = parser.parse_args()
    if args.upload or args.download:
        upload() if args.upload else download()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    todo = list(runs(args.env, args.algo))
    print(f"{len(todo)} runs, {args.jobs} at once", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for env, algo, seconds, last in pool.map(lambda r: train(*r), todo):
            print(f"{env:15s} {algo:16s} {seconds / 60:5.1f} min  {last}", flush=True)


if __name__ == "__main__":
    main()
