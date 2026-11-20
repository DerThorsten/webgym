"""Do the pretrained runs still fit their envs? (after changing an env)

    pixi run -e ml python web/check_pretrained.py

Each run of public/pretrained/ is restored as the page does it (refused: the policy's
sizes no longer fit), then its policy plays 16 episodes of the current env. "DROP?": it
scores clearly below its mean in training: the env changed since, retrain it
(web/pretrain.py <env> <algo>). (Noisy envs, like the Car, score lower on new tracks
anyway.)
"""

import time
from pathlib import Path

import numpy as np

from webgym import checkpoints
from webgym.app import App
from webgym.core import make

FOLDER = Path(__file__).resolve().parent / "public" / "pretrained"


def check(path, n=16):
    meta = checkpoints.loads(path.read_bytes())[1]  # (its env and algorithm; restore sets the settings)
    app = App(env=meta["env"], algo=meta["algo"], draws=False)
    try:
        app.restore(path.read_bytes())
    except Exception as e:  # noqa: BLE001
        return f"REFUSED: {e}"
    trained = app.history[-1]["mean_return"]
    env = make(app.env_name, n=n, **app.settings)
    obs = env.reset(seed=123)
    params = np.stack([app.params] * n)
    for _ in range(env.max_steps + 5):
        out = app.policy(params, obs)
        action = env.discrete_actions[np.argmax(out, axis=1)] if not app.policy.tanh_out else out
        obs = env.step(action)[0]
        if env.done.all():
            break
    now = float(np.mean(env.episode_return))
    env.close()
    flag = "OK" if now >= trained - max(20, 0.25 * abs(trained)) else "DROP?"
    return f"key {app.env_key!r:28s} trained mean {trained:8.1f}  now {now:8.1f}  {flag}"


def main():
    for path in sorted(FOLDER.glob("*.safetensors")):
        start = time.perf_counter()
        print(f"{path.name:40s} {check(path)}  ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
