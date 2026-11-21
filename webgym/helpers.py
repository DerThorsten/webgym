"""Simulation helpers: more workers that run episodes for the trainer (CEM / ES).

The trainer's population is split: its own env runs the first candidates, the rest go
in chunks to the helpers (each its own python + box2d, another CPU core). A helper gets
the env (name + settings), the policy layout, the candidates' weights and a seed, runs
one episode per candidate and sends the returns back. Messages go through the page
(host_web.py / main.js); natively a test can connect both ends directly.
"""

import base64
import json
import time

import numpy as np

from webgym.core import make
from webgym.policies import from_sizes

TIMEOUT = 60.0  # s: a helper that doesn't answer by then: its candidates count as failed


def encode(array):
    """float64 array -> text (JSON-able)"""
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.float64).tobytes()).decode()


def decode(text, shape):
    return np.frombuffer(base64.b64decode(text), dtype=np.float64).reshape(shape)


class Pool:
    """the trainer's side: `count` helpers, reached with `post(message)`; their answers
    come back through `receive(message)`"""

    def __init__(self, post):
        self.post = post
        self.count = 0  # helpers ready
        self.spec = {}  # what they simulate: {"env", "settings"} (set by the app)
        self.next_id = 0
        self.pending = {}  # request id -> Request
        self.steps = 0  # agent-steps the helpers simulated (the steps/s in the top bar)

    def submit(self, policy, params, seed):
        """run `params` (k, num_params) on the helpers (in `count` chunks) -> a Request"""
        request = Request(len(params))
        self.next_id += 1
        self.pending[self.next_id] = request
        for part, rows in enumerate(np.array_split(np.arange(len(params)), self.count)):
            if not len(rows):
                continue
            request.parts[part] = rows
            self.post({
                "type": "evaluate", "id": self.next_id, "part": part, **self.spec,
                "kind": policy.kind, "sizes": list(policy.sizes), "tanh_out": policy.tanh_out,
                "seed": int(seed) + 1 + part,
                "n": len(rows), "params": encode(params[rows]),
            })
        return request

    def receive(self, message):
        """a helper's answer {"id", "part", "returns", "steps"}"""
        self.steps += message.get("steps", 0)
        request = self.pending.get(message["id"])
        if request is not None:
            request.done(message["part"], np.asarray(message["returns"], dtype=float))
            if request.finished:
                del self.pending[message["id"]]


class Request:
    """the returns of a submitted chunk of candidates, as they come in"""

    def __init__(self, n):
        self.returns = np.full(n, np.nan)
        self.parts = {}  # part -> the rows it covers
        self.started = time.monotonic()

    def done(self, part, returns):
        self.returns[self.parts.pop(part)] = returns

    @property
    def finished(self):
        return not self.parts

    @property
    def timed_out(self):
        return time.monotonic() - self.started > TIMEOUT


_envs = {}  # the helper's envs (kept: building one takes a while), by (env, settings, n)


def evaluate(message):
    """the helper's side: one episode per candidate of `message` (see Pool.submit) ->
    the answer {"type": "evaluated", "id", "part", "returns", "steps" (agent-steps
    simulated)}"""
    from webgym.algos.cem import rollout

    key = (message["env"], json.dumps(message["settings"], sort_keys=True), message["n"])
    if key not in _envs:
        for old in [k for k in _envs if k[:2] == key[:2]]:  # (another size: the old one goes)
            _envs.pop(old).close()
        _envs[key] = make(message["env"], n=message["n"], **message["settings"])
    policy = from_sizes(message.get("kind", "mlp"), message["sizes"], message["tanh_out"])
    params = decode(message["params"], (message["n"], policy.num_params))
    env = _envs[key]
    episode = rollout(env, policy, params, message["seed"])
    steps = 0
    try:
        while True:
            next(episode)
            steps += int(env.n - env.done.sum())  # (the agents still going)
    except StopIteration as end:
        returns = end.value
    return {"type": "evaluated", "id": message["id"], "part": message["part"], "returns": returns.tolist(),
            "steps": steps}
