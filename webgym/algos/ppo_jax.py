"""PPO with jax, on the CPU (jax.jit) or on the GPU (WebGPU in the browser, via IREE).

The whole minibatch update (policy + value loss, gradient, gradient clipping, Adam) is
one pure function on flat vectors (actor | critic | log_std, the Adam state), so it can
be jitted by jax or compiled by IREE for WebGPU. On the GPU the weights and the Adam
state stay there: all minibatch updates of an iteration are queued (chained by fences),
only the new weights come back. The episodes run on the CPU (box2d).

The actor uses the layout of webgym.policies.MLP (watching needs no jax).
"""

import asyncio
import math
import time

import numpy as np

from webgym import gpu


# constants as python floats: jnp.log(constant) would be computed by the compiled module
# when it is loaded, which needs a blocking wait (hangs in the browser, REPORT_BUGS.md)
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def layer_shapes(sizes):
    return list(zip(sizes[:-1], sizes[1:]))


def count(sizes):
    return sum(a * b + b for a, b in layer_shapes(sizes))


def part(xp, flat, i, n):
    """flat[i : i + n]. A 1-element slice is taken as 2 elements and picked: IREE's
    compiler in the browser (wasm) crashes on 1-element slices (REPORT_BUGS.md)"""
    if n != 1 or i == 0:
        return flat[i : i + n]
    return flat[i - 1 : i + 1] @ xp.asarray([[0.0], [1.0]], dtype=flat.dtype)


def forward(xp, flat, sizes, x, tanh_out):
    """the MLP `flat` (MLP layout) on x (batch, in) with numpy or jax.numpy (`xp`)"""
    i = 0
    shapes = layer_shapes(sizes)
    for k, (a, b) in enumerate(shapes):
        w = flat[i : i + a * b].reshape(a, b)
        i += a * b
        x = x @ w + part(xp, flat, i, b)
        i += b
        if tanh_out or k < len(shapes) - 1:
            x = xp.tanh(x)
    return x


def make_update(actor_sizes, critic_sizes):
    """update(theta, m, v, t, obs, act, logp_old, adv, ret, mask, lr, clip) ->
    (theta, m, v, t, losses [policy, value, entropy]): one PPO minibatch step with Adam;
    theta = actor | critic | log_std; mask: 0 for padding rows"""
    import jax
    import jax.numpy as jnp

    na, nc = count(actor_sizes), count(critic_sizes)

    def loss_fn(theta, obs, act, logp_old, adv, ret, mask, clip):
        actor, critic = theta[:na], theta[na : na + nc]
        log_std = part(jnp, theta, na + nc, theta.shape[0] - na - nc)
        mu = forward(jnp, actor, actor_sizes, obs, tanh_out=True)
        z = (act - mu) / jnp.exp(log_std)
        logp = jnp.sum(-0.5 * z**2 - log_std - HALF_LOG_2PI, axis=-1)
        ratio = jnp.exp(logp - logp_old)
        n = jnp.maximum(jnp.sum(mask), 1.0)
        surrogate = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv)
        policy_loss = -jnp.sum(mask * surrogate) / n
        value = forward(jnp, critic, critic_sizes, obs, tanh_out=False)[:, 0]
        value_loss = jnp.sum(mask * (value - ret) ** 2) / n
        entropy = jnp.sum(log_std + HALF_LOG_2PI + 0.5)
        return policy_loss + 0.5 * value_loss, (policy_loss, value_loss, entropy)

    def update(theta, m, v, t, obs, act, logp_old, adv, ret, mask, lr, clip):
        (_, (pl, vl, ent)), g = jax.value_and_grad(loss_fn, has_aux=True)(
            theta, obs, act, logp_old, adv, ret, mask, clip
        )
        g = g * jnp.minimum(1.0, 0.5 / (jnp.sqrt(jnp.sum(g**2)) + 1e-8))  # clip the gradient norm
        t = t + 1.0
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        theta = theta - lr * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
        return theta, m, v, t, jnp.stack([pl, vl, ent])

    update.loss_fn = loss_fn  # (for experiments, web/compile_test.py)
    return update


def batch_arrays(n, obs_dim, act_dim):
    """example arrays of one minibatch of n rows (for jit / compiling)"""
    f = np.float32
    return (np.zeros((n, obs_dim), f), np.zeros((n, act_dim), f), np.zeros(n, f), np.zeros(n, f),
            np.zeros(n, f), np.zeros(n, f))


class GpuUpdater:
    """the update compiled for WebGPU; weights + Adam state live on the GPU between
    calls (`run` queues all minibatches of an iteration, downloads the result)"""

    @classmethod
    async def create(cls, update, theta, minibatch, obs_dim, act_dim):
        self = cls()
        f = np.float32
        examples = (theta, theta, theta, f(0), *batch_arrays(minibatch, obs_dim, act_dim), f(0), f(0))
        self.fn = await gpu.GpuFunction.create(update, *examples)
        self.adapter = await gpu.adapter_name()  # (which GPU: shown in the policy line)
        self.state = None  # (theta, m, v, t) on the GPU + the fence when they are ready
        return self

    def upload_state(self, theta, m, v, t):
        ups = [self.fn.upload(np.asarray(a, np.float32)) for a in (theta, m, v, np.float32(t))]
        self.state = [u.value for u in ups], [u.fence for u in ups]

    async def run(self, batches, lr, clip):
        """the minibatches one after the other on the GPU (the weights stay there) ->
        (theta, m, v, t, last losses) numpy. Each one is awaited before the next is
        queued: queueing all of them at once ran out of GPU memory (each holds its
        batch + ~1 MB of temporaries until it is done)"""
        (theta, m, v, t), fences = self.state
        self.runs = getattr(self, "runs", 0) + 1
        gpu.log(f"GPU update {self.runs}: {len(batches)} minibatches")
        scalars = [self.fn.upload(np.float32(lr)), self.fn.upload(np.float32(clip))]
        for batch in batches:
            ups = [self.fn.upload(a) for a in batch]
            waits = [*fences, *(u.fence for u in ups + scalars)]
            result = self.fn.submit(theta, m, v, t, *(u.value for u in ups),
                                    *(s.value for s in scalars), wait_fences=waits)
            theta, m, v, t, losses = result.value
            fences = [result.fence]
            await result
        self.state = (theta, m, v, t), fences
        out = [await self.fn.download(a) for a in (theta, m, v, t, losses)]
        return out


def train(env, policy, iterations=1000, lr=3e-3, gamma=0.99, clip=0.2, epochs=4, seed=0, state=None,
          device="cpu", lam=0.95, minibatch=2048):
    """PPO on the batched env, device "cpu" (jax.jit) or "webgpu" (falls back to the CPU
    without WebGPU). Yields None per env step / minibatch (CPU) / while the GPU works,
    stats per iteration (losses, "device"); `state` resumes (weights + Adam state)."""
    import jax

    rng = np.random.default_rng(seed)
    actor_sizes = policy.sizes
    critic_sizes = [actor_sizes[0], *actor_sizes[1:-1], 1]
    na, nc, act_dim = count(actor_sizes), count(critic_sizes), actor_sizes[-1]
    if state is None:
        def init(sizes):  # small random weights (layer by layer), zero biases
            parts = []
            for a, b in layer_shapes(sizes):
                parts += [rng.standard_normal(a * b) / np.sqrt(a), np.zeros(b)]
            return np.concatenate(parts)

        theta = np.concatenate([init(actor_sizes), init(critic_sizes), np.full(act_dim, -0.5)]).astype(np.float32)
        state = {"iteration": 0, "theta": theta, "m": np.zeros_like(theta), "v": np.zeros_like(theta), "t": np.array(0.0)}
    theta, m, v = (np.asarray(state[k], np.float32) for k in ("theta", "m", "v"))
    t, start = float(np.ravel(state["t"])[0]), int(state["iteration"])
    update = make_update(actor_sizes, critic_sizes)

    # the device: WebGPU (compiled once, in the background) or jax on the CPU
    on_gpu, gpu_updater = False, None
    if device == "webgpu":
        if gpu.IN_BROWSER:
            task = asyncio.ensure_future(_gpu_setup(update, theta, minibatch, actor_sizes[0], act_dim))
            while not task.done():
                yield "wait"
            gpu_updater, gpu_error = task.result()
            on_gpu = gpu_updater is not None
            device_info = f"GPU: {gpu_updater.adapter}" if on_gpu else f"CPU: no WebGPU ({gpu_error})"
        else:
            device_info = "CPU: no WebGPU outside the browser"
    else:
        device_info = "CPU"
    if on_gpu:
        gpu_updater.upload_state(theta, m, v, t)
    else:
        update_cpu = jax.jit(update)

    for iteration in range(start, start + iterations):
        # one episode per agent, actions sampled around the actor's output
        actor, critic, log_std = theta[:na], theta[na : na + nc], theta[na + nc :]
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        O, A, LP, R, V, M = [], [], [], [], [], []
        while not env.done.all():
            alive = ~env.done
            mean = forward(np, actor, actor_sizes, obs, tanh_out=True)
            a = (mean + np.exp(log_std) * rng.standard_normal(mean.shape)).astype(np.float32)
            z = (a - mean) / np.exp(log_std)
            lp = np.sum(-0.5 * z**2 - log_std - 0.5 * np.log(2 * np.pi), axis=-1)
            value = forward(np, critic, critic_sizes, obs, tanh_out=False)[:, 0]
            next_obs, reward, _, _ = env.step(a)
            O.append(obs), A.append(a), LP.append(lp), R.append(reward), V.append(value), M.append(alive)
            obs = next_obs
            yield None
        R, V, M = np.array(R, np.float32), np.array(V, np.float32), np.array(M)
        # advantages (GAE); an agent's episode ends where it is not alive any more
        T = len(R)
        adv = np.zeros_like(R)
        last = np.zeros(env.n, np.float32)
        for k in reversed(range(T)):
            next_alive = M[k + 1] if k + 1 < T else np.zeros(env.n, bool)
            next_v = V[k + 1] if k + 1 < T else np.zeros(env.n, np.float32)
            delta = R[k] + gamma * next_v * next_alive - V[k]
            last = delta + gamma * lam * next_alive * last
            adv[k] = last
        valid = M.reshape(-1)
        o = np.array(O, np.float32).reshape(-1, actor_sizes[0])[valid]
        a = np.array(A, np.float32).reshape(-1, act_dim)[valid]
        lp, ret = np.array(LP, np.float32).reshape(-1)[valid], (adv + V).reshape(-1)[valid]
        adv = adv.reshape(-1)[valid]
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)

        # minibatches of a fixed size (padded, mask 0): one compiled function for all
        batches = []
        for _ in range(int(epochs)):
            order = rng.permutation(len(o))
            for s in range(0, len(o), minibatch):
                idx = order[s : s + minibatch]
                pad = minibatch - len(idx)
                mask = np.concatenate([np.ones(len(idx), np.float32), np.zeros(pad, np.float32)])
                idx = np.concatenate([idx, np.zeros(pad, int)])
                batches.append((o[idx], a[idx], lp[idx], adv[idx], ret[idx], mask))
        # the update, timed (shown per minibatch: compare GPU and CPU by toggling)
        busy = 0.0
        if on_gpu:
            started = time.perf_counter()
            task = asyncio.ensure_future(gpu_updater.run(batches, lr, clip))
            while not task.done():
                yield "wait"
            theta, m, v, t, losses = task.result()
            t = float(np.ravel(t)[0])
            busy = time.perf_counter() - started
        else:
            for batch in batches:
                started = time.perf_counter()
                out = update_cpu(theta, m, v, np.float32(t), *batch, np.float32(lr), np.float32(clip))
                theta, m, v, t, losses = jax.block_until_ready(out)
                busy += time.perf_counter() - started
                yield None
            theta, m, v, t = np.asarray(theta), np.asarray(m), np.asarray(v), float(t)
        losses = np.asarray(losses)
        returns = env.episode_return
        it = iteration + 1
        yield {
            "iteration": it,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": theta[:na].astype(np.float64),
            "losses": {"policy_loss": float(losses[0]), "value_loss": float(losses[1]), "entropy": float(losses[2])},
            "device": f"{device_info}, {1e3 * busy / max(len(batches), 1):.0f} ms/minibatch",
            "state": {"iteration": it, "theta": theta, "m": m, "v": v, "t": np.array(t)},
        }


async def _gpu_setup(update, theta, minibatch, obs_dim, act_dim):
    """(GpuUpdater, None) or (None, why not) when there is no WebGPU / it fails"""
    try:
        if not await gpu.webgpu_available():
            return None, "this browser has no WebGPU adapter"
        return await GpuUpdater.create(update, theta, minibatch, obs_dim, act_dim), None
    except Exception as e:  # (e.g. compile errors): train on the CPU
        gpu.log(f"GPU setup failed: {e!r}")
        return None, repr(e)
