"""Which StableHLO programs does the IREE compiler in the browser compile for WebGPU?
Small programs up to the PPO update are lowered by the browser's own jax (in the
training worker), then compiled one by one in the page's compile worker (a fresh one
after a crash).

    pixi run -e web-test python web/compile_test.py [--save build/mlir]
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from test_page import DIST, READY, serve

# runs in the browser (training worker, jax loaded): {name: StableHLO text} as JSON
VARIANTS = """
import json
import jax, jax.numpy as jnp, numpy as np
from webgym import gpu
from webgym.algos import ppo_jax

out = {}
x = np.ones((64, 4), np.float32)
w = np.ones((4, 8), np.float32)
out["add"] = gpu.stablehlo(lambda a: a + 1, x)
out["matmul"] = gpu.stablehlo(lambda a, b: a @ b, x, w)
out["tanh"] = gpu.stablehlo(lambda a: jnp.tanh(a), x)
out["sum"] = gpu.stablehlo(lambda a: a.sum(), x)
out["mean_sq"] = gpu.stablehlo(lambda a: jnp.mean(a ** 2), x)
out["grad_matmul"] = gpu.stablehlo(jax.grad(lambda b, a: jnp.sum(jnp.tanh(a @ b))), w, x)
out["exp_log"] = gpu.stablehlo(lambda a: jnp.log(jnp.exp(a) + 1), x)
out["slice_concat"] = gpu.stablehlo(lambda a: jnp.concatenate([a[:, :2], a[:, 2:] * 2], 1), x)
out["clip_min"] = gpu.stablehlo(lambda a, b: jnp.minimum(a * 2, jnp.clip(a, 0.8, 1.2) * b), x, x)
# (bisecting the MLP forward: what of it crashes the compiler?)
flat = np.ones(99, np.float32)
h = np.ones((64, 8), np.float32)
out["matvec"] = gpu.stablehlo(lambda a, b: a @ b, h, np.ones((8, 1), np.float32))
out["matvec_n2"] = gpu.stablehlo(lambda a, b: a @ b, h, np.ones((8, 2), np.float32))
out["slice_reshape"] = gpu.stablehlo(lambda f: f[0:32].reshape(4, 8) * 2, flat)
out["slice_reshape_matmul"] = gpu.stablehlo(lambda f, a: a @ f[0:32].reshape(4, 8), flat, x)
out["matmul_tanh_matmul"] = gpu.stablehlo(lambda a, b, c: jnp.tanh(a @ b) @ c, x, w, np.ones((8, 2), np.float32))
out["matmul_bias"] = gpu.stablehlo(lambda a, b, c: a @ b + c, x, w, np.ones(8, np.float32))
out["slice_bias"] = gpu.stablehlo(lambda a, f: a + f[32:40], h, flat)
for hidden, batch in ((8, 64), (32, 2048)):
    actor, critic = [4, hidden, 1], [4, hidden, 1]
    n = ppo_jax.count(actor) + ppo_jax.count(critic) + 1
    theta = np.zeros(n, np.float32)
    obs, act, lp, adv, ret, mask = ppo_jax.batch_arrays(batch, 4, 1)
    na = ppo_jax.count(actor)

    def forward(theta, obs):
        return ppo_jax.forward(jnp, theta[:na], actor, obs, tanh_out=True)

    def loss(theta, obs, ret):
        v = ppo_jax.forward(jnp, theta[:na], actor, obs, tanh_out=False)[:, 0]
        return jnp.mean((v - ret) ** 2)

    update = ppo_jax.make_update(actor, critic)
    out[f"forward_h{hidden}_b{batch}"] = gpu.stablehlo(forward, theta, obs)
    out[f"loss_h{hidden}_b{batch}"] = gpu.stablehlo(loss, theta, obs, ret)
    out[f"grad_h{hidden}_b{batch}"] = gpu.stablehlo(jax.grad(loss), theta, obs, ret)
    out[f"update_h{hidden}_b{batch}"] = gpu.stablehlo(
        update, theta, theta, theta, np.float32(0), obs, act, lp, adv, ret, mask, np.float32(0), np.float32(0))
json.dumps(out)
"""

# runs in the browser (training worker, jax + IREE runtime loaded): one PPO variant
# compiled, loaded onto the WebGPU device and run once -> its timings
LOAD = """
import time
import jax, jax.numpy as jnp, numpy as np
from webgym import gpu
from webgym.algos import ppo_jax

actor, critic = [4, 8, 1], [4, 8, 1]
na, n = ppo_jax.count(actor), ppo_jax.count(actor) + ppo_jax.count(critic) + 1
theta = np.zeros(n, np.float32)
obs, act, lp, adv, ret, mask = ppo_jax.batch_arrays(globals().get("BATCH", 64), 4, 1)
fns = {
    "forward": (lambda theta, obs: ppo_jax.forward(jnp, theta[:na], actor, obs, True), (theta, obs)),
    "grad": (jax.grad(lambda theta, obs, ret: jnp.mean(
        (ppo_jax.forward(jnp, theta[:na], actor, obs, False)[:, 0] - ret) ** 2)), (theta, obs, ret)),
    "update": (ppo_jax.make_update(actor, critic),
               (theta, theta, theta, np.float32(0), obs, act, lp, adv, ret, mask, np.float32(0), np.float32(0))),
}
# (tiny cases: output size, slice at the end, constants)
fns["tiny_out"] = (lambda a: a * 2.0, (np.ones(2, np.float32),))
fns["const_add"] = (lambda a: a * 2.0 + 1.2345, (theta,))
fns["tail_slice64"] = (lambda a: a[-64:] * 2.0, (theta,))
fns["head_slice2"] = (lambda a: a[:2] * 2.0, (theta,))
fns["tail_slice2"] = (lambda a: a[-2:] * 2.0, (theta,))
# (0-d scalars in / out)
fns["scalar_out"] = (lambda a: jnp.sum(a), (obs,))
fns["scalar_in"] = (lambda a, s: a * s, (obs, np.float32(0.5)))
# (the loss terms one by one)
mu = lambda th, o: ppo_jax.forward(jnp, th[:na], actor, o, True)  # noqa: E731
fns["entropy"] = (lambda th: jnp.reshape(th[-2:] * 2.0 + 0.5 * jnp.log(2 * jnp.pi * jnp.e), (2,)), (theta,))
fns["n_div"] = (lambda r, m: r / jnp.maximum(jnp.sum(m), 1.0), (ret, mask))
fns["logp"] = (lambda th, o, a: jnp.sum(-0.5 * ((a - mu(th, o)) / jnp.exp(th[-2:-1] * 1.0)) ** 2, axis=-1),
               (theta, obs, act))
fns["ratio_clip"] = (lambda l, l0, ad: jnp.minimum(jnp.exp(l - l0) * ad, jnp.clip(jnp.exp(l - l0), 0.8, 1.2) * ad),
                     (lp, lp, adv))
# (the steps from the loss to the update)
loss_fn = ppo_jax.make_update(actor, critic).loss_fn
batch = (theta, obs, act, lp, adv, ret, mask, np.float32(0.2))
fns["loss"] = (lambda *a: loss_fn(*a)[0], batch)
fns["loss_grad"] = (jax.grad(lambda *a: loss_fn(*a)[0]), batch)
fns["value_and_grad"] = (lambda *a: (lambda r: (r[1], jnp.stack(r[0][1])))(
    jax.value_and_grad(loss_fn, has_aux=True)(*a)), batch)
fn, examples = fns[NAME]
t = time.perf_counter()
f = await gpu.GpuFunction.create(fn, *examples)
loaded = time.perf_counter() - t
args = []
for k, a in enumerate(examples):
    gpu.log(f"upload {k}: shape {np.shape(a)} {np.asarray(a).dtype}")
    up = f.upload(a)
    await up
    gpu.log(f"upload {k}: done")
    args.append(up)
gpu.log("submit")
result = f.submit(*[a.value for a in args])
out = await result
gpu.log(f"submit: done ({type(out).__name__})")
first = out[0] if isinstance(out, (list, tuple)) else out
value = await f.download(first)
gpu.log("download: done")
f"loaded in {loaded:.1f} s, ran: shape {np.shape(value)}"
"""


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, help="also write the StableHLO files here")
    parser.add_argument("--only", default="", help="only variants whose name contains this")
    parser.add_argument("--files", type=Path, nargs="*", help="compile these .mlir files instead (no jax)")
    parser.add_argument("--load", nargs="*", help="load + run these on the GPU: forward grad update")
    args = parser.parse_args()
    url = serve(DIST) + "?workers=1&train=0"
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--enable-unsafe-webgpu",
                  "--enable-features=Vulkan"]
        )
        page = browser.new_page()
        page.on("console", lambda m: m.type == "error" and print("console:", m.text[:300]))
        with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
            page.goto(url)
        page.on("console", lambda m: "webgym.gpu" in m.text and print(" ", m.text, flush=True))
        if args.load:
            # (the IREE runtime loads in the background: wait for its check)
            page.wait_for_function("document.getElementById('package-webgpu')", timeout=900_000)
            print("webgpu:", page.text_content("#package-webgpu"))
            for name in args.load:
                code = f"NAME = {name!r}\n" + LOAD
                # (a hang: reported after 60 s; the worker stays stuck, so stop there)
                result = page.evaluate(
                    "(code) => Promise.race([webgymExec(code),"
                    " new Promise((r) => setTimeout(() => r('TIMEOUT (60 s)'), 60000))])",
                    code,
                )
                print(name, result, flush=True)
                if result.startswith("TIMEOUT"):
                    break
            browser.close()
            return
        if args.files:
            variants = {path.stem: path.read_text() for path in args.files}
        else:
            # jax loads in the background by itself
            page.wait_for_function("document.getElementById('package-jax')", timeout=900_000)
            variants = json.loads(page.evaluate("(code) => webgymExec(code)", VARIANTS))
            print("jax:", page.text_content("#package-jax"))
        for name, mlir in variants.items():
            if args.only not in name:
                continue
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                (args.save / f"{name}.mlir").write_text(mlir)
            result = page.evaluate("(mlir) => webgymCompileTest(mlir)", mlir)
            print(f"{name:22s} {len(mlir):7d} chars: {result}", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
