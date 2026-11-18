"""Run the reproducers of REPORT_UPSTREAM_WEBGPU.md in the browser, each in a fresh page
with its workaround switched off; prints whether each bug still shows up.

    pixi run -e web-test web-build
    pixi run -e web-test python web/repro_test.py --arch wasm64
"""

import argparse

from playwright.sync_api import sync_playwright

from compile_test import LOAD
from test_page import DIST, READY, serve

TIMEOUT_S = 90

# 1: the compiler on a 1-element slice
SLICE1 = """module @m {
  func.func public @main(%arg0: tensor<99xf32>, %y: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.slice %arg0 [48:49] : (tensor<99xf32>) -> tensor<1xf32>
    %1 = stablehlo.add %y, %0 : tensor<1xf32>
    return %1 : tensor<1xf32>
  }
}"""

# compile worker: the WebGPU workaround options off (all), or only the alignment off
NO_FLAGS = "from webgym import gpu\ngpu._global_flags_set = True\n'workaround options off'"
ALIAS_ONLY = """from iree.compiler.api import ctypes_dl
from webgym import gpu
ctypes_dl._initializeGlobalCL("iree", "--iree-stream-resource-alias-mutable-bindings=true")
gpu._global_flags_set = True
'alignment option off'"""

# 2b: the small MLP of the report (compile, load, run)
REPORT_MLP = """
import jax, jax.numpy as jnp, numpy as np
from webgym import gpu

def fn(theta, obs):
    w1 = theta[0:32].reshape(4, 8); b1 = theta[32:40]
    w2 = theta[40:48].reshape(8, 1); b2 = theta[47:49] @ jnp.array([[0.0], [1.0]])
    return jnp.tanh(jnp.tanh(obs @ w1 + b1) @ w2 + b2)

examples = (np.zeros(99, np.float32), np.zeros((64, 4), np.float32))
f = await gpu.GpuFunction.create(fn, *examples)
ups = [f.upload(a) for a in examples]
out = await f.submit(*[u.value for u in ups], wait_fences=[u.fence for u in ups])
v = await f.download(out[0] if isinstance(out, (list, tuple)) else out)
f"ran, shape {np.shape(v)}"
"""

# 3: two chained PPO updates as in the page (the 1st call's outputs, slices of one
# packed result buffer, are the 2nd call's inputs)
CHAINED_UPDATES = """
import numpy as np
from webgym import gpu
from webgym.algos import ppo_jax

actor, critic = [4, 32, 1], [4, 32, 1]
theta = np.zeros(ppo_jax.count(actor) + ppo_jax.count(critic) + 1, np.float32)
f32 = np.float32
args = (theta, theta, theta, f32(0), *ppo_jax.batch_arrays(2048, 4, 1), f32(3e-3), f32(0.2))
f = await gpu.GpuFunction.create(ppo_jax.make_update(actor, critic), *args)
ups = [f.upload(a) for a in args]
first = f.submit(*[u.value for u in ups], wait_fences=[u.fence for u in ups])
th, m, v, t, losses = first.value
out = await f.submit(th, m, v, t, *[u.value for u in ups[4:]], wait_fences=[first.fence])
result = await f.download(out[0])
f"2 chained updates ran, theta {np.shape(result)}"
"""

# (name, needs jax + IREE runtime, compile worker setup, how to run, bug shown if ...)
REPROS = [
    ("1 compiler: 1-element slice", False, None, ("compile", SLICE1),
     lambda r, log: "memory access out of bounds" in r),
    ("2 read-only + read-write binding (PPO forward)", True, NO_FLAGS, ("exec", "NAME = 'forward'\n" + LOAD),
     lambda r, log: "includes writable usage" in log),
    ("2b same, the report's small MLP", True, NO_FLAGS, ("exec", REPORT_MLP),
     lambda r, log: "includes writable usage" in log),
    # (control: with all workarounds the chained updates run)
    ("3a control: 2 chained PPO updates, workarounds on", True, None, ("exec", CHAINED_UPDATES),
     lambda r, log: not r.startswith("2 chained updates ran")),
    ("3 offsets not aligned to 256 (2 chained PPO updates)", True, ALIAS_ONLY, ("exec", CHAINED_UPDATES),
     lambda r, log: "does not satisfy the minimum" in log),
    ("4 jnp.log(constant): load hangs", True, None, ("exec", "NAME = 'entropy'\n" + LOAD),
     lambda r, log: r.startswith("TIMEOUT")),
    ("5 jaxlib + iree-base-compiler in one python", False, None, ("install", "import jax\nimport iree.compiler.api\n'both imported'"),
     lambda r, log: "nanobind" in r or r.startswith("fatal")),
]


def run(page, how, code):
    """run a reproducer in the page -> its result (or TIMEOUT)"""
    race = f"new Promise((r) => setTimeout(() => r('TIMEOUT ({TIMEOUT_S} s)'), {TIMEOUT_S * 1000}))"
    call = {
        "compile": "webgymCompileTest(code)",
        "exec": "webgymExec(code)",
        "install": "webgymCompilerInstall(['jax'], code)",
    }[how]
    return str(page.evaluate(f"(code) => Promise.race([{call}, {race}])", code))


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", default="wasm32", choices=["wasm32", "wasm64"])
    parser.add_argument("--only", default="", help="only reproducers whose name starts with this")
    args = parser.parse_args()
    url = serve(DIST) + f"?workers=1&train=0&arch={args.arch}"
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--enable-unsafe-webgpu",
                  "--enable-features=Vulkan"]
        )
        context = browser.new_context()  # (one context: the package cache is shared)
        for name, packages, setup, (how, code), shown in REPROS:
            if not name.startswith(args.only):
                continue
            page = context.new_page()
            log = []
            page.on("console", lambda m: log.append(m.text))
            with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
                page.goto(url)
            if packages:  # (jax + the IREE runtime load in the background)
                page.wait_for_function("document.getElementById('package-webgpu')", timeout=900_000)
            if setup:
                # (compiler options are global: a fresh compile worker, then the setup,
                # before anything is compiled in it)
                page.evaluate("webgymCompilerReset()")
                print(" ", page.evaluate("(code) => webgymCompilerExec(code)", setup))
            result = run(page, how, code)
            reproduced = shown(result, "\n".join(log))
            rows.append((name, reproduced, result.splitlines()[0][:110] if result else ""))
            print(f"{name}: {'REPRODUCED' if reproduced else 'not reproduced'} | {rows[-1][2]}", flush=True)
            page.close()
        browser.close()
    print(f"\n{args.arch}:")
    for name, reproduced, result in rows:
        print(f"  {'yes' if reproduced else 'no ':3s}  {name}")


if __name__ == "__main__":
    main()
