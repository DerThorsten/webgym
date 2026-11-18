# Bugs found in the browser stack (jax / IREE / WebGPU / pytorch on emscripten-forge)

> **Note (2026-11-18):** webgym has since dropped jax and the GPU path; it now uses pytorch only, on pyjs 5. The jax/GPU code and the reproducer scripts named below are in commit `bb4c07a`.

This file covers bugs found while building webgym. The environment:
- pyjs npm 4.4.1 (Python 3.14, wasm32), with packages from emscripten-forge.
- Chrome (headless chromium with swiftshader, and desktop).
- Date: 2026-11-17.

Reproducers: `web/compile_test.py` (StableHLO → IREE WebGPU in the page's compile worker).

---

## 1. IREE compiler (wasm32): a 1-element `stablehlo.slice` crashes the compiler

**Packages:** `iree-base-compiler` from emscripten-forge (wasm32), target `--iree-hal-target-device=webgpu --iree-execution-model=async-external`.

**Symptom:** compiling aborts the whole runtime with `RuntimeError: memory access out of bounds`. This is a wasm trap; Python is dead afterwards. It breaks any MLP with a single output (the bias slice) and any policy with one action (log_std).

**Minimal reproducer** (259 chars):
```mlir
module @m {
  func.func public @main(%arg0: tensor<99xf32>, %y: tensor<1xf32>) -> tensor<1xf32> {
    %12 = stablehlo.slice %arg0 [48:49] : (tensor<99xf32>) -> tensor<1xf32>
    %15 = stablehlo.add %y, %12 : tensor<1xf32>
    return %15 : tensor<1xf32>
  }
}
```

**Bisection results:**

| variant | browser (wasm32, webgpu) |
|---|---|
| 1-element slice (`[48:49]`, also `[0:1]`, also of a `tensor<4xf32>`), added or broadcast | **crash** |
| 2-element slice, broadcast to 64x2 | ok |
| 8-element slice, broadcast to 64x8 | ok |
| a `tensor<1xf32>` *argument* broadcast to 64x1 (no slice) | ok |
| 2-element slice → dot with constant `[[0],[1]]` → broadcast | ok (the workaround) |
| 2-element slice × `[0, 1]` → reduce → broadcast | ok |
| elementwise ops, matmul, tanh, reductions, grad of a matmul, slice+concat | ok |

**Native compiler:** the same files compile with the PyPI iree-base-compiler 3.14.0 for `vulkan` (SPIR-V) and `llvm-cpu`. The PyPI build has no `webgpu` target ("unregistered device alias webgpu"). So this report cannot tell a wasm-only memory bug from a bug in the WebGPU (SPIR-V → WGSL) path. The next step would be a native build with the WebGPU target, or an ASAN/debug wasm build.

**Workaround in webgym:** `webgym/algos/ppo_jax.py`, `part()`, never slices a single element: it slices 2 elements and picks one with a matmul against `[[0], [1]]`. With it, the whole PPO update compiles in the browser (80–95 KB modules) and the native IREE-vs-jax equivalence test still passes.

## 1b. IREE WebGPU (compiler + runtime): five more problems on the way to running PPO

Found with `web/compile_test.py --load <variant>`, which compiles, loads and runs on the GPU and logs every step. The browser was headless chromium with swiftshader; one problem also showed in desktop Chrome. With all four workarounds, the full PPO update (74 KB module) runs on WebGPU.

1. **One buffer bound read-only and read-write in the same dispatch.** IREE packs intermediate results into one buffer and binds slices of it with different access. WebGPU rejects this:
   ```
   [Buffer] usage (Storage(read-write)|Storage(read-only)) includes writable usage and
   another usage in the same synchronization scope. - While validating compute pass usage.
   ```
   - **What follows:** `RESOURCE_EXHAUSTED; failed to create staging buffer for queue_write`, and then the download waits forever, so the page hangs silently.
   - **Workaround:** the global compiler option `--iree-stream-resource-alias-mutable-bindings=true`. It is not a session flag (`Session.set_flags` says "option not found"); it has to be set with `ctypes_dl._initializeGlobalCL(...)`.
   - **Fix in IREE:** the WebGPU target should enable this by default, since WebGPU forbids mixed access to one buffer.
2. **Constant expressions run when the module loads, and hang.** `jnp.log(2 * jnp.pi * jnp.e)` is lowered to a `log` op on a constant. With `--iree-opt-const-eval=false` it becomes a module initializer. That initializer runs on the GPU inside `add_vm_module` and needs a blocking wait, so it hangs forever without an error.
   - **Reproducer:** `lambda th: th[-2:] * 2 + 0.5 * jnp.log(2 * jnp.pi * jnp.e)`, an 8 KB module.
   - **Workaround:** compute the constants in Python (`math.log`).
   - **Fix in IREE:** initializers on WebGPU need an async path, or at least an error instead of a hang.
3. **0-d inputs (our bug, not IREE's).** We got `input7 shape rank mismatch; expected 0 (scalar) but have 1`. The cause was our `gpu.upload`: it used `np.ascontiguousarray`, which always returns at least 1-d. `enqueue_upload` itself keeps 0-d arrays 0-d (checked with `web/repro_test.py`). Fixed with `np.asarray(a, order="C")`.
4. **Binding offsets not aligned to 256.** With (1), merged bindings use offsets like 1664, but WebGPU requires `minStorageBufferOffsetAlignment` = 256 (`Offset (1664) ... does not satisfy the minimum ReadOnlyStorage alignment (256)`). After that, every `createBuffer` fails.
   - **Workaround:** the global option `--iree-stream-resource-min-offset-alignment=256`.
   - With 1–4 fixed, PPO trains on WebGPU in the page (`test_page.py --gpu-train` passes).
   - Full write-up for upstream: REPORT_UPSTREAM_WEBGPU.md.
5. **Many queued submissions run out of GPU memory (desktop Chrome, real GPU).** This was possibly just a consequence of 4. All minibatches of an iteration were queued at once, each with its own uploads and about 1 MB of temporaries:
   ```
   RESOURCE_EXHAUSTED; WebGPU device.createBuffer failed for 964736 bytes
   ```
   - **Workaround:** await each minibatch before queueing the next. The weights stay on the GPU.
   - **Open question:** whether IREE frees transients late, or the queue was simply too deep.

---

## 2. jaxlib + iree-base-compiler cannot be imported into one process (wasm)

**Packages:** `jaxlib` and `iree-base-compiler` (emscripten-forge, wasm32).

**Symptom:** importing both aborts Python:
```
Critical nanobind error: refusing to add duplicate key "NotSpeculatable" to enumeration "Speculatability"
Aborted()  → wasm `unreachable`
```

**Cause:** both ship their own MLIR Python bindings, built with nanobind. In the emscripten builds they register into the same nanobind domain. Natively (PyPI) they coexist.

**Fix (recipes):** build one of them with its own `NB_DOMAIN`, e.g. `-DNB_DOMAIN=iree` for the IREE compiler's bindings.

**Workaround in webgym:** the IREE compiler runs in a separate worker (the "compile worker"). It costs one more Python runtime: start-up time, memory, and its own 29 MB download, which is cached after the first visit.

---

## 3. IREE runtime (WebGPU HAL) in the browser: no blocking waits

**Package:** `iree-base-runtime` (emscripten-forge, wasm32), WebGPU driver.

**Symptom:** the usual synchronous upload of input buffers fails with `blocking waits are unavailable`. Other blocking calls presumably fail the same way; this has not been checked one by one.

**Cause:** this is expected in a browser, which has no blocking on the main loop or in JS-driven workers. But it is not documented, and it is the first thing that fails.

**What works:** the async API: `AsyncSession`, `enqueue_upload`, fence-chained submissions with `--iree-execution-model=async-external`, and async downloads. webgym wraps this in `webgym/gpu.py` (`GpuFunction`).

**Suggestion:** document it, and have the sync API raise a clear "use the async API in the browser" error. A small example in the package's tests would help.

---

## 4. Version skew: browser jax ≠ native jax

**Facts:** the browser has jax 0.11.2.dev20260824 (emscripten-forge); conda-forge native has 0.12.

**Impact:** StableHLO generated natively and compiled in the browser is not a valid test. Reproducers must be generated by the browser's own jax, which `web/compile_test.py` now does. This is not a bug, but it cost a wrong turn. Publishing jax and jaxlib at matching versions would help.

---

## 5. Noise rather than bugs

- **jax filelock warning:** on import, jax prints `UserWarning: only soft file lock is available … import filelock` to stderr, which shows up as a `console.error`. It is harmless, but test harnesses have to filter it.
- **jax import time:** about 20 s ("loaded in 20.3 s") for download, unpacking and import, and still slow on a second visit. See REPORT_PYJS_CACHING.md.

## pytorch

There are no hard bugs so far. PPO (torch) and DQN (torch) train correctly in the browser. webgym calls `torch.autograd.set_multithreading_enabled(False)` as a precaution, since the worker has no threads. Whether this is actually needed has not been re-checked.
