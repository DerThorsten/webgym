# jax → IREE → WebGPU in the browser (emscripten-forge): bugs found, with reproducers

**Context.** webgym trains a small PPO policy in the browser. The pipeline:
1. jax (`jax.jit(fn).lower(...).as_text()`) produces StableHLO;
2. `iree-base-compiler` compiles it for WebGPU, in the browser;
3. `iree-base-runtime` runs it on the WebGPU device, from a Web Worker.

**Versions:**
- pyjs npm 4.4.1 (Python 3.14), all packages from emscripten-forge (6x channel), checked on **wasm32 and wasm64**;
- jax 0.11.2.dev20260824 (jaxlib from emscripten-forge);
- iree-base-compiler and iree-base-runtime 20261115 from emscripten-forge;
- Chrome / chromium (headless with swiftshader, and desktop Chrome on a real GPU).

**Results:** the problems below have workarounds, and with them, the full PPO update compiles and runs on WebGPU. Items 1–4 are IREE; 5 is the jaxlib and IREE packaging; 6 is usability. **pytorch:** no bugs found; PPO and DQN with torch train fine in the browser.

The compile options used everywhere (the "WebGPU flags"):
```
--iree-input-type=stablehlo --iree-opt-const-eval=false
--iree-hal-target-device=webgpu --iree-execution-model=async-external
```

---

## Shared code for the reproducers

### Compile (in a worker that has iree-base-compiler; *not* the same worker as jaxlib, see 5)
```python
from iree.compiler.api import Output, Session, Source

FLAGS = ["--iree-input-type=stablehlo", "--iree-opt-const-eval=false",
         "--iree-hal-target-device=webgpu", "--iree-execution-model=async-external"]

def compile_webgpu(mlir: str) -> bytes:
    with Session() as session:
        session.set_flags(*FLAGS)
        with session.invocation() as inv, Source.wrap_buffer(session, mlir.encode()) as src:
            inv.enable_console_diagnostics()
            assert inv.parse_source(src) and inv.execute(), "compile failed"
            with Output.open_membuffer() as out:
                inv.output_vm_bytecode(out)
                return bytes(out.map_memory())
```

### Run (in a worker with iree-base-runtime, top-level `await` as in pyjs' `async_exec_eval`)
```python
import numpy as np
import iree.runtime as rt
from iree.runtime.asyncio import AsyncSession
from iree.runtime.webgpu import get_webgpu_device

async def run(binary: bytes, *arrays):
    device = await get_webgpu_device()
    config = rt.Config(device=device)
    ctx = rt.SystemContext(config=config)
    module = rt.VmModule.copy_buffer(config.vm_instance, binary)
    ctx.add_vm_module(module)                       # <- 4 hangs here
    session = AsyncSession(device)
    fn = session.bind(getattr(ctx.modules, module.name).main)
    # (np.asarray, not np.ascontiguousarray: that turns 0-d scalars into shape (1,))
    ups = [session.enqueue_upload(np.asarray(a, order="C")) for a in arrays]
    result = fn.submit(*(u.value for u in ups), wait_fences=[u.fence for u in ups])
    out = await result                              # <- 2 fails here (3: in a chained call)
    first = out[0] if isinstance(out, (list, tuple)) else out
    return await session.download(first)
```

### StableHLO from jax (in the jax worker)
```python
import jax
mlir = jax.jit(fn).lower(*example_arrays).as_text()
```

---

## Which bugs show up where

All of them are checked by `web/repro_test.py --arch wasm32|wasm64`. Each reproducer runs in a fresh page with its workaround switched off.

| bug | wasm32 | wasm64 |
|---|---|---|
| 1 compiler crash on a 1-element slice | **yes** | no |
| 2 buffer bound read-only + read-write in one dispatch | **yes** | **yes** |
| 3 binding offsets not aligned to 256 (chained calls) | **yes** | **yes** |
| 4 `jnp.log(constant)` → initializer hangs `add_vm_module` | **yes** | **yes** |
| 5 jaxlib + iree-base-compiler in one Python (nanobind) | **yes** | **yes** |
| control: 2 chained PPO updates with all workarounds | runs | runs |

Bugs 2–5 do not depend on the architecture. Bug 1 is a wasm32 build problem.

## 1. Compiler (wasm32 only): a 1-element `stablehlo.slice` → `memory access out of bounds`

**What happens:** the whole compiler process dies with a wasm trap, `RuntimeError: memory access out of bounds`. Any MLP with one output (its bias is a 1-element slice of the flat parameters) and any 1-D action policy (`log_std`) triggers it.

**Reproducer** (compile only, the WebGPU flags):
```mlir
module @m {
  func.func public @main(%arg0: tensor<99xf32>, %y: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.slice %arg0 [48:49] : (tensor<99xf32>) -> tensor<1xf32>
    %1 = stablehlo.add %y, %0 : tensor<1xf32>
    return %1 : tensor<1xf32>
  }
}
```

**Bisection:**
- **Crashes:** a 1-element slice at any offset (also `[0:1]`, also from a `tensor<4xf32>`), whether added, broadcast to `64x1`, or reshaped to a scalar and broadcast.
- **Compile fine:** a 2-element slice broadcast to `64x2`; an 8-element slice broadcast to `64x8`; a `tensor<1xf32>` *argument* broadcast to `64x1`.
- **Only on wasm32:** the same program with the same WebGPU flags compiles fine with the **wasm64** build of iree-base-compiler (emscripten-forge, same version). It also compiles natively (PyPI iree-base-compiler 3.14.0, targets `vulkan` and `llvm-cpu`). So this is a memory bug of the 32-bit compiler build, not of the WebGPU code generation. A 32-bit pointer or size assumption somewhere is likely.

**Workaround:** never slice one element. Slice 2 elements and pick one: `x[i-1:i+1] @ [[0],[1]]`.

## 2. WebGPU: one buffer bound read-only *and* read-write in one dispatch

**What happens:** IREE packs intermediate results into one buffer and binds two ranges of it, one read-only and one read-write, in the same dispatch. WebGPU forbids this, even for ranges that do not overlap:
```
[Buffer (unlabeled)] usage (Storage(read-write)|Storage(read-only)) includes writable usage
and another usage in the same synchronization scope.
 - While validating compute pass usage.
[Invalid CommandBuffer] is invalid due to a previous error. - While calling [Queue].Submit(...)
```
Then: `webgpu_queue.c:1505: RESOURCE_EXHAUSTED; failed to create staging buffer for queue_write (256 bytes)`, and **the download never resolves, so the page hangs silently**.

**Reproducer** (jax → compile → run; a small 2-layer MLP on flat parameters):
```python
import jax.numpy as jnp, numpy as np
def fn(theta, obs):                      # theta (99,), obs (64, 4)
    w1 = theta[0:32].reshape(4, 8); b1 = theta[32:40]
    w2 = theta[40:48].reshape(8, 1); b2 = theta[47:49] @ jnp.array([[0.0], [1.0]])  # (see 1)
    return jnp.tanh(jnp.tanh(obs @ w1 + b1) @ w2 + b2)
mlir = jax.jit(fn).lower(np.zeros(99, np.float32), np.zeros((64, 4), np.float32)).as_text()
# compile_webgpu(mlir) in the compile worker, then: await run(binary, theta, obs)
```

**Workaround:** the global option `--iree-stream-resource-alias-mutable-bindings=true`. It is not accepted by `Session.set_flags` ("option not found"), so it has to be set with `iree.compiler.api.ctypes_dl._initializeGlobalCL("iree", "--iree-stream-resource-alias-mutable-bindings=true")` before compiling.

Global options can be set only **once per process**: a second `_initializeGlobalCL` call aborts the wasm process (`RuntimeError: unreachable`). So set every option you need in one call.

**Suggestion:** the WebGPU target should apply this by itself, since WebGPU always forbids mixed access to one buffer. `Session.set_flags` should also accept the option.

## 3. WebGPU: binding offsets not aligned to 256

**What happens:** with the workaround from 2, the merged bindings use offsets like 1664 and 3264. WebGPU requires `minStorageBufferOffsetAlignment` (256):
```
Offset (1664) of [Buffer (unlabeled)] does not satisfy the minimum
BufferBindingType::ReadOnlyStorage alignment (256).
 - While validating entries[1] against { binding: 1, ... ReadOnlyStorage ... }
 - While validating [BindGroupDescriptor] against [BindGroupLayout (unlabeled)]
```
After that, every allocation fails: `webgpu_allocator.c:368: RESOURCE_EXHAUSTED; WebGPU device.createBuffer failed for 32768 bytes`.

It shows up when a call's outputs are fed into the next call. IREE returns the outputs as slices of one packed result buffer, at unaligned offsets. A single call runs fine.

**Reproducer** (only option 2 set; two chained PPO updates as in training; `webgym.algos.ppo_jax.make_update` is plain jax: an MLP policy and value loss, its gradient and Adam):
```python
actor, critic = [4, 32, 1], [4, 32, 1]
theta = np.zeros(ppo_jax.count(actor) + ppo_jax.count(critic) + 1, np.float32)
args = (theta, theta, theta, f32(0), *ppo_jax.batch_arrays(2048, 4, 1), f32(3e-3), f32(0.2))
# compile ppo_jax.make_update(actor, critic) for args, then:
first = fn.submit(*uploaded_args, wait_fences=upload_fences)
theta, m, v, t, losses = first.value                      # outputs stay on the GPU
await fn.submit(theta, m, v, t, *uploaded_args[4:], wait_fences=[first.fence])   # -> the error
```

**Workaround:** add the global option `--iree-stream-resource-min-offset-alignment=256`, set the same way as in 2. **Status:** with options 2 and 3, PPO training ran on WebGPU in headless chromium (swiftshader) with no GPU errors: 6 updates of 4–12 minibatches each. Desktop Chrome has not been re-tested yet.

**Suggestion:** the WebGPU target should take the alignment from the device limits by default.

## 4. Constant expressions → a module initializer that hangs `add_vm_module`

**What happens:** jax lowers `jnp.log(<python constant>)` to a `log` op on a constant. With `--iree-opt-const-eval=false` this becomes a module initializer. `ctx.add_vm_module(module)` runs it on the GPU, it needs a blocking wait, and in the browser it **hangs forever, without an error**.

**Reproducer:**
```python
import jax.numpy as jnp, numpy as np
fn = lambda th: th[-2:] * 2.0 + 0.5 * jnp.log(2 * jnp.pi * jnp.e)
mlir = jax.jit(fn).lower(np.zeros(99, np.float32)).as_text()
# compile, then run(binary, theta): hangs in add_vm_module
```
Without the `jnp.log(...)` term (`th[-2:] * 2.0 + 1.2345`) it loads and runs in 0.1 s.

**Workaround:** compute constants in Python (`math.log(...)`).

**Suggestion:** initializers on WebGPU need an async path, or `add_vm_module` should raise "blocking wait unavailable" instead of hanging. Enabling const-eval would also remove this case. We disabled it because const-eval needs a host (llvm-cpu) backend in the wasm compiler; that hasn't been re-checked.

## 5. jaxlib and iree-base-compiler cannot be imported into one Python (wasm)

**Reproducer:**
```python
import jax          # jaxlib's MLIR bindings
import iree.compiler.api
```
```
Critical nanobind error: refusing to add duplicate key "NotSpeculatable" to enumeration "Speculatability"
Aborted()   -> wasm `unreachable`
```

**Cause:** both ship MLIR Python bindings built with nanobind, and in the emscripten builds they register into the same nanobind domain. Natively (PyPI) they coexist.

**Fix (recipe):** build one of them with its own domain, e.g. `-DNB_DOMAIN=iree`.

**Workaround:** compile in a second Web Worker (a second Python). It costs start-up time, memory and the 29 MB compiler download.

## 6. Usability

- **Blocking waits:** the synchronous runtime API fails with `blocking waits are unavailable`, which is the first thing everyone runs into in a browser. A clear error that names the async API (`AsyncSession`, `enqueue_upload`, `--iree-execution-model=async-external`), plus a minimal browser example, would help.
- **Many queued submissions:** queueing about 64 chained submissions at once, each with its own uploads and about 1 MB of temporaries, ended in `createBuffer failed` on desktop Chrome. It is not yet clear whether this was a consequence of 3 or a separate problem.
- **jax versions:** the browser's jax 0.11.2.dev differs from native jax 0.12. So StableHLO for browser tests must be generated by the browser's jax.

## Speed, for reference
All numbers are from headless chromium with swiftshader, a CPU emulating the GPU; a real GPU should do better.
- **Probe:** a 4096x256 @ 256x256 tanh layer, chained 20 times, takes about 10 ms per call on WebGPU, against about 21 ms with numpy on the CPU (wasm). It matches numpy to within 1.5e-6.
- **PPO update:** one minibatch of 2048 (gradient + Adam, hidden 32) takes 21 ms on WebGPU, against 87 ms with jax on the CPU (wasm).
