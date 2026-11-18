"""jax functions on the GPU in the browser: jax -> StableHLO -> IREE (compiled in the
page for WebGPU) -> run asynchronously through IREE's WebGPU runtime.

Natively the same path compiles for IREE's CPU target (the PyPI compiler has no
WebGPU backend): it tests the export, the compiler and the numbers.

Needs jax and iree-base-runtime; the compiler (iree-base-compiler) runs in the browser
in a separate compile worker: jaxlib and the IREE compiler both bring MLIR python
bindings, which abort when loaded into one process (nanobind: duplicate types).
"""

import base64
import sys
import time

import numpy as np

IN_BROWSER = sys.platform == "emscripten"


def log(message):
    """progress of the GPU setup (the browser console; the last line before a crash
    tells where it happened)"""
    print(f"webgym.gpu: {message}", flush=True)


def stablehlo(fn, *examples):
    """the StableHLO (MLIR text) of jax.jit(fn) for arguments shaped like `examples`"""
    import jax

    return jax.jit(fn).lower(*examples).as_text()


_global_flags_set = False


def compile_module(mlir, target):
    """IREE bytecode of a StableHLO module; target "webgpu" (browser) or "cpu".
    WebGPU functions use the coarse-fences ABI (async submission, see AsyncSession)"""
    from iree.compiler.api import Output, Session, Source, ctypes_dl

    flags = ["--iree-input-type=stablehlo", "--iree-opt-const-eval=false"]
    if target == "webgpu":
        # WebGPU rules IREE does not apply by itself (REPORT_BUGS.md): one binding per
        # buffer (no buffer bound read-only and read-write in the same dispatch) and
        # binding offsets aligned to 256 bytes (minStorageBufferOffsetAlignment).
        # Global options (not session flags): set once per process
        global _global_flags_set
        if not _global_flags_set:
            ctypes_dl._initializeGlobalCL("iree", "--iree-stream-resource-alias-mutable-bindings=true",
                                          "--iree-stream-resource-min-offset-alignment=256")
            _global_flags_set = True
        flags += ["--iree-hal-target-device=webgpu", "--iree-execution-model=async-external"]
    else:
        flags += ["--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu"]
        if IN_BROWSER:
            bits = 64 if sys.maxsize > 2**32 else 32
            flags += [f"--iree-llvmcpu-target-triple=wasm{bits}-unknown-emscripten",
                      "--iree-llvmcpu-target-cpu-features=+simd128"]
    with Session() as session:
        session.set_flags(*flags)
        with session.invocation() as invocation, Source.wrap_buffer(session, mlir.encode()) as source:
            invocation.enable_console_diagnostics()
            if not (invocation.parse_source(source) and invocation.execute()):
                raise RuntimeError("IREE could not compile the function (see the console)")
            with Output.open_membuffer() as output:
                invocation.output_vm_bytecode(output)
                return bytes(output.map_memory())


def compile_b64(mlir, target="webgpu"):
    """compile_module as base64 (the page's compile worker runs this: it has the IREE
    compiler, but not jax: their MLIR python bindings cannot share one process)"""
    return base64.b64encode(compile_module(mlir, target)).decode()


async def compile_webgpu(mlir):
    """IREE bytecode for WebGPU: in the browser from the page's compile worker
    (worker.js: self.webgymCompile), natively directly"""
    if IN_BROWSER:
        import pyjs

        return base64.b64decode(str(await pyjs.js.webgymCompile(mlir)))
    return compile_module(mlir, "webgpu")


class GpuFunction:
    """a jax function compiled for the WebGPU device: `submit(*arrays)` queues it
    (returns an AsyncResult, chain with wait_fences=[previous.fence]), `download`
    copies a result to numpy (await both). Create with `await GpuFunction.create(...)`."""

    @classmethod
    async def create(cls, fn, *examples):
        log("importing the IREE runtime")
        import iree.runtime as rt
        from iree.runtime.asyncio import AsyncSession
        from iree.runtime.webgpu import get_webgpu_device

        self = cls()
        t0 = time.perf_counter()
        log("lowering to StableHLO")
        mlir = stablehlo(fn, *examples)
        log("compiling for WebGPU (compile worker)")
        binary = await compile_webgpu(mlir)
        self.compile_seconds = time.perf_counter() - t0
        log(f"compiled ({len(binary)} bytes), getting the WebGPU device")
        self.device = await get_webgpu_device()
        log("loading the module")
        config = rt.Config(device=self.device)
        self.context = rt.SystemContext(config=config)
        module = rt.VmModule.copy_buffer(config.vm_instance, binary)
        log("adding the module (creates the GPU pipelines)")
        self.context.add_vm_module(module)
        main = getattr(self.context.modules, module.name).main
        self.session = AsyncSession(self.device)
        self.function = self.session.bind(main)
        log(f"loaded ({time.perf_counter() - t0:.1f} s since lowering)")
        return self

    def upload(self, array):
        """queue a copy of a numpy array to the GPU -> AsyncResult (.value: the GPU
        array, .fence: when it is there). Arrays must be uploaded like this: the
        browser allows no blocking (synchronous) copies"""
        # (np.asarray, not np.ascontiguousarray: that makes 0-d scalars 1-d)
        return self.session.enqueue_upload(np.asarray(array, order="C"))

    def submit(self, *args, wait_fences=()):
        """queue a call with GPU arrays (after `wait_fences`) -> AsyncResult"""
        return self.function.submit(*args, wait_fences=wait_fences)

    async def download(self, value):
        return await self.session.download(value)


async def webgpu_available():
    """is there a WebGPU adapter (browser, also in a worker)?"""
    if not IN_BROWSER:
        return False
    import pyjs

    gpu = pyjs.js.navigator.gpu
    if pyjs.js.Boolean(gpu) is False:
        return False
    adapter = await gpu.requestAdapter()
    return pyjs.js.Boolean(adapter) is not False


async def adapter_name():
    """which GPU the browser gives WebGPU, e.g. "apple metal-3"; "software" marks a
    fallback adapter (a CPU emulating a GPU, e.g. swiftshader)"""
    import pyjs

    adapter = await pyjs.js.navigator.gpu.requestAdapter()
    info = adapter.info
    parts = [str(getattr(info, k, "")) for k in ("vendor", "architecture")]
    name = " ".join(p for p in parts if p and p != "undefined") or str(getattr(info, "description", "")) or "unknown"
    fallback = getattr(info, "isFallbackAdapter", None)
    if fallback is None or str(fallback) == "undefined":
        fallback = getattr(adapter, "isFallbackAdapter", False)
    return f"{name} (software fallback!)" if str(fallback) == "true" else name


async def probe(n=4096, d=256, repeat=20):
    """compile a small dense network step for WebGPU, run it `repeat` times chained on
    the GPU, compare with numpy; returns a one-line report (timings GPU vs CPU)"""
    import jax.numpy as jnp

    log("checking for a WebGPU adapter")
    if not await webgpu_available():
        return "WebGPU: not available in this browser (training uses the CPU)"
    log(f"WebGPU adapter found: {await adapter_name()}")

    def layer(x, w):
        return jnp.tanh(x @ w)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    fn = await GpuFunction.create(layer, x, w)
    log("uploading")
    x_gpu, w_gpu = fn.upload(x), fn.upload(w)
    uploaded = [x_gpu.fence, w_gpu.fence]
    # the first call warms up (pipelines), then `repeat` chained calls, one download
    log("first run on the GPU")
    await fn.download(await fn.submit(x_gpu.value, w_gpu.value, wait_fences=uploaded))
    log("timing")
    t0 = time.perf_counter()
    result = fn.submit(x_gpu.value, w_gpu.value)
    for _ in range(repeat - 1):
        result = fn.submit(result.value, w_gpu.value, wait_fences=[result.fence])
    out = await fn.download(await result)
    gpu_ms = 1e3 * (time.perf_counter() - t0) / repeat
    t0 = time.perf_counter()
    expected = x
    for _ in range(repeat):
        expected = np.tanh(expected @ w)
    cpu_ms = 1e3 * (time.perf_counter() - t0) / repeat
    error = float(np.abs(out - expected).max())
    return (f"WebGPU ok ({await adapter_name()}): compiled in {fn.compile_seconds:.1f} s, {n}x{d} @ {d}x{d} layer: "
            f"GPU {gpu_ms:.2f} ms, numpy CPU {cpu_ms:.2f} ms, max diff {error:.1e}")
