"""Browser host (runs in a pyjs web worker), in one of three roles:

* "single": everything in one worker (phones / tablets): WebGL2 on the OffscreenCanvas
  the page transferred, frames from requestAnimationFrame, training sliced into frames
* "display": the same, but it never trains: the trainer's latest policy (snapshots)
  drives the watched env (replay in real time, play, drag)
* "trainer": no canvas: trains in chunks as fast as it can (between them it takes
  commands), sends snapshots of the policy, the history, checkpoints

worker.js sets `self.canvas` (the OffscreenCanvas, display / single) and
`self.webgymOptions` (the page's URL options + the role) before starting this.
"""

import asyncio
import base64
import ctypes
import json
import time
import traceback

import pyjs

from webgym.app import App, NeedPackage

AUTOSAVE = "autosave"  # the name of the automatic checkpoint (one per env + algo)
AUTOSAVE_SECONDS = 2.0  # while training: at most this often
TRAIN_CHUNK_S = 0.25  # trainer: training time between looking at commands
SNAPSHOT_SECONDS = 0.25  # trainer: the policy to the display at most this often


class _WebGLContextAttributes(ctypes.Structure):
    """`EmscriptenWebGLContextAttributes` (emscripten/html5_webgl.h)
    (from pyb2d3/examples/pyb2d3_sandbox_webgl/webgl_frontend.py)"""

    _fields_ = [
        ("alpha", ctypes.c_bool),
        ("depth", ctypes.c_bool),
        ("stencil", ctypes.c_bool),
        ("antialias", ctypes.c_bool),
        ("premultipliedAlpha", ctypes.c_bool),
        ("preserveDrawingBuffer", ctypes.c_bool),
        ("powerPreference", ctypes.c_int),
        ("failIfMajorPerformanceCaveat", ctypes.c_bool),
        ("majorVersion", ctypes.c_int),
        ("minorVersion", ctypes.c_int),
        ("enableExtensionsByDefault", ctypes.c_bool),
        ("explicitSwapControl", ctypes.c_bool),
        ("proxyContextToMainThread", ctypes.c_int),
        ("renderViaOffscreenBackBuffer", ctypes.c_bool),
        ("desynchronized", ctypes.c_bool),
    ]


def create_webgl2_context(canvas, antialias=True):
    """create a WebGL2 context on `canvas` (also an OffscreenCanvas) and make it
    current (the GL functions PyOpenGL calls use emscripten's current context)"""
    pyjs.set_canvas(canvas)
    lib = ctypes.CDLL(None)
    attrs = _WebGLContextAttributes()
    lib.emscripten_webgl_init_context_attributes(ctypes.byref(attrs))
    attrs.majorVersion, attrs.minorVersion = 2, 0
    attrs.alpha = attrs.depth = attrs.stencil = False
    attrs.antialias = bool(antialias)
    create = lib.emscripten_webgl_create_context
    create.restype = ctypes.c_size_t
    create.argtypes = [ctypes.c_char_p, ctypes.POINTER(_WebGLContextAttributes)]
    ctx = create(b"#canvas", ctypes.byref(attrs))
    if ctx == 0:
        raise RuntimeError("cannot create a WebGL2 context on the canvas")
    make_current = lib.emscripten_webgl_make_context_current
    make_current.argtypes = [ctypes.c_size_t]
    if make_current(ctx) != 0:
        raise RuntimeError("cannot make the WebGL2 context current")
    return ctx


def post(message):
    """send a JSON-able dict to the page (main thread)"""
    pyjs.js.postMessage(pyjs.js.JSON.parse(json.dumps(message)))


class WebHost:
    """drives the shared App in the worker (see the module docstring for the roles)"""

    def __init__(self, options):
        """`options`: the page's URL query parameters (env, algo, population, seed,
        speed) and "role" ("single" / "display" / "trainer")"""
        self.role = options.get("role", "single")
        kwargs = {k: options[k] for k in ("env", "algo") if k in options}
        kwargs.update({k: int(options[k]) for k in ("population", "seed", "speed") if k in options})
        self.app = App(**kwargs)
        self.frames = 0
        self.history_sent = 0  # entries of app.history already sent to the page
        self.autosaved_at = 0.0  # time.monotonic() of the last autosave
        self.snapshot_at = 0.0  # trainer: time.monotonic() of the last snapshot
        self.state_at = 0.0  # trainer: ... of the last state message
        # worker.js forwards the page's pointer events / widget changes / checkpoints
        # to restore / snapshots of the trainer's policy to these
        functions = (self.event, self.command, self.restore, self.snapshot)
        self._callables = [pyjs.create_callable(f) for f in functions]
        pyjs.js.Function(
            "e", "c", "r", "s",
            "self.webgymEvent = e; self.webgymCommand = c; self.webgymRestore = r; self.webgymSnapshot = s",
        )(*(js for js, _ in self._callables))
        self.send_controls()
        if self.role == "trainer":
            self.send_snapshot()
            self.task = asyncio.ensure_future(self.train_loop())
        else:
            self.canvas = pyjs.js.canvas
            self.ctx = create_webgl2_context(self.canvas)
            self.loop = pyjs.request_animation_frame_loop(self.frame)

    # -- trainer ---------------------------------------------------------------------

    async def train_loop(self):
        """train in chunks of TRAIN_CHUNK_S as long as training is on; between the
        chunks the worker handles messages (commands, stop)"""
        while True:
            try:
                training = self.app.training
                if training:
                    self.app._train_steps(TRAIN_CHUNK_S)
                self.after_update(training)
                now = time.monotonic()
                if now - self.state_at > 0.25:
                    post({"type": "state", "state": self.app.state()})
                    self.state_at = now
            except Exception:
                post({"type": "error", "error": traceback.format_exc()})
                self.app.stop()
                self.send_controls()
            await asyncio.sleep(0 if self.app.training else 0.05)

    def send_snapshot(self):
        """the current policy (a checkpoint) for the display worker"""
        post({"type": "snapshot", "data": base64.b64encode(self.app.checkpoint()).decode()})
        self.snapshot_at = time.monotonic()

    # -- display -----------------------------------------------------------------------

    def snapshot(self, data):
        """display: the trainer's latest policy drives the watched env"""
        try:
            self.app.load_policy(base64.b64decode(str(data)))
        except Exception:
            post({"type": "error", "error": traceback.format_exc()})

    def after_update(self, training_before):
        """after training steps: controls when training ended, new history entries (+
        autosave, snapshot for the display)"""
        if self.app.training != training_before:  # training finished / stopped
            self.send_controls()
            self.autosave()
        if len(self.app.history) < self.history_sent:
            self.send_controls()
        elif len(self.app.history) > self.history_sent:
            post({"type": "history", "entries": self.app.history[self.history_sent :]})
            self.history_sent = len(self.app.history)
            if time.monotonic() - self.autosaved_at > AUTOSAVE_SECONDS:
                self.autosave()
            if self.role == "trainer" and time.monotonic() - self.snapshot_at > SNAPSHOT_SECONDS:
                self.send_snapshot()

    def send_controls(self):
        """the widgets (app.controls()) and the whole history (after a reset / new run)"""
        post({"type": "controls", "controls": self.app.controls()})
        post({"type": "history", "reset": True, "entries": self.app.history})
        self.history_sent = len(self.app.history)

    # -- checkpoints (stored by the page in IndexedDB, per env + algo) --------------

    def send_checkpoint(self, name):
        """the current run as a checkpoint named `name` to the page (base64 .safetensors)"""
        app = self.app
        last = app.history[-1] if app.history else {}
        post({
            "type": "checkpoint",
            "name": name,
            "env": app.env_name,
            "algo": app.algo,
            "iteration": last.get("iteration", 0),
            "mean_return": last.get("mean_return"),
            "data": base64.b64encode(app.checkpoint()).decode(),
        })
        self.autosaved_at = time.monotonic()

    def autosave(self):
        """the run's autosave slot (one per env + algo), when there is something to save"""
        if self.app.train_state is not None:
            self.send_checkpoint(AUTOSAVE)

    def restore(self, data):
        """a checkpoint from the page (base64); the run continues from it"""
        try:
            self.app.restore(base64.b64decode(str(data)))
            self.send_controls()
            if self.role == "trainer":
                self.send_snapshot()
        except Exception:
            post({"type": "error", "error": traceback.format_exc()})

    def frame(self, time_ms):
        """one animation frame; errors are reported, the loop goes on in watch mode"""
        try:
            training = self.app.training
            self.app.frame(int(self.canvas.width), int(self.canvas.height))
            self.after_update(training)
            self.frames += 1
            if self.frames % 15 == 0:
                post({"type": "state", "state": self.app.state()})
        except Exception:
            post({"type": "error", "error": traceback.format_exc()})
            self.app.watch()
        return True

    def command(self, name, value):
        """a widget of the page changed, see App.command"""
        if name == "__crash__":
            # tests of the page's crash handling: a real wasm trap (memory access out
            # of bounds: read beyond the 4 GB of wasm32)
            ctypes.string_at(2**32 - 16, 64)
        try:
            name = str(name)
            if name == "save":  # a named checkpoint (the page stores it)
                self.send_checkpoint(str(value))
                return
            training = self.app.training
            try:
                self.app.command(name, None if value is None else value)
            except NeedPackage as e:
                # the page loads it (progress bar), then sends the command again
                post({"type": "need_package", "package": e.package, "command": name})
                return
            if name == "stop" or (training and not self.app.training):
                self.autosave()
            if name == "new_run":  # the run is gone: so is its autosave
                post({"type": "delete_checkpoint", "name": AUTOSAVE, "env": self.app.env_name,
                      "algo": self.app.algo})
            self.send_controls()
            if self.role == "trainer" and name in ("env", "algo", "new_run", "stop", "train"):
                self.send_snapshot()  # (the display shows the current policy / env)
        except Exception:
            post({"type": "error", "error": traceback.format_exc()})


    def event(self, kind, x, y, delta, key):
        """pointer / key event from the page (drawing-buffer px), see App.event"""
        try:
            self.app.event(str(kind), float(x), float(y), float(delta), str(key))
        except Exception:
            post({"type": "error", "error": traceback.format_exc()})


def start():
    """start the host with the page's options (worker.js: self.webgymOptions, JSON);
    returns it (keep it alive in a global)"""
    host = WebHost(json.loads(str(pyjs.js.self.webgymOptions)))
    post({"type": "ready"})
    return host
