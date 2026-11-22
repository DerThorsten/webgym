"""The env base class (gymnasium-like, no gymnasium dependency) + the env registry.

An env holds `n` agents in ONE box2d world: they share the scenery (ground) but don't
touch each other (collision filters). The API is batched: numpy arrays with one row per
agent, much faster than one world per agent (one world.step, batch reads / writes).

Writing an env (see envs/cartpole.py, the shortest one):

    @register_env("MyEnv")
    class MyEnv(Env):
        obs_names = ("x", "angle")       # what the agent sees: one name per value ...
        action_names = ("push",)         # ... and what it controls (actions in [-1, 1])

        def build_scenery(self):         # once: the static world
            self.scenery(b2d.box(10, 0.1))

        def build_agent(self, i):        # once per agent (i = 0 .. n-1): ONE agent
            cart = self.body((0, 1), b2d.box(0.5, 0.2))
            return {"cart": cart}        # its parts, by name

        # from here on all agents at once: self.cart is a b2d.Bodies batch of the n carts
        def randomize(self, rng):        # the random start of an episode
            self.cart.angular_velocity = rng.uniform(-1, 1, self.n)

        def act(self, actions):          # actions: (n, len(action_names))
            self.cart.apply_force_to_center(...)

        def observe(self):               # columns, one per obs_name: (n,) arrays
            return [self.cart.position[:, 0], self.cart.angle]

        def reward(self):                # -> reward (n,), episode over (n,)
            return np.ones(self.n), np.abs(self.cart.angle) > 1

Settings the user can choose (e.g. how many balls): `settings = {"balls": (1, (1, 2),
"explanation")}`; the env reads `self.balls`; obs_names / action_names may depend on them
(make them properties). See envs/water_cannon.py.
"""

import itertools
import sys

import numpy as np
import pyb2d3 as b2d

# collision categories: agents collide with the scenery and with category 1 (the
# default category of queries: the pointer can grab them), not with each other
SCENERY, AGENT, PARTICLE, SENSOR = 4, 2, 8, 16
AGENT_FILTER = b2d.make_filter(category_bits=AGENT, mask_bits=1 | SCENERY)
SCENERY_FILTER = b2d.make_filter(category_bits=SCENERY)
DECORATION_FILTER = b2d.make_filter(category_bits=SCENERY, mask_bits=0)  # (collides with nothing)
SENSOR_FILTER = b2d.make_filter(category_bits=SENSOR, mask_bits=SCENERY)  # (senses the scenery)
# decoration particles (watch mode only): bounce off the scenery, not off the agents or
# each other, not grabbable
PARTICLE_FILTER = b2d.make_filter(category_bits=PARTICLE, mask_bits=SCENERY)
# colors (0xRRGGBB): an env may give its bodies their own (body / shape: color=)
PARTICLE_COLOR = 0xFFA040
SCENERY_COLOR = 0x4E8A68
PROP_COLOR = 0xB08850
AGENT_COLOR = 0xE8955A  # (an agent's bodies, unless the env chooses)
DARK_COLOR, LIGHT_COLOR = 0x5E6C82, 0xEFE2C8  # (more for an agent's parts: slate, cream)
SENSOR_COLOR = 0x010203  # (sensor shapes: not drawn)


class Box:
    """continuous space: arrays of `shape` in [low, high]"""

    def __init__(self, low, high, shape):
        self.shape = tuple(shape)
        self.low = np.broadcast_to(np.float32(low), self.shape)
        self.high = np.broadcast_to(np.float32(high), self.shape)

    def sample(self, rng, n=None):
        size = self.shape if n is None else (n, *self.shape)
        return rng.uniform(self.low, self.high, size).astype(np.float32)


class Env:
    """n agents in one box2d world, stepped with a fixed dt per env step (see the module
    docstring: how to write one). The module docstring of an env is its description in
    the UI."""

    obs_names = ()  # the observation values ...
    action_names = ()  # ... and the actions (each in [-1, 1]); the network view shows them
    dt = 1 / 60
    substeps = 4
    max_steps = 1000
    # the best return an episode can get, if known (e.g. CartPole: 1 per step, 500 steps);
    # the page shows it next to the returns and marks a perfect score; None: unknown
    max_return = None
    # looks: a disc on each hinge of the current policy's agent (draw_hinges); False: none
    hinge_discs = True
    gravity = (0, -10)
    # initial camera: world point at the view center, half height in world units
    view_center = (0.0, 0.0)
    view_zoom = 10.0
    # True: n agents share one world without affecting each other (only static scenery);
    # False: the agents change the world (e.g. a heap of loose balls): a world per agent
    # (make() returns Separate; the Population slider ends at MAX_SEPARATE)
    batchable = True
    # optional: the env also works with a few discrete actions (for DQN & co): their
    # action vectors (k, len(action_names)) and names; None: continuous only
    discrete_actions = None
    discrete_names = ()
    # settings the user can choose (the page shows them, a change starts a new run; runs
    # and checkpoints are kept per setting): {name: (default, choices, explanation)};
    # the env reads them as attributes (self.<name>), also in obs_names / action_names
    settings = {}
    # which algorithms work well here, per time budget ("quick" ~1 min, "medium" ~5 min,
    # "long" ~20 min of training in the browser), best first, each with parameters that
    # differ from its defaults: {budget: {algo: {param: value}}}; the page picks the first
    # one when you open the env or change the budget (rough guesses, not measured)
    suggested = {}

    def __init__(self, n=1, decorated=False, **settings):
        for name, value in self.chosen(settings).items():
            setattr(self, name, value)
        self.n = n
        # decoration: looks only (flags, thrust particles), for the env you watch; never
        # in training (build_* leave it out, decorate() is called by the viewer only)
        self.decorated = decorated
        self.world = b2d.World(gravity=self.gravity)  # (raises beyond 128 worlds)
        self._bodies, self._owner = [], []  # all agent bodies, the agent of each
        self.build_scenery()
        parts = {}
        for i in range(n):
            self._agent = i
            for name, part in self.build_agent(i).items():
                parts.setdefault(name, []).extend(part if isinstance(part, (list, tuple)) else [part])
        for name, items in parts.items():
            setattr(self, name, _batch(items))
        self._hinges = _hinges([b for b, i in zip(self._bodies, self._owner) if i == 0]) if self.hinge_discs else []
        self._bodies = b2d.Bodies(self._bodies)
        self._owner = np.array(self._owner)
        self._start = self.world.get_snapshot()
        self.steps = 0
        self.done = np.zeros(n, dtype=bool)  # the agent's episode ended
        self.episode_return = np.zeros(n)
        # finished agents are disabled (not simulated, not drawn) until the next reset;
        # False: they go on (watch mode without auto reset)
        self.freeze_done = True

    @property
    def observation_space(self):
        return Box(-np.inf, np.inf, (len(self.obs_names),))

    @property
    def action_space(self):
        return Box(-1.0, 1.0, (len(self.action_names),))

    # -- what an env implements (see the module docstring) -------------------------

    def build_scenery(self):
        """the static world (once): ground, walls, ... (with self.scenery)"""

    def build_agent(self, i):
        """ONE agent (bodies with self.body / self.shape, joints with self.world,
        sensors with self.sensor) -> its parts {name: object or list of objects}. Each
        name becomes a batch over all agents (b2d.Bodies, b2d.Shapes, b2d.RevoluteJoints,
        ...; other objects: a list), agent after agent: with a list of k objects per
        agent the batch has n * k entries (reshape(n, k) gives one row per agent)"""
        raise NotImplementedError

    def randomize(self, rng):
        """the random start of all agents (after the world went back to its start)"""

    def act(self, actions):
        """apply actions (n, len(action_names)), each in [-1, 1]"""
        raise NotImplementedError

    def observe(self):
        """-> the observation as columns: (n,) arrays, or (n, k) for k values"""
        raise NotImplementedError

    def reward(self):
        """-> (reward (n,), episode over (n,) bool) after a step"""
        raise NotImplementedError

    def decorate(self):
        """decorated envs only, after each watched step: looks that follow the agents
        (e.g. thrust particles); default: nothing"""

    def camera_target(self, i):
        """the world point (x, y) the camera may follow for agent i ("Camera follows");
        default None: a fixed scene, nothing to follow"""
        return None

    def draw(self, debug_draw):
        """looks only: more to draw after the world (watch mode), e.g. lines with
        debug_draw.draw_segment(p1, p2, 0xRRGGBB); the watched env has one agent, the
        current policy; default: nothing"""

    # -- building blocks for build_scenery / build_agent -----------------------------

    def scenery(self, geometry, position=(0, 0), solid=True, color=SCENERY_COLOR, **material):
        """a static body with one shape (ground, walls, ...) -> the body; solid=False:
        decoration only, drawn but nothing collides with it (e.g. the lander's flags);
        `color`: how it looks (0xRRGGBB)"""
        body = self.world.create_static_body(position=position)
        shape_def = b2d.shape_def(
            filter=SCENERY_FILTER if solid else DECORATION_FILTER,
            material=b2d.surface_material(custom_color=color, **material),
            enable_sensor_events=solid,  # (foot sensors see it, see sensor)
        )
        body.create_shape(shape_def, geometry)
        return body

    def prop(self, position, geometry, density=0.5, friction=0.6):
        """a loose object of the scenery (a crate, a ball): dynamic, the agents push it
        around, so they change the world (an env with props: batchable = False)"""
        body = self.world.create_dynamic_body(position=position)
        shape_def = b2d.shape_def(
            density=density, filter=SCENERY_FILTER, enable_sensor_events=True,
            material=b2d.surface_material(friction=friction, custom_color=PROP_COLOR),
        )
        body.create_shape(shape_def, geometry)
        return body

    def body(self, position, geometry, density=1.0, friction=0.8, filter=AGENT_FILTER, color=AGENT_COLOR,
             material=None, **body_settings):
        """a dynamic body of the agent being built, with one shape (more: self.shape);
        `filter`: what it collides with (default: the scenery, not other agents);
        `color`: how it looks (0xRRGGBB); `material`: more of its surface
        ({"rolling_resistance": 0.2}). Agent bodies never sleep: a joint motor doesn't
        wake a sleeping body, and an agent lying still (fallen, auto reset off) must still
        move when its policy acts"""
        body = self.world.create_dynamic_body(position=position, **{"enable_sleep": False, **body_settings})
        self._bodies.append(body)
        self._owner.append(self._agent)
        self.shape(body, geometry, density, friction, filter, color, **(material or {}))
        return body

    def shape(self, body, geometry, density=1.0, friction=0.8, filter=AGENT_FILTER, color=AGENT_COLOR, **material):
        """one more shape on an agent body (`color`: how it looks; `material`: more of
        its surface, e.g. rolling_resistance)"""
        shape_def = b2d.shape_def(density=density, filter=filter,
                                  material=b2d.surface_material(friction=friction, custom_color=color, **material))
        return body.create_shape(shape_def, rounded(geometry))

    def sensor(self, body, geometry, filter=SENSOR_FILTER, density=1.0):
        """a sensor shape on an agent body: `sensor_capacity > 0` (batched: b2d.Shapes)
        while it overlaps the scenery, e.g. a foot touching the ground. (It has mass, as
        any shape: `density=0` for none; the older envs were trained with 1)"""
        shape_def = b2d.shape_def(is_sensor=True, enable_sensor_events=True, filter=filter, density=density,
                                  material=b2d.surface_material(custom_color=SENSOR_COLOR))  # (not drawn)
        return body.create_shape(shape_def, geometry)

    # -- the API (used by the trainers and the app) ----------------------------------

    @classmethod
    def chosen(cls, settings=None):
        """all settings: the defaults, overridden by `settings` (checked)"""
        values = {name: default for name, (default, _, _) in cls.settings.items()}
        for name, value in (settings or {}).items():
            if name not in cls.settings or value not in cls.settings[name][1]:
                raise ValueError(f"{cls.__name__}: no setting {name} = {value!r}")
            values[name] = value
        return values

    @classmethod
    def description(cls):
        """what the env is about (its module's docstring)"""
        return (sys.modules[cls.__module__].__doc__ or "").strip()

    def reset(self, seed=None):
        """a new episode for all agents -> obs (n, obs_dim)"""
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        self.world.restore_from_snapshot(self._start)  # (re-enables every body)
        self.steps = 0
        self.done[:] = False
        self.episode_return[:] = 0.0
        self.randomize(self.rng)
        return self._observation()

    def step(self, actions):
        """apply `actions` (n, act_dim) for one env step ->
        (obs (n, obs_dim), reward (n,), terminated (n,), truncated (n,)); agents whose
        episode already ended get reward 0 and no new end"""
        actions = np.clip(np.asarray(actions, dtype=np.float32).reshape(self.n, -1), -1, 1)
        if self.freeze_done and self.done.any():
            actions = np.where(self.done[:, None], 0, actions)
        self.act(actions)
        self.world.step(self.dt, self.substeps)
        self.steps += 1
        reward, terminated = self.reward()
        reward = np.where(self.done, 0.0, reward)
        terminated = terminated & ~self.done
        truncated = np.full(self.n, self.steps >= self.max_steps) & ~self.done
        ended = terminated | truncated
        self.episode_return += reward
        if ended.any():
            self.done |= ended
            if self.freeze_done:
                self._bodies[np.isin(self._owner, np.flatnonzero(ended))].enabled = False
        return self._observation(), reward, terminated, truncated

    def _observation(self):
        return np.column_stack(self.observe()).astype(np.float32)

    def draw_hinges(self, debug_draw):
        """looks: a disc on each hinge (revolute joint) of agent 0, the current policy,
        sized to the parts it joins (the app draws them on top, for every env)"""
        for body, p, radius in self._hinges:
            debug_draw.draw_solid_circle(b2d.transform(p=body.world_point(p)), radius, LIGHT_COLOR)

    @property
    def follows(self):
        """does this env suggest a camera target (overrides camera_target)?"""
        return type(self).camera_target is not Env.camera_target

    def close(self):
        """free the box2d world now (box2d has at most b2d.MAX_WORLDS = 128 worlds at a
        time; don't wait for the garbage collector)"""
        self.world.destroy()


ROUND = 0.04  # m: the corner radius of an agent's polygons (see rounded)


def rounded(geometry):
    """a polygon with rounded corners, the same outer size (an agent's boxes: they look
    softer and slide over edges instead of catching on them); thin ones (less than 4 cm
    across: the cart poles' poles) and other geometry (circles, capsules, polygons that
    are rounded already) stay as they are"""
    if not isinstance(geometry, b2d.Polygon) or geometry.radius > 0:
        return geometry
    v, n = np.array(geometry.vertices), np.array(geometry.normals)  # (edge i: v[i] -> v[i+1], normal n[i])
    extent = min(float(np.dot(n[i], v[i] - np.array(geometry.centroid))) for i in range(len(v)))
    if extent < ROUND:
        return geometry
    r = min(ROUND, 0.3 * extent)
    prev = np.roll(n, 1, axis=0)  # (the normal of the edge before each vertex)
    inset = v - r * (prev + n) / (1 + np.sum(prev * n, axis=1))[:, None]  # (each edge moved in by r)
    return b2d.polygon(points=inset, radius=r)


# -- helpers for envs: a ruler as the ground, discrete actions ---------------------------

RULER_COLOR, RULER_MARKS = 0xE6CF8E, 0x3B3326  # (a wooden ruler, dark marks)


def ruler(env, start=-10, end=390, depth=1.0, friction=1.0):
    """flat ground from x = start to end (m), `depth` deep, a ruler: a mark every 10 cm,
    longer ones every 50 cm and every metre (numbered: draw_ruler_numbers in the env's
    draw). The marks are looks only: nothing collides with them. Returns the ground body
    (its top at y = 0)"""
    ground = env.scenery(b2d.box((end - start) / 2, depth / 2, center=((start + end) / 2, -depth / 2)),
                         friction=friction, color=RULER_COLOR)
    mark = b2d.shape_def(filter=DECORATION_FILTER, material=b2d.surface_material(custom_color=RULER_MARKS))
    for k in range(start * 10, end * 10 + 1):  # (in cm / 10)
        length = 0.25 if k % 10 == 0 else 0.15 if k % 5 == 0 else 0.08
        ground.create_shape(mark, b2d.segment((k / 10, 0), (k / 10, -length)))
    return ground


# the digits of the ruler's numbers as segments (a 7-segment display: a top, b top
# right, c bottom right, d bottom, e bottom left, f top left, g middle)
_SEGMENTS = {"a": ((0, 1), (1, 1)), "b": ((1, 1), (1, 0.5)), "c": ((1, 0.5), (1, 0)), "d": ((0, 0), (1, 0)),
             "e": ((0, 0.5), (0, 0)), "f": ((0, 1), (0, 0.5)), "g": ((0, 0.5), (1, 0.5))}
_DIGITS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
           "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g"}


def draw_ruler_numbers(debug_draw, start=-10, end=390):
    """the ruler's metres in view (not its ends start, end), under their marks (lines:
    no text in the view)"""
    camera = debug_draw.camera
    half = camera.zoom * camera.width / camera.height
    w, h, gap = 0.07, 0.13, 0.03  # (a digit's width and height, the gap between digits, m)
    first = max(start + 1, int(np.floor(camera.center[0] - half)))
    for metre in range(first, min(end - 1, int(np.ceil(camera.center[0] + half))) + 1):
        text = str(metre)
        left = metre - (len(text) * (w + gap) - gap) / 2
        for i, char in enumerate(text):
            x = left + i * (w + gap)
            for (x0, y0), (x1, y1) in (_SEGMENTS[s] for s in _DIGITS[char]):
                debug_draw.draw_segment((x + x0 * w, -0.45 + y0 * h), (x + x1 * w, -0.45 + y1 * h), RULER_MARKS)


def bang_bang(groups, joints):
    """discrete actions (for DQN): every joint at full torque, one way or the other; the
    joints of a group (indices) together -> (actions (2^groups, joints), names like "+-+-")"""
    signs = np.array(list(itertools.product((1, -1), repeat=len(groups))))
    actions = np.zeros((len(signs), joints))
    for g, group in enumerate(groups):
        actions[:, group] = signs[:, g : g + 1]
    return actions, tuple("".join("+" if s > 0 else "-" for s in row) for row in signs)


def _thickness(body):
    """half the width of a body's (first, non-sensor) shape: a circle's or capsule's radius,
    a polygon's smallest distance from its center to an edge (+ its rounding)"""
    for shape in body.shapes:
        if shape.is_sensor:
            continue
        s = shape.cast()
        if shape.type == b2d.ShapeType.circle:
            return s.circle.radius
        if shape.type == b2d.ShapeType.capsule:
            return s.capsule.radius
        if shape.type == b2d.ShapeType.polygon:
            g = s.polygon
            v, n, c = np.array(g.vertices), np.array(g.normals), np.array(g.centroid)
            return min(float(np.dot(n[i], v[i] - c)) for i in range(len(v))) + g.radius
    return 0.05


def _hinges(bodies):
    """the revolute joints of these bodies as (body a, its local anchor point, a disc
    radius: 60 % of the thinner of the two parts it joins), each once (from its body b)"""
    hinges = []
    for body in bodies:
        for joint in body.joints:
            if joint.type == b2d.JointType.REVOLUTE and joint.body_b == body:
                a, p = joint.body_a, joint.local_frame_a.p
                radius = 0.6 * min(_thickness(a), _thickness(body))
                hinges.append((a, (float(p[0]), float(p[1])), radius))
    return hinges


def _batch(items):
    """the pyb2d3 batch of `items` (all of one type): bodies -> b2d.Bodies, shapes ->
    b2d.Shapes, joints -> b2d.RevoluteJoints / PrismaticJoints / ...; else the list"""
    first = items[0]
    if isinstance(first, b2d.Body):
        return b2d.Bodies(items)
    if isinstance(first, b2d.Shape):
        return b2d.Shapes(items)
    batch = getattr(b2d, type(first).__name__ + "s", None)  # (RevoluteJoint -> RevoluteJoints)
    return batch(items) if batch is not None else items


ENVS = {}


def register_env(name):
    """class decorator: make the env available as make(name)"""

    def register(cls):
        ENVS[name] = cls
        return cls

    return register


MAX_SEPARATE = 32  # non-batchable envs: at most this many agents (one world each)


def env_key(name, settings=None):
    """the name of an env with its settings, e.g. "WaterCannon · balls 2" (runs and
    checkpoints are kept per key); just the name for an env without settings"""
    values = ENVS[name].chosen(settings)
    return " · ".join([name, *(f"{k} {v}" for k, v in values.items())])


def make(name, n=1, **kwargs):
    """a new env with n agents (an env with batchable = False: n separate worlds);
    kwargs: decorated, the env's settings"""
    if name not in ENVS:
        import webgym.envs  # noqa: F401 (registers the built-in envs)
    cls = ENVS[name]
    if not cls.batchable and n > 1:
        return Separate(cls, min(n, MAX_SEPARATE), **kwargs)
    return cls(n=n, **kwargs)


class Separate:
    """n one-agent envs, each in its own world, with the batched API of one env with n
    agents: for envs whose agents change the world (batchable = False, e.g. an agent
    pushing boxes around). Slower (n world steps per step); for training (never drawn)"""

    def __init__(self, cls, n, **kwargs):
        self.envs = [cls(n=1, **kwargs) for _ in range(n)]
        self.n = n

    def __getattr__(self, name):  # (the env's settings: names, dt, view, ...)
        return getattr(self.envs[0], name)

    @property
    def world(self):
        return self.envs[0].world

    @property
    def done(self):
        return np.concatenate([e.done for e in self.envs])

    @property
    def episode_return(self):
        return np.concatenate([e.episode_return for e in self.envs])

    @property
    def freeze_done(self):
        return self.envs[0].freeze_done

    @freeze_done.setter
    def freeze_done(self, value):
        for e in self.envs:
            e.freeze_done = value

    def reset(self, seed=None):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        seeds = self.rng.integers(2**31, size=self.n)
        return np.concatenate([e.reset(int(s)) for s, e in zip(seeds, self.envs)])

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)
        steps = [e.step(actions[i : i + 1]) for i, e in enumerate(self.envs)]
        return tuple(np.concatenate(values) for values in zip(*steps))

    def decorate(self):
        for e in self.envs:
            e.decorate()

    def camera_target(self, i):
        return self.envs[i].camera_target(0)

    def close(self):
        for e in self.envs:
            e.close()
