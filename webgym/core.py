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
PARTICLE_COLOR = 0xFFA040
SCENERY_COLOR = 0x4E8A68  # custom debug draw colors (the view keeps these opaque)
PROP_COLOR = 0xB08850
LEADER_COLOR = 0xFF8C42
MARK_COLOR = 0xA8B8B0  # (marks on the scenery, e.g. the walkers' ground ticks)


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
    gravity = (0, -10)
    # initial camera: world point at the view center, half height in world units
    view_center = (0.0, 0.0)
    view_zoom = 10.0
    # True: n agents share one world without affecting each other (only static scenery);
    # False: the agents change the world (e.g. a dynamic parcours): one agent at a time
    # (the population / show-policies sliders are disabled)
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
        self._shapes = [[] for _ in range(n)]  # the shapes of each agent (leader color)
        self.build_scenery()
        parts = {}
        for i in range(n):
            self._agent = i
            for name, part in self.build_agent(i).items():
                parts.setdefault(name, []).extend(part if isinstance(part, (list, tuple)) else [part])
        for name, items in parts.items():
            setattr(self, name, _batch(items))
        self._bodies = b2d.Bodies(self._bodies)
        self._owner = np.array(self._owner)
        self._shapes = [b2d.Shapes(s) for s in self._shapes]
        self._start = self.world.get_snapshot()
        self.steps = 0
        self.done = np.zeros(n, dtype=bool)  # the agent's episode ended
        self.episode_return = np.zeros(n)
        # finished agents are disabled (not simulated, not drawn) until the next reset;
        # False: they go on (watch mode without auto reset)
        self.freeze_done = True
        # True: every reset gives all agents the same (random) start (the watched env: a
        # fair comparison of the policies you see); training: each agent its own
        self.same_start = False
        self._leader = None

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
        debug_draw.draw_segment(p1, p2, 0xRRGGBB); agent 0 is the current policy, the
        others are ghosts; default: nothing"""

    # -- building blocks for build_scenery / build_agent -----------------------------

    def scenery(self, geometry, position=(0, 0), solid=True, **material):
        """a static body with one shape (ground, walls, ...) -> the body; solid=False:
        decoration only, drawn but nothing collides with it (e.g. the lander's flags)"""
        body = self.world.create_static_body(position=position)
        shape_def = b2d.shape_def(
            filter=SCENERY_FILTER if solid else DECORATION_FILTER,
            material=b2d.surface_material(custom_color=SCENERY_COLOR, **material),
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

    def body(self, position, geometry, density=1.0, friction=0.8, filter=AGENT_FILTER, material=None,
             **body_settings):
        """a dynamic body of the agent being built, with one shape (more: self.shape);
        `filter`: what it collides with (default: the scenery, not other agents);
        `material`: more of its surface ({"rolling_resistance": 0.2}). Agent bodies never
        sleep: a joint motor doesn't wake a sleeping body, and an agent lying still
        (fallen, auto reset off) must still move when its policy acts"""
        body = self.world.create_dynamic_body(position=position, **{"enable_sleep": False, **body_settings})
        self._bodies.append(body)
        self._owner.append(self._agent)
        self.shape(body, geometry, density, friction, filter, **(material or {}))
        return body

    def shape(self, body, geometry, density=1.0, friction=0.8, filter=AGENT_FILTER, **material):
        """one more shape on an agent body (`material`: more of its surface, e.g.
        rolling_resistance)"""
        shape_def = b2d.shape_def(density=density, filter=filter,
                                  material=b2d.surface_material(friction=friction, **material))
        shape = body.create_shape(shape_def, geometry)
        self._shapes[self._agent].append(shape)
        return shape

    def sensor(self, body, geometry, filter=SENSOR_FILTER, density=1.0):
        """a sensor shape on an agent body: `sensor_capacity > 0` (batched: b2d.Shapes)
        while it overlaps the scenery, e.g. a foot touching the ground. (It has mass, as
        any shape: `density=0` for none; the older envs were trained with 1)"""
        shape_def = b2d.shape_def(is_sensor=True, enable_sensor_events=True, filter=filter, density=density)
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
        self._leader = None
        self.randomize(_SameForAll(self.rng, self.n) if self.same_start and self.n > 1 else self.rng)
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

    def set_leader(self, i):
        """draw agent i (None: none) in LEADER_COLOR"""
        if i == self._leader:
            return
        if self._leader is not None:
            self._shapes[self._leader].custom_color = 0
        if i is not None:
            self._shapes[i].custom_color = LEADER_COLOR
        self._leader = i

    @property
    def follows(self):
        """does this env suggest a camera target (overrides camera_target)?"""
        return type(self).camera_target is not Env.camera_target

    def close(self):
        """free the box2d world now (box2d has at most b2d.MAX_WORLDS = 128 worlds at a
        time; don't wait for the garbage collector)"""
        self.world.destroy()


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


class _SameForAll:
    """an rng for `randomize` whose draws for the n agents are one draw, repeated: e.g.
    `rng.uniform(-3, 3, n)` -> n times the same value (a draw for the agents: its size,
    the last positional argument or `size=`, is n or (n, ...); other draws as they are)"""

    def __init__(self, rng, n):
        self.rng, self.n = rng, n

    def __getattr__(self, name):
        draw = getattr(self.rng, name)

        def same(*args, size=None, **kwargs):
            if size is None and args and np.shape(args[-1]) in ((), (2,)) and _first(args[-1]) == self.n:
                *args, size = args
            if size is None:  # (also draws without a size, e.g. permutation)
                return draw(*args, **kwargs)
            if _first(size) != self.n:
                return draw(*args, size=size, **kwargs)
            shape = (size,) if np.isscalar(size) else tuple(size)
            return np.repeat(draw(*args, size=(1, *shape[1:]), **kwargs), shape[0], axis=0)

        return same


def _first(size):
    """n of a size n or (n, ...)"""
    return size if np.isscalar(size) else tuple(size)[0]


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
    pushing boxes around). Slower (n world steps per step); drawn: the leader's world"""

    def __init__(self, cls, n, **kwargs):
        self.envs = [cls(n=1, **kwargs) for _ in range(n)]
        self.n = n
        self.same_start = False
        self._leader = None

    def __getattr__(self, name):  # (the env's settings: names, dt, view, ...)
        return getattr(self.envs[0], name)

    @property
    def world(self):
        """the world to draw: the leader's"""
        return self.envs[self._leader or 0].world

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
        seeds = self.rng.integers(2**31, size=1 if self.same_start else self.n)
        return np.concatenate([e.reset(int(seeds[i % len(seeds)])) for i, e in enumerate(self.envs)])

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)
        steps = [e.step(actions[i : i + 1]) for i, e in enumerate(self.envs)]
        return tuple(np.concatenate(values) for values in zip(*steps))

    def set_leader(self, i):
        if self._leader is not None:
            self.envs[self._leader].set_leader(None)
        if i is not None:
            self.envs[i].set_leader(0)
        self._leader = i

    def decorate(self):
        for e in self.envs:
            e.decorate()

    def camera_target(self, i):
        return self.envs[i].camera_target(0)

    def close(self):
        for e in self.envs:
            e.close()
