"""Every env: batched API shapes, random actions, determinism, agents independent,
finished agents frozen."""

import numpy as np
import pytest

from webgym.core import ENVS, make

import webgym.envs  # noqa: F401 (registers the envs)

NAMES = sorted(ENVS)


@pytest.mark.parametrize("name", NAMES)
def test_random_actions(name):
    """reset / step shapes and types over a few episodes of random actions (8 agents)"""
    env = make(name, n=8)
    rng = np.random.default_rng(0)
    obs = env.reset(seed=0)
    assert obs.shape == (8, *env.observation_space.shape) and obs.dtype == np.float32
    for _ in range(300):
        obs, reward, terminated, truncated = env.step(env.action_space.sample(rng, 8))
        assert obs.shape == (8, *env.observation_space.shape) and np.isfinite(obs).all()
        assert reward.shape == terminated.shape == truncated.shape == (8,)
        if env.done.all():
            obs = env.reset()
    env.close()


@pytest.mark.parametrize("name", NAMES)
def test_seed_determinism(name):
    """same seed + actions -> same observations (snapshot reset restores everything)"""

    def run(env):
        env.reset(seed=3)
        return np.stack([env.step(np.full((env.n, *env.action_space.shape), 0.3))[0] for _ in range(30)])

    env = make(name, n=4)
    first = run(env)
    assert np.array_equal(first, run(env))  # after a reset of the same env
    assert np.array_equal(first, run(make(name, n=4)))  # a fresh env


@pytest.mark.parametrize("name", NAMES)
def test_agents_are_independent(name):
    """the agents of one world don't interact: other agents' actions don't change what
    happens to agent 0"""
    env = make(name, n=5)

    def run(others):
        env.reset(seed=7)
        out = []
        for _ in range(60):
            actions = np.zeros((5, *env.action_space.shape), dtype=np.float32)
            actions[1:] = others.uniform(-1, 1, (4, *env.action_space.shape))
            out.append(env.step(actions)[0][0])
        return np.stack(out)

    assert np.allclose(run(np.random.default_rng(1)), run(np.random.default_rng(99)), atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_discrete_actions(name):
    """every env also works with a few discrete actions (DQN): valid actions + names"""
    env = make(name)
    actions = np.asarray(env.discrete_actions)
    assert actions.shape[1:] == env.action_space.shape and len(env.discrete_names) == len(actions)
    assert np.abs(actions).max() <= 1
    env.reset(seed=0)
    env.step(actions[-1:])


def test_water_cannon_one_gun_per_object():
    """with N objects the cart has N cannons, each with its own angle, pressure and spread:
    1 + 3N actions; only the cannon that is given pressure sprays"""
    env = make("WaterCannon", n=2, objects=3)
    assert env.action_space.shape == (10,) and env.observation_space.shape == (6 + 6 + 1 + 9,)
    assert len(env.barrel) == len(env.hinge) == 6 and len(env.object) == 6
    env.reset(seed=0)
    actions = np.zeros((2, 10))
    actions[:, 3::3] = -1  # (narrow jets)
    actions[0, 5] = 1.0  # agent 0: cannon 2 (of 3) at full pressure
    for _ in range(20):
        env.step(actions)
    drops = [len(c) for c in env.cannon]
    assert drops[1] > 0 and drops[0] == drops[2] == 0 and sum(drops[3:]) == 0


def test_water_hits_only_its_own_object():
    """water cannons in one world: each cart's water carries its own object, not the
    others' (collision groups); an object without water falls as if alone"""
    env = make("WaterCannon", n=2)
    env.same_start = True  # (both objects start alike)
    env.reset(seed=0)
    for _ in range(50):  # agent 0 sprays straight up, agent 1 does nothing
        env.step(np.array([[0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, -1.0]]))
    alone = make("WaterCannon", n=1)
    alone.reset(seed=0)
    for _ in range(50):
        alone.step(np.array([[0.0, 0.0, 0.0, -1.0]]))
    y = env.object.position[:, 1]
    assert y[0] > y[1] + 0.5 and abs(y[1] - alone.object.position[0, 1]) < 1e-3


def test_agents_never_sleep():
    """a fallen agent lying still (watching without auto reset) must still react to its
    motors: box2d would put its bodies to sleep, and motors don't wake them"""
    env = make("Ragdoll", n=1)
    env.freeze_done = False
    env.reset(seed=0)
    for _ in range(300):  # (5 s limp: it falls and lies still)
        env.step(np.zeros((1, len(env.action_names))))
    before = env.hull.position[0].copy()
    for _ in range(60):
        env.step(np.ones((1, len(env.action_names))))
    assert np.linalg.norm(env.hull.position[0] - before) > 0.05


def test_finished_agents_are_frozen():
    """an agent whose episode ended gets no more reward and no new end"""
    env = make("CartPole", n=3)
    env.reset(seed=0)
    while not env.done.any():
        actions = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)  # agent 0 pushes: falls first
        env.step(actions)
    first = np.flatnonzero(env.done)
    returns = env.episode_return[first].copy()
    _, reward, terminated, truncated = env.step(np.zeros((3, 1), np.float32))
    assert (reward[first] == 0).all() and not terminated[first].any() and not truncated[first].any()
    assert np.array_equal(env.episode_return[first], returns)


def test_untrained_ragdoll_falls():
    """without a policy (zero actions: limp motors) every ragdoll collapses within 3 s:
    the episode ends with the fall penalty"""
    env = make("Ragdoll", n=4)
    env.reset(seed=0)
    for _ in range(180):
        env.step(np.zeros((4, len(env.action_names)), np.float32))
    assert env.done.all() and (env.episode_return < -90).all()


def test_lander_flies_through_the_flags():
    """the flags of the pad are decoration (as in the classic lander): a lander flying
    sideways at flag height passes the pole at x = 1.6"""
    env = make("LunarLander", n=1, decorated=True)
    env.reset(seed=0)
    env.lander.position = np.array([[0.8, 1.6]], np.float32)
    env.lander.angle = np.zeros(1, np.float32)
    env.lander.angular_velocity = np.zeros(1, np.float32)
    env.lander.linear_velocity = np.array([[4.0, 2.0]], np.float32)
    for _ in range(25):
        env.step(np.zeros((1, 2), np.float32))
    assert env.lander.position[0, 0] > 2.0


def test_decoration_only_when_watched():
    """a decorated lander (watch mode) has flags and exhaust emitters: the engines that
    fire emit particles (main: 2 per step at full thrust; turning: 1 from a side
    thruster), agent 0's orange, the ghosts' in the ghost color; they expire again. An
    undecorated lander (training) has none of it"""
    from webgym.core import PARTICLE_COLOR

    plain = make("LunarLander", n=2)
    decorated = make("LunarLander", n=2, decorated=True)
    assert decorated.world.counters.body_count - plain.world.counters.body_count == 4  # (2 flags: pole + cloth)
    assert not hasattr(plain, "main_exhaust")
    decorated.reset(seed=0)
    for _ in range(10):  # lander 0: main engine; lander 1: rotate left
        decorated.step(np.array([[1.0, 0.0], [0.0, 1.0]], np.float32))
        decorated.decorate()
    main, right, left = decorated.main_exhaust, decorated.right_exhaust, decorated.left_exhaust
    assert len(main[0]) == 20 and len(right[1]) == 10 and len(main[1]) == len(left[1]) == 0
    assert (main[0].shapes.custom_color == PARTICLE_COLOR).all()
    assert (right[1].shapes.custom_color == 0).all()  # (the ghost color)
    for _ in range(40):
        decorated.step(np.zeros((2, 2), np.float32))
        decorated.decorate()
    assert sum(len(e) for e in main + right + left) == 0  # (all expired)


def test_decoration_switch():
    """'Decoration' off: the watched env is rebuilt without it; an app that draws
    nothing (the training worker) never decorates, whatever the checkbox says"""
    from webgym.app import App

    app = App(env="LunarLander")
    assert app.env.decorated
    app.command("decorate", False)
    assert not app.env.decorated and not hasattr(app.env, "main_exhaust")
    app.command("decorate", True)
    assert app.env.decorated
    trainer = App(env="LunarLander", draws=False)
    assert not trainer.env.decorated and trainer.show_deco  # (the checkbox: on)
