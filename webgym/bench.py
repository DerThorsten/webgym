"""Throughput benchmark: CartPole population steps, separate worlds vs one world
(n carts, batch API). Runs natively and in a web worker (web/bench.html).

    python -m webgym.bench
"""

import json
import time

import numpy as np
import pyb2d3 as b2d

from webgym.core import make
from webgym.policies import MLP


def separate_worlds(n, steps):
    """n CartPole envs (own worlds), stepped one by one -> env steps per second"""
    envs = [make("CartPole") for _ in range(n)]
    obs = np.stack([env.reset(seed=i)[0] for i, env in enumerate(envs)])
    policy = MLP(4, 1)
    params = np.random.default_rng(0).standard_normal((n, policy.num_params)) * 0.1
    start = time.perf_counter()
    for _ in range(steps):
        actions = policy(params, obs)
        for i, env in enumerate(envs):
            obs[i], _, terminated, truncated, _ = env.step(actions[i])
            if terminated or truncated:
                obs[i] = env.reset()[0]
    seconds = time.perf_counter() - start
    for env in envs:
        env.close()
    return n * steps / seconds


def one_world(n, steps):
    """n carts in one world (no collisions between them), forces and observations
    with the batch API -> env steps per second (no resets: dynamics cost only)"""
    world = b2d.World(gravity=(0, -10))
    ground = world.create_static_body(position=(0, 0))
    no_contact = b2d.make_filter(category_bits=2, mask_bits=1)
    carts, poles = [], []
    for _ in range(n):
        cart = world.create_dynamic_body(position=(0, 0.3))
        cart.create_shape(b2d.shape_def(density=1 / 0.12, filter=no_contact), b2d.box(0.25, 0.12))
        pole = world.create_dynamic_body(position=(0, 0.92))
        pole.create_shape(b2d.shape_def(density=2.0, filter=no_contact), b2d.box(0.025, 0.5))
        world.create_prismatic_joint(
            body_a=ground, body_b=cart, local_frame_a=b2d.transform(p=(0, 0.3)),
            local_frame_b=b2d.transform(p=(0, 0)), enable_limit=True,
            lower_translation=-2.75, upper_translation=2.75,
        )
        world.create_revolute_joint(
            body_a=cart, body_b=pole, local_frame_a=b2d.transform(p=(0, 0.12)),
            local_frame_b=b2d.transform(p=(0, -0.5)),
        )
        carts.append(cart)
        poles.append(pole)
    carts, poles = b2d.Bodies(carts), b2d.Bodies(poles)
    policy = MLP(4, 1)
    params = np.random.default_rng(0).standard_normal((n, policy.num_params)) * 0.1
    obs = np.zeros((n, 4), np.float32)
    forces = np.zeros((n, 2), np.float32)
    start = time.perf_counter()
    for _ in range(steps):
        forces[:, 0] = policy(params, obs)[:, 0] * 10
        carts.apply_force_to_center(forces, True)
        world.step(1 / 60, 4)
        obs[:, 0] = carts.position[:, 0]
        obs[:, 1] = carts.linear_velocity[:, 0]
        obs[:, 2] = poles.angle
        obs[:, 3] = poles.angular_velocity
    seconds = time.perf_counter() - start
    world.destroy()
    return n * steps / seconds


def run(n=64, steps=300, kinds=("separate", "one_world")):
    """{kind: env steps per second} as JSON (for the web worker)"""
    funcs = {"separate": separate_worlds, "one_world": one_world}
    return json.dumps({k: round(funcs[k](n, steps)) for k in kinds})


if __name__ == "__main__":
    print(run())
