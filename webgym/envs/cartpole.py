"""CartPole: keep a pole upright by pushing its cart left / right.

A cart on a prismatic joint (slides along x) with a pole on a revolute joint.
obs: cart x, cart velocity, pole angle, pole angular velocity.
action: force in [-1, 1] (x max_force). reward: 1 per step while the pole is up.
Episode ends when the pole tilts > 12 degrees or the cart leaves |x| < 2.4.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Box, Env, register_env


@register_env("CartPole")
class CartPole(Env):
    observation_space = Box(-np.inf, np.inf, (4,))
    action_space = Box(-1.0, 1.0, (1,))
    max_steps = 500
    max_force = 20.0
    view_center = (0.0, 1.0)
    view_zoom = 3.0

    def __init__(self):
        world = b2d.World(gravity=(0, -10))
        ground = world.create_static_body(position=(0, 0))
        # cart and pole don't collide with the ground or each other (the joints hold
        # them): ground category 4, cart / pole category 2 with mask 1 = only the
        # default category of queries (box2d checks the filters both ways, so the
        # pointer's query still finds them for dragging)
        ground.create_shape(
            b2d.shape_def(filter=b2d.make_filter(category_bits=4)), b2d.box(3.0, 0.05)
        )
        no_contact = b2d.make_filter(category_bits=2, mask_bits=1)
        self.cart = world.create_dynamic_body(position=(0, 0.3))
        self.cart.create_shape(b2d.shape_def(density=1.0, filter=no_contact), b2d.box(0.25, 0.12))
        self.pole = world.create_dynamic_body(position=(0, 0.92))
        self.pole.create_shape(b2d.shape_def(density=0.5, filter=no_contact), b2d.box(0.025, 0.5))
        world.create_prismatic_joint(
            body_a=ground,
            body_b=self.cart,
            local_frame_a=b2d.transform(p=(0, 0.3)),
            local_frame_b=b2d.transform(p=(0, 0)),
        )
        world.create_revolute_joint(
            body_a=self.cart,
            body_b=self.pole,
            local_frame_a=b2d.transform(p=(0, 0.12)),
            local_frame_b=b2d.transform(p=(0, -0.5)),
        )
        self.world = world
        self.start = world.get_snapshot()

    def _reset(self, rng):
        self.world.restore_from_snapshot(self.start)
        vx, w = rng.uniform(-0.1, 0.1, 2)
        self.cart.linear_velocity = (vx, 0)
        self.pole.angular_velocity = w

    def _apply(self, action):
        force = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1)) * self.max_force
        self.cart.apply_force_to_center((force, 0), True)

    def _observe(self):
        return np.array(
            [self.cart.position[0], self.cart.linear_velocity[0], self.pole.angle, self.pole.angular_velocity],
            dtype=np.float32,
        )

    def _outcome(self):
        failed = abs(self.pole.angle) > 0.21 or abs(self.cart.position[0]) > 2.4
        return 1.0, failed
