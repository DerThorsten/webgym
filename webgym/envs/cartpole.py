"""CartPole: keep a pole upright by pushing its cart left / right.

A cart on a prismatic joint (slides along x) with a pole on a revolute joint.
obs: cart x, cart velocity, pole angle, pole angular velocity.
action: force in [-1, 1] (x max_force). reward: 1 per step while the pole is up.
Episode ends when the pole tilts > 12 degrees or the cart leaves |x| < 2.4.

Keys (human controller): left / right arrow push the cart.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Box, Env, register_env


@register_env("CartPole")
class CartPole(Env):
    observation_space = Box(-np.inf, np.inf, (4,))
    action_space = Box(-1.0, 1.0, (1,))
    max_steps = 500
    max_force = 10.0  # N; cart 1 kg, pole 0.1 kg (as the classic cart-pole)
    view_center = (0.0, 1.0)
    view_zoom = 3.0
    keys = "← / → push the cart"

    def build(self):
        ground = self.world.create_static_body(position=(0, 0))
        self.scenery_shape(ground, b2d.box(3.0, 0.05))
        carts, poles = [], []
        for i in range(self.n):
            # densities for 1 kg (0.5 x 0.24 box) and 0.1 kg (0.05 x 1 box)
            cart = self.agent_body(i, (0, 0.3), [(b2d.box(0.25, 0.12), 1 / 0.12)])
            pole = self.agent_body(i, (0, 0.92), [(b2d.box(0.025, 0.5), 2.0)])
            # the rail: stops at its ends (episodes end before, at |x| = 2.4)
            self.world.create_prismatic_joint(
                body_a=ground,
                body_b=cart,
                local_frame_a=b2d.transform(p=(0, 0.3)),
                local_frame_b=b2d.transform(p=(0, 0)),
                enable_limit=True,
                lower_translation=-2.75,
                upper_translation=2.75,
            )
            self.world.create_revolute_joint(
                body_a=cart,
                body_b=pole,
                local_frame_a=b2d.transform(p=(0, 0.12)),
                local_frame_b=b2d.transform(p=(0, -0.5)),
            )
            carts.append(cart)
            poles.append(pole)
        self.carts, self.poles = b2d.Bodies(carts), b2d.Bodies(poles)
        self.forces = np.zeros((self.n, 2), dtype=np.float32)

    def _reset(self, rng):
        velocity = np.zeros((self.n, 2), dtype=np.float32)
        velocity[:, 0] = rng.uniform(-0.1, 0.1, self.n)
        self.carts.linear_velocity = velocity
        self.poles.angular_velocity = rng.uniform(-0.1, 0.1, self.n).astype(np.float32)

    def _apply(self, actions):
        self.forces[:, 0] = np.clip(actions[:, 0], -1, 1) * self.max_force
        self.carts.apply_force_to_center(self.forces, True)

    def _observe(self):
        return np.column_stack(
            [self.carts.position[:, 0], self.carts.linear_velocity[:, 0], self.poles.angle,
             self.poles.angular_velocity]
        ).astype(np.float32)

    def _outcome(self):
        failed = (np.abs(self.poles.angle) > 0.21) | (np.abs(self.carts.position[:, 0]) > 2.4)
        return np.ones(self.n), failed

    def focus(self, i=0):
        return None

    def human_action(self, keys):
        """left / right arrow: full force"""
        return np.array([("right" in keys) - ("left" in keys)], dtype=np.float32)
