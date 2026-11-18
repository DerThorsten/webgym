"""CartPole: keep a pole upright by pushing its cart left / right.

A cart on a rail (a prismatic joint) with a pole on a hinge (a revolute joint).
obs: cart x, cart velocity, pole angle, pole angular velocity.
action: the force on the cart. reward: 1 per step while the pole is up.
The episode ends when the pole tilts more than 12 degrees or the cart leaves |x| < 2.4.

Keys (human controller): left / right arrow push the cart.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Env, register_env

MAX_FORCE = 10.0  # N; the cart weighs 1 kg, the pole 0.1 kg (as the classic cart-pole)


@register_env("CartPole")
class CartPole(Env):
    obs_names = ("cart x", "cart velocity", "pole angle", "pole ang. vel.")
    action_names = ("force",)
    discrete_actions = np.array([[-1.0], [1.0]])  # (for DQN: push left / right)
    discrete_names = ("push left", "push right")
    max_steps = 500
    view_center, view_zoom = (0.0, 1.0), 3.0
    keys = "← / → push the cart"

    def build_scenery(self):
        self.ground = self.scenery(b2d.box(3.0, 0.05))

    def build_agent(self, i):
        cart = self.body((0, 0.3), b2d.box(0.25, 0.12), density=1 / 0.12)  # (1 kg)
        pole = self.body((0, 0.92), b2d.box(0.025, 0.5), density=2.0)  # (0.1 kg)
        # the rail, with stops at its ends (the episode ends before, at |x| = 2.4)
        self.world.create_prismatic_joint(
            body_a=self.ground, body_b=cart,
            local_frame_a=b2d.transform(p=(0, 0.3)), local_frame_b=b2d.transform(p=(0, 0)),
            enable_limit=True, lower_translation=-2.75, upper_translation=2.75,
        )
        # the hinge: the pole's lower end on the top of the cart
        self.world.create_revolute_joint(
            body_a=cart, body_b=pole,
            local_frame_a=b2d.transform(p=(0, 0.12)), local_frame_b=b2d.transform(p=(0, -0.5)),
        )
        return {"cart": cart, "pole": pole}

    def randomize(self, rng):
        self.cart.linear_velocity = np.column_stack([rng.uniform(-0.1, 0.1, self.n), np.zeros(self.n)])
        self.pole.angular_velocity = rng.uniform(-0.1, 0.1, self.n)

    def act(self, actions):
        force = np.column_stack([actions[:, 0] * MAX_FORCE, np.zeros(self.n)])
        self.cart.apply_force_to_center(force, True)

    def observe(self):
        return [self.cart.position[:, 0], self.cart.linear_velocity[:, 0], self.pole.angle,
                self.pole.angular_velocity]

    def reward(self):
        fallen = (np.abs(self.pole.angle) > 0.21) | (np.abs(self.cart.position[:, 0]) > 2.4)
        return np.ones(self.n), fallen

    def human_action(self, keys):
        return np.array([("right" in keys) - ("left" in keys)], dtype=np.float32)
