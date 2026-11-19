"""DoubleCartPole: keep two poles, one on top of the other, upright by pushing the cart.

As CartPole, with a second, shorter pole on a hinge at the top of the first (a double
inverted pendulum on a cart): much harder, the upper pole falls one way while the lower
one is caught.
obs: cart x, cart velocity, per pole its angle (0: upright) and angular velocity.
action: the force on the cart. reward: 1 per step while both poles are up.
The episode ends when a pole tilts more than 20 degrees or the cart leaves |x| < 2.4.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Env, register_env

MAX_FORCE = 20.0  # N; the cart weighs 1 kg
POLES = [(1.0, 0.1), (0.5, 0.05)]  # (length m, mass kg), from the bottom


@register_env("DoubleCartPole")
class DoubleCartPole(Env):
    obs_names = ("cart x", "cart velocity", "lower angle", "lower ang. vel.", "upper angle", "upper ang. vel.")
    action_names = ("force",)
    discrete_actions = np.array([[-1.0], [-0.3], [0.3], [1.0]])  # (for DQN: push left / right, hard / soft)
    discrete_names = ("push left", "nudge left", "nudge right", "push right")
    max_steps = 500
    view_center, view_zoom = (0.0, 1.2), 3.0
    suggested = {"quick": {"cem": {}, "neat": {}},
                 "medium": {"cem": {}, "ppo_torch": {}, "neat": {}},
                 "long": {"ppo_torch": {}, "cem": {}}}

    def build_scenery(self):
        self.ground = self.scenery(b2d.box(3.0, 0.05))

    def build_agent(self, i):
        cart = self.body((0, 0.3), b2d.box(0.25, 0.12), density=1 / 0.12)  # (1 kg)
        # the rail, with stops at its ends (the episode ends before, at |x| = 2.4)
        self.world.create_prismatic_joint(
            body_a=self.ground, body_b=cart,
            local_frame_a=b2d.transform(p=(0, 0.3)), local_frame_b=b2d.transform(p=(0, 0)),
            enable_limit=True, lower_translation=-2.75, upper_translation=2.75,
        )
        # the poles: each on a hinge at the top of what is below it
        poles, below, top, y = [], cart, 0.12, 0.42
        for length, mass in POLES:
            pole = self.body((0, y + length / 2), b2d.box(0.025, length / 2), density=mass / (0.05 * length))
            self.world.create_revolute_joint(
                body_a=below, body_b=pole,
                local_frame_a=b2d.transform(p=(0, top)), local_frame_b=b2d.transform(p=(0, -length / 2)),
            )
            poles.append(pole)
            below, top, y = pole, length / 2, y + length
        return {"cart": cart, "lower": poles[0], "upper": poles[1]}

    def randomize(self, rng):
        self.cart.linear_velocity = np.column_stack([rng.uniform(-0.1, 0.1, self.n), np.zeros(self.n)])
        self.lower.angular_velocity = rng.uniform(-0.1, 0.1, self.n)
        self.upper.angular_velocity = rng.uniform(-0.1, 0.1, self.n)

    def act(self, actions):
        force = np.column_stack([actions[:, 0] * MAX_FORCE, np.zeros(self.n)])
        self.cart.apply_force_to_center(force, True)

    def observe(self):
        return [self.cart.position[:, 0], self.cart.linear_velocity[:, 0],
                self.lower.angle, self.lower.angular_velocity, self.upper.angle, self.upper.angular_velocity]

    def reward(self):
        fallen = ((np.abs(self.lower.angle) > 0.35) | (np.abs(self.upper.angle) > 0.35)
                  | (np.abs(self.cart.position[:, 0]) > 2.4))
        return np.ones(self.n), fallen
