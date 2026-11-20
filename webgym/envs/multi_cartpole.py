"""MultiCartPole: keep a stack of poles upright by pushing the cart.

As CartPole, with 2 to 5 poles on hinges, each on top of the one below (a multiple
inverted pendulum on a cart), each shorter than the one below. Every pole falls its own
way, so each one more is much harder. Setting "poles": how many. With 4 or 5 a little
friction in the hinges between the poles keeps it learnable.
obs: cart x, cart velocity, per pole its angle (0: upright) and angular velocity.
action: the force on the cart. reward: 1 per step while all poles are up.
The episode ends when a pole tilts more than 20 degrees or the cart leaves |x| < 2.4.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import DARK_COLOR, Env, register_env

MAX_FORCE = 30.0  # N; the cart weighs 1 kg
# friction in the hinges between the poles, per number of poles (N m; a motor at speed 0:
# the poles bend less against each other, the stack falls slower). As little as works:
# without, 4 and 5 poles hardly learn (PPO, 6 min: 4 poles 478, 5 poles 61); with it
# (4: 0.0005, 5: 0.001) 485 and 330. (Angular damping of the poles barely helped.)
HINGE_FRICTION = {2: 0.0, 3: 0.0, 4: 0.0005, 5: 0.001}
POLES = [(1.0, 0.1), (0.6, 0.06), (0.4, 0.04), (0.3, 0.03), (0.25, 0.025)]  # (length m, mass kg), from the bottom


@register_env("MultiCartPole")
class MultiCartPole(Env):
    settings = {"poles": (2, (2, 3, 4, 5), "How many poles on top of each other (harder with each one).")}
    action_names = ("force",)
    discrete_actions = np.array([[-1.0], [-0.3], [0.3], [1.0]])  # (for DQN: push left / right, hard / soft)
    discrete_names = ("push left", "nudge left", "nudge right", "push right")
    max_steps = 500
    max_return = 500  # (1 per step)

    @property
    def suggested(self):
        """CEM is fast up to 4 poles; 5 need PPO"""
        if self.poles == 5:
            return {"quick": {"ppo_torch": {}}, "medium": {"ppo_torch": {}, "cem": {}},
                    "long": {"ppo_torch": {}, "cem": {}}}
        return {"quick": {"cem": {}}, "medium": {"cem": {}, "ppo_torch": {}}, "long": {"ppo_torch": {}, "cem": {}}}

    @property
    def obs_names(self):
        poles = [(f"pole {k + 1} angle", f"pole {k + 1} ang. vel.") for k in range(self.poles)]
        return ("cart x", "cart velocity", *(name for pair in poles for name in pair))

    @property
    def view_center(self):
        return (0.0, 1.0 + 0.15 * self.poles)  # (the whole stack in view)

    @property
    def view_zoom(self):
        return 2.4 + 0.35 * self.poles

    def build_scenery(self):
        self.ground = self.scenery(b2d.box(3.0, 0.05))

    def build_agent(self, i):
        cart = self.body((0, 0.3), b2d.box(0.25, 0.12), density=1 / 0.12, color=DARK_COLOR)  # (1 kg)
        # the rail, with stops at its ends (the episode ends before, at |x| = 2.4)
        self.world.create_prismatic_joint(
            body_a=self.ground, body_b=cart,
            local_frame_a=b2d.transform(p=(0, 0.3)), local_frame_b=b2d.transform(p=(0, 0)),
            enable_limit=True, lower_translation=-2.75, upper_translation=2.75,
        )
        # the poles: each on a hinge at the top of what is below it
        poles, below, top, y = [], cart, 0.12, 0.42
        for length, mass in POLES[: self.poles]:
            pole = self.body((0, y + length / 2), b2d.box(0.025, length / 2), density=mass / (0.05 * length))
            friction = HINGE_FRICTION[self.poles] if below is not cart else 0.0  # (the cart's hinge turns freely)
            self.world.create_revolute_joint(
                body_a=below, body_b=pole,
                local_frame_a=b2d.transform(p=(0, top)), local_frame_b=b2d.transform(p=(0, -length / 2)),
                enable_motor=friction > 0, motor_speed=0.0, max_motor_torque=friction,
            )
            poles.append(pole)
            below, top, y = pole, length / 2, y + length
        return {"cart": cart, "pole": poles}  # (self.pole: all agents' poles, `poles` per agent)

    def randomize(self, rng):
        self.cart.linear_velocity = np.column_stack([rng.uniform(-0.1, 0.1, self.n), np.zeros(self.n)])
        self.pole.angular_velocity = rng.uniform(-0.1, 0.1, (self.n, self.poles)).ravel()

    def act(self, actions):
        force = np.column_stack([actions[:, 0] * MAX_FORCE, np.zeros(self.n)])
        self.cart.apply_force_to_center(force, True)

    def observe(self):
        angle = self.pole.angle.reshape(self.n, self.poles)  # (a row per agent)
        speed = self.pole.angular_velocity.reshape(self.n, self.poles)
        return [self.cart.position[:, 0], self.cart.linear_velocity[:, 0],
                *(column for k in range(self.poles) for column in (angle[:, k], speed[:, k]))]

    def reward(self):
        tilted = np.abs(self.pole.angle.reshape(self.n, self.poles)).max(axis=1) > 0.35
        return np.ones(self.n), tilted | (np.abs(self.cart.position[:, 0]) > 2.4)
