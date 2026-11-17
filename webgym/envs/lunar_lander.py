"""LunarLander: land softly between the flags.

The lander starts high up with a random push. Main engine (thrust along the
lander's up axis) and attitude thrusters (torque).
obs: position, velocity, angle, angular velocity, leg contacts (8 values).
action: [main engine (<= 0: off, 1: full), attitude (-1 .. 1)].
reward: getting closer to the pad, slower and upright, legs down; fuel costs;
+100 for coming to rest on the legs, -100 for touching the ground with the hull
or flying away.

Keys (human controller): up = main engine, left / right = rotate.
"""

import math

import numpy as np
import pyb2d3 as b2d

from webgym.core import Box, Env, register_env

HULL = np.array([(-0.5, -0.25), (0.5, -0.25), (0.35, 0.35), (-0.35, 0.35)])
LEG_ANGLE = 0.35  # the legs splay outwards
LEG_CENTERS = [(-0.55, -0.45), (0.55, -0.45)]
LEG_HALF_LENGTH = 0.3
# the lower ends of the legs (local points)
LEG_TIPS = np.array(
    [(c[0] + LEG_HALF_LENGTH * math.sin(a), c[1] - LEG_HALF_LENGTH * math.cos(a))
     for c, a in zip(LEG_CENTERS, (-LEG_ANGLE, LEG_ANGLE))]
)


def world_points(position, angle, local):
    """(n, k, 2) world points of `local` (k, 2) points on n bodies"""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    x = position[:, None, 0] + c * local[None, :, 0] - s * local[None, :, 1]
    y = position[:, None, 1] + s * local[None, :, 0] + c * local[None, :, 1]
    return np.stack([x, y], axis=-1)


@register_env("LunarLander")
class LunarLander(Env):
    observation_space = Box(-np.inf, np.inf, (8,))
    action_space = Box(-1.0, 1.0, (2,))
    max_steps = 600
    view_center = (0.0, 5.0)
    view_zoom = 6.0
    gravity = (0, -5.0)
    keys = "↑ main engine · ← / → rotate"
    # nothing / main engine / rotate left / rotate right (as the classic discrete lander)
    discrete_actions = np.array([[0, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float32)

    def build(self):
        ground = self.world.create_static_body(position=(0, -0.5))
        self.scenery_shape(ground, b2d.box(14, 0.5), friction=0.8)
        for x in (-1.6, 1.6):  # the flags of the pad (decoration)
            flag = self.world.create_static_body(position=(x, 0))
            self.scenery_shape(flag, b2d.box(0.02, 0.6, center=(0, 0.6)))
            self.scenery_shape(flag, b2d.polygon([(0, 1.2), (0, 0.9), (0.35, 1.05)]))
        shapes = [(b2d.polygon(HULL.tolist()), 1.0)] + [
            (b2d.box(0.05, LEG_HALF_LENGTH, center=c, rotation=a), 0.5)
            for c, a in zip(LEG_CENTERS, (-LEG_ANGLE, LEG_ANGLE))
        ]
        landers = [self.agent_body(i, (0, 9), shapes) for i in range(self.n)]
        self.landers = b2d.Bodies(landers)
        mass, inertia = landers[0].mass, landers[0].rotational_inertia
        self.main_force = 2.2 * mass * -self.gravity[1]
        self.max_torque = 3.0 * inertia
        self.rest_height = -LEG_TIPS[:, 1].min()  # center above ground when standing
        self.forces = np.zeros((self.n, 2), dtype=np.float32)

    def _reset(self, rng):
        n = self.n
        self.landers.position = np.column_stack([rng.uniform(-3, 3, n), np.full(n, 9.0)]).astype(np.float32)
        self.landers.angle = rng.uniform(-0.2, 0.2, n).astype(np.float32)
        self.landers.linear_velocity = np.column_stack(
            [rng.uniform(-1.5, 1.5, n), rng.uniform(-1.0, 0.0, n)]
        ).astype(np.float32)
        self.landers.angular_velocity = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        self.main = np.zeros(n)
        self.side = np.zeros(n)
        self.shaping = self._shaping(self._observe())

    def _apply(self, actions):
        a = np.clip(actions, -1, 1)
        self.main, self.side = np.maximum(a[:, 0], 0.0), a[:, 1]
        angle = self.landers.angle
        thrust = self.main * self.main_force
        self.forces[:, 0] = -np.sin(angle) * thrust  # along the lander's up axis
        self.forces[:, 1] = np.cos(angle) * thrust
        self.landers.apply_force_to_center(self.forces, True)
        self.landers.apply_torque((self.side * self.max_torque).astype(np.float32), True)

    def _observe(self):
        pos, vel, angle = self.landers.position, self.landers.linear_velocity, self.landers.angle
        legs = (world_points(pos, angle, LEG_TIPS)[:, :, 1] < 0.03).astype(np.float32)
        return np.column_stack(
            [pos[:, 0] / 5, (pos[:, 1] - self.rest_height) / 5, vel / 5, angle,
             self.landers.angular_velocity / 2, legs]
        ).astype(np.float32)

    @staticmethod
    def _shaping(o):
        """potential: high near the pad, slow, upright, legs down"""
        return (
            -100 * np.hypot(o[:, 0], o[:, 1]) - 100 * np.hypot(o[:, 2], o[:, 3])
            - 100 * np.abs(o[:, 4]) + 10 * (o[:, 6] + o[:, 7])
        )

    def _outcome(self):
        o = self._observe()
        shaping = self._shaping(o)
        reward = shaping - self.shaping - 0.3 * self.main - 0.03 * np.abs(self.side)
        self.shaping = shaping
        pos, angle = self.landers.position, self.landers.angle
        hull_down = world_points(pos, angle, HULL)[:, :, 1].min(axis=1) < 0.02
        crashed = hull_down | (np.abs(pos[:, 0]) > 9) | (pos[:, 1] > 16)
        speed = np.hypot(*self.landers.linear_velocity.T)
        resting = (o[:, 6] > 0) & (o[:, 7] > 0) & (speed < 0.1) & (np.abs(self.landers.angular_velocity) < 0.2)
        reward = reward - 100 * crashed + 100 * (resting & ~crashed)
        return reward, crashed | resting

    def human_action(self, keys):
        """up: main engine, left / right: rotate"""
        return np.array([float("up" in keys), ("left" in keys) - ("right" in keys)], dtype=np.float32)
