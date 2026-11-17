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

HULL = [(-0.5, -0.25), (0.5, -0.25), (0.35, 0.35), (-0.35, 0.35)]
LEG_ANGLE = 0.35  # the legs splay outwards
LEG_CENTERS = [(-0.55, -0.45), (0.55, -0.45)]
LEG_HALF_LENGTH = 0.3


def _leg_tip(center, angle):
    """local point at the lower end of a leg (a box rotated by `angle`)"""
    return (
        center[0] + LEG_HALF_LENGTH * math.sin(angle),
        center[1] - LEG_HALF_LENGTH * math.cos(angle),
    )


@register_env("LunarLander")
class LunarLander(Env):
    observation_space = Box(-np.inf, np.inf, (8,))
    action_space = Box(-1.0, 1.0, (2,))
    max_steps = 600
    view_center = (0.0, 5.0)
    view_zoom = 6.0
    gravity = 5.0

    def __init__(self):
        world = b2d.World(gravity=(0, -self.gravity))
        ground = world.create_static_body(position=(0, -0.5))
        ground.create_shape(b2d.shape_def(material=b2d.surface_material(friction=0.8)), b2d.box(14, 0.5))
        # the flags of the pad: decoration, nothing collides with them
        no_contact = b2d.make_filter(category_bits=8, mask_bits=0)
        for x in (-1.6, 1.6):
            flag = world.create_static_body(position=(x, 0))
            flag.create_shape(b2d.shape_def(filter=no_contact), b2d.box(0.02, 0.6, center=(0, 0.6)))
            flag.create_shape(b2d.shape_def(filter=no_contact), b2d.polygon([(0, 1.2), (0, 0.9), (0.35, 1.05)]))

        self.lander = world.create_dynamic_body(position=(0, 9))
        material = b2d.surface_material(friction=0.8)
        self.lander.create_shape(b2d.shape_def(density=1.0, material=material), b2d.polygon(HULL))
        self.leg_tips = []
        for center, angle in zip(LEG_CENTERS, (-LEG_ANGLE, LEG_ANGLE)):
            self.lander.create_shape(
                b2d.shape_def(density=0.5, material=material),
                b2d.box(0.05, LEG_HALF_LENGTH, center=center, rotation=angle),
            )
            self.leg_tips.append(_leg_tip(center, angle))
        self.main_force = 2.2 * self.lander.mass * self.gravity
        self.max_torque = 3.0 * self.lander.rotational_inertia
        self.rest_height = -min(t[1] for t in self.leg_tips)  # center above ground when standing
        self.world = world
        self.start = world.get_snapshot()

    def _reset(self, rng):
        self.world.restore_from_snapshot(self.start)
        self.lander.set_transform((rng.uniform(-3, 3), 9.0), b2d.make_rot(rng.uniform(-0.2, 0.2)))
        self.lander.linear_velocity = (rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 0.0))
        self.lander.angular_velocity = rng.uniform(-0.5, 0.5)
        self.main = self.side = 0.0
        self.shaping = self._shaping()

    def _apply(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        self.main, self.side = max(a[0], 0.0), a[1]
        if self.main > 0:
            up = self.lander.world_vector((0, 1))
            self.lander.apply_force_to_center(
                (up[0] * self.main * self.main_force, up[1] * self.main * self.main_force), True
            )
        self.lander.apply_torque(self.side * self.max_torque, True)

    def _legs_down(self):
        """1.0 per leg whose tip touches the ground (flat at y = 0)"""
        return [float(self.lander.world_point(tip)[1] < 0.03) for tip in self.leg_tips]

    def _observe(self):
        (x, y), (vx, vy) = self.lander.position, self.lander.linear_velocity
        return np.array(
            [x / 5, (y - self.rest_height) / 5, vx / 5, vy / 5, self.lander.angle,
             self.lander.angular_velocity / 2, *self._legs_down()],
            dtype=np.float32,
        )

    def _shaping(self):
        """potential: high near the pad, slow, upright, legs down"""
        o = self._observe()
        return (
            -100 * math.hypot(o[0], o[1]) - 100 * math.hypot(o[2], o[3]) - 100 * abs(o[4])
            + 10 * (o[6] + o[7])
        )

    def _outcome(self):
        shaping = self._shaping()
        reward = shaping - self.shaping - 0.3 * self.main - 0.03 * abs(self.side)
        self.shaping = shaping
        (x, y), (vx, vy) = self.lander.position, self.lander.linear_velocity
        hull_down = min(self.lander.world_point(p)[1] for p in HULL) < 0.02
        if hull_down or abs(x) > 9 or y > 16:
            return reward - 100, True
        resting = all(self._legs_down()) and math.hypot(vx, vy) < 0.1 and abs(self.lander.angular_velocity) < 0.2
        if resting:
            return reward + 100, True
        return reward, False

    def human_action(self, keys):
        """up: main engine, left / right: rotate"""
        return np.array([float("up" in keys), ("left" in keys) - ("right" in keys)], dtype=np.float32)
