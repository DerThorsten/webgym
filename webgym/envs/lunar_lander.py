"""LunarLander: land softly between the flags.

The lander starts high up with a random push, above a jagged moon surface with a flat
landing pad between the flags (landing elsewhere counts too, but the pad is better). Main engine (thrust along the
lander's up axis) and attitude thrusters (turning).
obs: position, velocity, angle, angular velocity, leg contacts (8 values).
action: [main engine (<= 0: off, 1: full), turn (-1 .. 1, + = left)].
reward: getting closer to the pad, slower and upright, legs down; fuel costs;
+100 for coming to rest on the legs, -100 for touching the ground with the hull
or flying away.
"""

import math

import numpy as np
import pyb2d3 as b2d
from pyb2d3 import extensions as ext

from webgym.core import PARTICLE_COLOR, PARTICLE_FILTER, Env, register_env

HULL = [(-0.5, -0.25), (0.5, -0.25), (0.35, 0.35), (-0.35, 0.35)]
LEGS = [((-0.55, -0.45), -0.35), ((0.55, -0.45), 0.35)]  # center, tilt (splayed outwards)
LEG_LENGTH = 0.6
FEET = [(-0.653, -0.732), (0.653, -0.732)]  # the lower ends of the legs (local points)
PAD = 1.6  # the landing pad: flat, from -PAD to PAD, between the flags
# the moon's surface (as in the classic lander): heights at x = -12, -10.8, ... 12,
# jagged, flat at the pad (the same for every episode)
SURFACE_X = np.linspace(-12, 12, 21)
SURFACE_Y = np.where(np.abs(SURFACE_X) <= PAD + 1.0, 0.0, np.random.default_rng(3).uniform(0.0, 2.2, 21))


@register_env("LunarLander")
class LunarLander(Env):
    obs_names = ("x", "height", "vel. x", "vel. y", "angle", "ang. vel.", "left leg", "right leg")
    action_names = ("main engine", "rotate")
    # (for DQN: nothing / main engine / rotate left / rotate right, as the classic lander)
    discrete_actions = np.array([[0, 0], [1, 0], [0, 1], [0, -1]])
    discrete_names = ("nothing", "main engine", "rotate left", "rotate right")
    max_steps = 600
    gravity = (0, -5.0)
    view_center, view_zoom = (0.0, 5.0), 6.0
    suggested = {"quick": {"cem": {}, "neat": {}},
                 "medium": {"reinforce_torch": {}, "cem": {}, "neat": {}},
                 "long": {"ppo_torch": {"hidden": 64}, "sac_torch": {}, "td3_torch": {}, "reinforce_torch": {}}}

    def build_scenery(self):
        # the surface: one 4-sided piece per stretch between two heights
        for x0, x1, y0, y1 in zip(SURFACE_X, SURFACE_X[1:], SURFACE_Y, SURFACE_Y[1:]):
            self.scenery(b2d.polygon([(x0, -1), (x1, -1), (x1, y1), (x0, y0)]), friction=0.8)
        if self.decorated:  # the pad's flags: looks only, nothing hits them
            for x in (-PAD, PAD):
                self.scenery(b2d.box(0.02, 0.6, center=(0, 0.6)), position=(x, 0), solid=False)
                self.scenery(b2d.polygon([(0, 1.2), (0, 0.9), (0.35, 1.05)]), position=(x, 0), solid=False)

    def build_agent(self, i):
        lander = self.body((0, 9), b2d.polygon(HULL))
        for center, tilt in LEGS:
            self.shape(lander, b2d.box(0.05, LEG_LENGTH / 2, center=center, rotation=tilt), density=0.5)
        # sensors: the feet touch the ground (legs down), the hull touches it (a crash)
        feet = [self.sensor(lander, b2d.circle(center=foot, radius=0.06)) for foot in FEET]
        hull = self.sensor(lander, b2d.polygon([(1.05 * x, 1.05 * y) for x, y in HULL]))
        parts = {"lander": lander, "feet": feet, "hull": hull}
        if self.decorated:  # exhaust (looks only): the watched policy's orange, the ghosts' blue
            color = PARTICLE_COLOR if i == 0 else 0  # (0: drawn in the ghost color)
            parts["main_exhaust"] = exhaust(self.world, lander, (0, -0.3), -math.pi / 2, color)
            parts["left_exhaust"] = exhaust(self.world, lander, (-0.375, 0.25), math.pi, color)
            parts["right_exhaust"] = exhaust(self.world, lander, (0.375, 0.25), 0.0, color)
        return parts

    def randomize(self, rng):
        n = self.n
        self.lander.position = np.column_stack([rng.uniform(-3, 3, n), np.full(n, 9.0)])
        self.lander.angle = rng.uniform(-0.2, 0.2, n)
        self.lander.linear_velocity = np.column_stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1.0, 0.0, n)])
        self.lander.angular_velocity = rng.uniform(-0.5, 0.5, n)
        self.main, self.turn = np.zeros(n), np.zeros(n)
        self.shaping = self._shaping(self._observation())

    def act(self, actions):
        self.main, self.turn = np.maximum(actions[:, 0], 0.0), actions[:, 1]
        mass, inertia = self.lander[0].mass, self.lander[0].rotational_inertia
        thrust = self.main * 2.2 * mass * -self.gravity[1]  # along the lander's up axis
        angle = self.lander.angle
        self.lander.apply_force_to_center(np.column_stack([-np.sin(angle) * thrust, np.cos(angle) * thrust]), True)
        self.lander.apply_torque(self.turn * 3.0 * inertia, True)

    def observe(self):
        return [self.lander.position[:, 0] / 5, (self.lander.position[:, 1] - 0.732) / 5,
                self.lander.linear_velocity / 5, self.lander.angle, self.lander.angular_velocity / 2,
                self._legs_down()]

    def reward(self):
        obs = self._observation()
        shaping = self._shaping(obs)
        reward = shaping - self.shaping - 0.3 * self.main - 0.03 * np.abs(self.turn)
        self.shaping = shaping
        x = self.lander.position[:, 0]
        hull_down = self.hull.sensor_capacity > 0
        crashed = hull_down |(np.abs(x) > 9) | (self.lander.position[:, 1] > 16)
        speed = np.linalg.norm(self.lander.linear_velocity, axis=1)
        resting = (obs[:, 6] > 0) & (obs[:, 7] > 0) & (speed < 0.1) & (np.abs(self.lander.angular_velocity) < 0.2)
        return reward - 100 * crashed + 100 * (resting & ~crashed), crashed | resting

    @staticmethod
    def _shaping(obs):
        """potential: high near the pad, slow, upright, legs down"""
        return (-100 * np.hypot(obs[:, 0], obs[:, 1]) - 100 * np.hypot(obs[:, 2], obs[:, 3])
                - 100 * np.abs(obs[:, 4]) + 10 * (obs[:, 6] + obs[:, 7]))

    def _legs_down(self):
        """(n, 2): each foot on the ground (its sensor)"""
        return self.feet.sensor_capacity.reshape(self.n, 2) > 0

    def decorate(self):
        """the exhaust of the engines that fire (watch mode): the side thruster on the
        right blows right to turn the lander left, the left one the other way"""
        on = ~self.done
        rates = {"main_exhaust": 120 * self.main * on, "right_exhaust": 60 * np.maximum(self.turn, 0) * on,
                 "left_exhaust": 60 * np.maximum(-self.turn, 0) * on}
        for name, rate in rates.items():
            for emitter, r in zip(getattr(self, name), rate):
                emitter.rate = float(r)
                emitter.step(self.dt)


def exhaust(world, body, position, angle, color):
    """a nozzle on `body` at `position`, blowing towards `angle`: an emitter of small
    particles (rate set each step) that live 0.5 s, slowed by drag (as smoke)"""
    return ext.RadialEmitter(
        world,
        b2d.shape_def(density=0.2, filter=PARTICLE_FILTER, material=b2d.surface_material(restitution=0.3)),
        b2d.circle(radius=0.025),
        body=body,
        local_frame=b2d.transform(position, b2d.make_rot(angle)),
        body_def=b2d.body_def(linear_damping=3.0),
        rate=0.0, capacity=80, angle_min=-0.12, angle_max=0.12,
        speed=ext.uniform(5.0, 8.0), lifetime=ext.constant(0.5), max_lifetime=0.5,
        colors=[(0.0, color), (1.0, color)],
    )
