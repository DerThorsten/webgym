"""LunarLander: land softly on the pad between the flags (the classic one).

As Gymnasium's LunarLander (continuous): the same lander, legs on springy joints, moon
surface with a flat pad, engines, observations and reward (written anew for box2d 3).
Only the surface is the same every episode (all agents share one world).
obs: x, y (relative to the pad), velocity x, y, angle, angular velocity, leg contacts.
action: main engine (<= 0: off, 0..1: half to full power), side engines (|a| <= 0.5:
off, else the left (-) or right (+) one, half to full power).
reward: closer to the pad, slower, upright, legs down (shaping); fuel costs 0.3 per
step at full main power, 0.03 for the side engines; -100 for a crash (the hull touches
the ground, or the lander leaves the screen sideways), +100 for landing (it comes to
rest). An episode lasts up to 1000 steps (20 s).
"""

import math

import numpy as np
import pyb2d3 as b2d
from pyb2d3 import extensions as ext

from webgym.core import DARK_COLOR, GHOST_COLOR, LIGHT_COLOR, PARTICLE_COLOR, PARTICLE_FILTER, Env, register_env

# Gymnasium's constants (pixels / SCALE = meters)
FPS = 50
SCALE = 30.0
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_RANDOM = 1000.0  # the random push at the start (a force for one step)
LANDER_POLY = [(-14, 17), (-17, 0), (-17, -10), (17, -10), (17, 0), (14, 17)]
LEG_AWAY, LEG_DOWN, LEG_W, LEG_H = 20, 18, 2, 8
LEG_SPRING_TORQUE = 40
SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY = 14, 12
MAIN_ENGINE_Y_LOCATION = 4
W, H = 600 / SCALE, 400 / SCALE  # the screen, in meters
# the moon's surface: as Gymnasium makes it, but once (seeded), not every episode
CHUNKS = 11
HELIPAD_Y = H / 4
_height = np.random.default_rng(3).uniform(0, H / 2, CHUNKS + 1)
CHUNK_X = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
HELIPAD_X1, HELIPAD_X2 = CHUNK_X[CHUNKS // 2 - 1], CHUNK_X[CHUNKS // 2 + 1]
_height[CHUNKS // 2 - 2: CHUNKS // 2 + 3] = HELIPAD_Y
SMOOTH_Y = [0.33 * (_height[i - 1] + _height[i] + _height[i + 1]) for i in range(CHUNKS)]


@register_env("LunarLander")
class LunarLander(Env):
    obs_names = ("x", "y", "vel. x", "vel. y", "angle", "ang. vel.", "left leg", "right leg")
    action_names = ("main engine", "side engines")
    # (for DQN: nothing / left engine / main engine / right engine, as the classic lander)
    discrete_actions = np.array([[0, 0], [0, -1], [1, 0], [0, 1]])
    discrete_names = ("nothing", "left engine", "main engine", "right engine")
    dt, substeps = 1 / FPS, 4
    max_steps = 1000
    hinge_discs = False  # (the legs' joints sit at the lander's center, as in Gymnasium: no discs)
    gravity = (0, -10.0)
    view_center, view_zoom = (W / 2, H / 2), H / 2 + 0.5
    suggested = {"quick": {"cem": {}, "neat": {}},
                 "medium": {"cem": {}, "reinforce_torch": {}, "neat": {}},
                 "long": {"ppo_torch": {"hidden": 64}, "reinforce_torch": {}, "cem": {}}}
    # (SAC / TD3: no landing within 12 min of pretraining on the classic one)

    def build_scenery(self):
        for x0, x1, y0, y1 in zip(CHUNK_X, CHUNK_X[1:], SMOOTH_Y, SMOOTH_Y[1:]):
            self.scenery(b2d.polygon([(x0, 0), (x1, 0), (x1, y1), (x0, y0)]), friction=0.1)
        if self.decorated:  # the pad's flags: looks only, nothing hits them
            for x in (HELIPAD_X1, HELIPAD_X2):
                top = HELIPAD_Y + 50 / SCALE
                self.scenery(b2d.box(0.03, 25 / SCALE, center=(x, HELIPAD_Y + 25 / SCALE)), solid=False)
                self.scenery(b2d.polygon([(x, top), (x, top - 10 / SCALE), (x + 25 / SCALE, top - 5 / SCALE)]),
                             solid=False)

    def build_agent(self, i):
        lander = self.body((W / 2, H), b2d.polygon([(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                           density=5.0, friction=0.1, enable_sleep=True, color=LIGHT_COLOR)  # (asleep: landed)
        legs, joints, feet = [], [], []
        for side in (-1, 1):
            # (a capsule as long and wide as Gymnasium's box leg)
            end = (LEG_H - LEG_W) / SCALE
            leg = self.body((W / 2 - side * LEG_AWAY / SCALE, H), b2d.capsule((0, -end), (0, end), LEG_W / SCALE),
                            density=1.0, friction=0.1, enable_sleep=True, rotation=side * 0.05, color=DARK_COLOR)
            # (the springy legs: a motor pushes them out against a limit)
            joints.append(self.world.create_revolute_joint(
                body_a=lander, body_b=leg, local_frame_a=b2d.transform(p=(0, 0)),
                local_frame_b=b2d.transform(p=(side * LEG_AWAY / SCALE, LEG_DOWN / SCALE)),
                enable_motor=True, enable_limit=True, max_motor_torque=LEG_SPRING_TORQUE, motor_speed=0.3 * side,
                lower_angle=0.4 if side == -1 else -0.9, upper_angle=0.9 if side == -1 else -0.4,
            ))
            # (a leg touches the ground: a sensor around it)
            feet.append(self.sensor(leg, b2d.box(1.5 * LEG_W / SCALE, 1.1 * LEG_H / SCALE), density=0.0))
            legs.append(leg)
        # (a crash: the hull touches the ground; a sensor a little larger than the hull)
        hull = self.sensor(lander, b2d.polygon([(1.05 * x / SCALE, 1.05 * y / SCALE) for x, y in LANDER_POLY]),
                           density=0.0)  # (sensors weigh nothing here: Gymnasium's masses)
        parts = {"lander": lander, "legs": legs, "leg_joints": joints, "feet": feet, "hull": hull}
        if self.decorated:  # exhaust (looks only): the watched policy's orange, the ghosts' blue
            color = PARTICLE_COLOR if i == 0 else GHOST_COLOR  # (the ghosts': translucent)
            parts["main_exhaust"] = exhaust(self.world, lander, (0, -MAIN_ENGINE_Y_LOCATION / SCALE), -math.pi / 2,
                                            color)
            parts["left_exhaust"] = exhaust(self.world, lander, (-17 / SCALE, SIDE_ENGINE_HEIGHT / SCALE), math.pi,
                                            color)
            parts["right_exhaust"] = exhaust(self.world, lander, (17 / SCALE, SIDE_ENGINE_HEIGHT / SCALE), 0.0, color)
        return parts

    def randomize(self, rng):
        n = self.n
        # the random push: Gymnasium's force for one step, as a velocity
        force = rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM, (n, 2))
        self.lander.linear_velocity = force / FPS / self.lander[0].mass
        self.rng = rng
        self.m_power, self.s_power, self.actions_side = np.zeros(n), np.zeros(n), np.zeros(n)
        self.shaping = self._shaping(self._observation())

    def act(self, actions):
        n, angle = self.n, self.lander.angle
        tip = np.column_stack([np.sin(angle), np.cos(angle)])
        side = np.column_stack([-tip[:, 1], tip[:, 0]])
        dispersion = self.rng.uniform(-1, 1, (n, 2)) / SCALE
        pos = self.lander.position
        main = actions[:, 0]
        self.m_power = np.where(main > 0, (np.clip(main, 0, 1) + 1) * 0.5, 0.0)
        reach = MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[:, 0]
        ox = tip[:, 0] * reach + side[:, 0] * dispersion[:, 1]
        oy = -tip[:, 1] * reach - side[:, 1] * dispersion[:, 1]
        impulse = np.column_stack([-ox, -oy]) * (MAIN_ENGINE_POWER * self.m_power)[:, None]
        self.lander.apply_linear_impulse(impulse, pos + np.column_stack([ox, oy]))
        steer = self.actions_side = actions[:, 1]
        direction = np.sign(steer)
        self.s_power = np.where(np.abs(steer) > 0.5, np.clip(np.abs(steer), 0.5, 1.0), 0.0)
        away = 3 * dispersion[:, 1] + direction * SIDE_ENGINE_AWAY / SCALE
        ox = tip[:, 0] * dispersion[:, 0] + side[:, 0] * away
        oy = -tip[:, 1] * dispersion[:, 0] - side[:, 1] * away
        point = pos + np.column_stack([ox - tip[:, 0] * 17 / SCALE, oy + tip[:, 1] * SIDE_ENGINE_HEIGHT / SCALE])
        impulse = np.column_stack([-ox, -oy]) * (SIDE_ENGINE_POWER * self.s_power)[:, None]
        self.lander.apply_linear_impulse(impulse, point)

    def observe(self):
        pos, vel = self.lander.position, self.lander.linear_velocity
        legs = self.feet.sensor_capacity.reshape(self.n, 2) > 0
        return [(pos[:, 0] - W / 2) / (W / 2), (pos[:, 1] - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H / 2),
                vel[:, 0] * (W / 2) / FPS, vel[:, 1] * (H / 2) / FPS,
                self.lander.angle, 20.0 * self.lander.angular_velocity / FPS, legs]

    def reward(self):
        obs = self._observation()
        shaping = self._shaping(obs)
        reward = shaping - self.shaping - 0.30 * self.m_power - 0.03 * self.s_power
        self.shaping = shaping
        crashed = (self.hull.sensor_capacity > 0) | (np.abs(obs[:, 0]) >= 1.0)
        landed = ~np.asarray(self.lander.awake) & ~crashed  # (it came to rest: box2d put it to sleep)
        reward = np.where(crashed, -100.0, np.where(landed, 100.0, reward))
        return reward, crashed | landed

    @staticmethod
    def _shaping(obs):
        """Gymnasium's potential: near the pad, slow, upright, legs down"""
        return (-100 * np.hypot(obs[:, 0], obs[:, 1]) - 100 * np.hypot(obs[:, 2], obs[:, 3])
                - 100 * np.abs(obs[:, 4]) + 10 * obs[:, 6] + 10 * obs[:, 7])

    def decorate(self):
        """the exhaust of the engines that fire (watch mode): the right side engine
        pushes the lander left, the left one right"""
        on = ~self.done
        rates = {"main_exhaust": 120 * self.m_power * on,
                 "right_exhaust": 60 * np.where(self.actions_side > 0, self.s_power, 0) * on,
                 "left_exhaust": 60 * np.where(self.actions_side < 0, self.s_power, 0) * on}
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
