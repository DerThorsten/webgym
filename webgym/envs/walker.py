"""Walker: a two-legged robot learns to walk to the right.

A hull with two legs (thigh + shin), motors at the hips and knees.
obs: hull angle, angular velocity and velocity, the 4 joint angles and speeds,
foot contacts (14 values).
action: 4 joint torques in [-1, 1] (hips, knees).
reward: forward progress, minus a small effort cost; -100 when the hull hits
the ground. Episodes last 10 s, or end when the walker makes no progress for 1.5 s.
"""

import itertools

import numpy as np
import pyb2d3 as b2d

from webgym.core import DECORATION_FILTER, MARK_COLOR, Env, register_env

# no progress (the hull did not get STALL_DISTANCE further right than ever before) for
# STALL_SECONDS: the episode ends (no penalty), the population does not wait for it
STALL_SECONDS, STALL_DISTANCE = 1.5, 0.2


def ground_with_ticks(env):
    """flat ground to the right, with ticks every metre (every 5 m a longer one): you see
    the progress. The ticks are looks only: nothing collides with them"""
    ground = env.scenery(b2d.box(200, 0.5, center=(190, -0.5)), friction=1.0)
    tick = b2d.shape_def(filter=DECORATION_FILTER, material=b2d.surface_material(custom_color=MARK_COLOR))
    for x in range(-10, 390):
        ground.create_shape(tick, b2d.segment((x, 0), (x, -0.4 if x % 5 == 0 else -0.15)))


def legged_names(legs):
    """obs_names, action_names of a robot with these legs (a hip and a knee each)"""
    joints = [f"{kind} {leg}" for leg in legs for kind in ("hip", "knee")]
    obs = ("hull angle", "hull ang. vel.", "vel. x", "vel. y", *joints,
           *(f"{joint} speed" for joint in joints), *(f"foot {leg}" for leg in legs))
    return obs, tuple(joints)


def bang_bang(groups, joints):
    """discrete actions: every joint at full torque, one way or the other; the joints of
    a group (indices) together -> (actions (2^groups, joints), names like "+-+-")"""
    signs = np.array(list(itertools.product((1, -1), repeat=len(groups))))
    actions = np.zeros((len(signs), joints))
    for g, group in enumerate(groups):
        actions[:, group] = signs[:, g : g + 1]
    return actions, tuple("".join("+" if s > 0 else "-" for s in row) for row in signs)


class Legged(Env):
    """walker-like robots (Walker, Crawler): a hull with legs at LEG_XS, each leg a
    thigh and a shin, motors at the hip and knee joints"""

    HULL = (0.4, 0.15)  # half size of the hull box
    LEG_XS = (0.0, 0.0)  # x of the hips along the hull (one per leg)
    HIP_Y = 1.2  # standing height of the hips
    THIGH, SHIN = 0.32, 0.3  # half lengths
    MAX_TORQUE = 12.0  # N m at full action
    MAX_SPEED = 20.0  # rad/s the motors drive towards
    LIMITS = {"hip": (-0.8, 1.1), "knee": (-1.5, 0.05)}
    FALL_Y, FALL_ANGLE = 0.55, 1.0  # the hull lower / tilted more: fallen (-100, end)
    max_steps = 600
    view_center, view_zoom = (0.0, 2.0), 3.5

    def build_scenery(self):
        ground_with_ticks(self)

    def build_agent(self, i):
        hull = self.body((0, self.HIP_Y + self.HULL[1]), b2d.box(*self.HULL), density=5.0)
        joints, feet = [], []
        for x in self.LEG_XS:
            thigh = self.body((x, self.HIP_Y - self.THIGH), b2d.box(0.07, self.THIGH), density=3.0)
            shin = self.body((x, self.HIP_Y - 2 * self.THIGH - self.SHIN),
                             b2d.capsule((0, -self.SHIN + 0.05), (0, self.SHIN - 0.05), 0.05), density=3.0)
            joints.append(self.joint(hull, thigh, (x, -self.HULL[1]), (0, self.THIGH), "hip"))
            joints.append(self.joint(thigh, shin, (0, -self.THIGH), (0, self.SHIN), "knee"))
            feet.append(self.sensor(shin, b2d.circle(radius=0.07, center=(0, -self.SHIN))))
        return {"hull": hull, "joints": joints, "feet": feet}

    def joint(self, a, b, anchor_a, anchor_b, kind):
        """a revolute joint (anchors: local points of a and b) with a limit and a motor"""
        lower, upper = self.LIMITS[kind]
        return self.world.create_revolute_joint(
            body_a=a, body_b=b,
            local_frame_a=b2d.transform(p=anchor_a), local_frame_b=b2d.transform(p=anchor_b),
            enable_limit=True, lower_angle=lower, upper_angle=upper, enable_motor=True,
        )

    def randomize(self, rng):
        self.hull.linear_velocity = np.column_stack([rng.uniform(-0.2, 0.2, self.n), np.zeros(self.n)])
        self.effort = np.zeros(self.n)
        self.prev_x = self.hull.position[:, 0]
        self.best_x = self.prev_x.copy()  # progress: the farthest x so far
        self.progress_step = np.zeros(self.n, dtype=int)  # step of the last progress

    def act(self, actions):
        # each motor pushes towards +- MAX_SPEED with a torque up to |action| (at 0: a
        # little friction holds the joint)
        self.joints.target_motor_speed = (np.sign(actions) * self.MAX_SPEED).ravel()
        self.joints.max_motor_torque = np.maximum(np.abs(actions) * self.MAX_TORQUE, 0.5).ravel()
        self.effort = np.square(actions).sum(axis=1)

    def observe(self):
        n = self.n
        return [self.hull.angle, self.hull.angular_velocity / 2, self.hull.linear_velocity / 3,
                self.joints.current_angle.reshape(n, -1), self.joints.current_speed.reshape(n, -1) / 6,
                self.feet.sensor_capacity.reshape(n, -1) > 0]

    def reward(self):
        x, y = self.hull.position.T
        reward = 10 * (x - self.prev_x) - 0.005 * self.effort
        self.prev_x = x
        fallen = (y < self.FALL_Y) | (np.abs(self.hull.angle) > self.FALL_ANGLE)
        progressed = x > self.best_x + STALL_DISTANCE
        self.best_x = np.where(progressed, x, self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        return reward - 100 * fallen, fallen | stalled

    def camera_target(self, i):
        """follow the hull to the right (the height stays: no bobbing)"""
        return float(self.hull.position[i, 0]), self.view_center[1]


@register_env("Walker")
class Walker(Legged):
    obs_names, action_names = legged_names(("L", "R"))
    # (for DQN: each joint at full torque, + or -: "+-+-" = hip L +, knee L -, ...)
    discrete_actions, discrete_names = bang_bang([[0], [1], [2], [3]], 4)
    suggested = {"quick": {"cem": {}, "es": {}},
                 "medium": {"cem": {}, "sac_torch": {}, "es": {}},
                 "long": {"ppo_torch": {"hidden": 64}, "sac_torch": {"updates": 4}, "es": {}}}
