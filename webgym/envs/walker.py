"""Walker: a two-legged robot learns to walk to the right.

A hull with two legs (thigh + shin), torques at the hips and knees.
obs: hull angle, angular velocity and velocity, the 4 joint angles and speeds,
foot contacts (14 values).
action: 4 joint torques in [-1, 1] (hips, knees).
reward: forward progress, minus a small effort cost; -100 when the hull hits
the ground. Episodes last 10 s, or end when the walker makes no progress for 1.5 s.

Keys (human controller): q / a = left hip, w / s = left knee,
o / l = right hip, p / ; = right knee.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Box, Env, register_env

# no progress (the hull did not get STALL_DISTANCE further right than ever before)
# for STALL_SECONDS: the episode ends (no penalty), the population does not wait for it
STALL_SECONDS, STALL_DISTANCE = 1.5, 0.2
FOOT_H = 0.04  # half height of a foot
JOINT_KINDS = ("hip", "knee", "ankle")


class Legged(Env):
    """walker-like robots (Walker, Crawler, Hopper): a hull with legs at LEG_XS; each leg
    a thigh and a shin (+ a foot when FOOT > 0), torques at the hip, knee (and ankle)
    joints. Spaces and names follow from the legs (see __init_subclass__)."""

    HULL = (0.4, 0.15)  # half size of the hull box
    LEG_XS = (0.0, 0.0)  # x of the hips along the hull
    LEG_NAMES = ("L", "R")
    HIP_Y = 1.2  # standing height of the hips
    THIGH, SHIN = 0.32, 0.3  # half lengths
    FOOT = 0.0  # half length of a foot (0: no foot, no ankle)
    MAX_TORQUE = 12.0  # N m at full action
    LIMITS = {"hip": (-0.8, 1.1), "knee": (-1.5, 0.05), "ankle": (-0.6, 0.6)}
    FALL_Y, FALL_ANGLE = 0.55, 1.0  # the hull lower / tilted more: fallen (-100, end)
    # reward per step upright: with the no-progress end, staying alive = moving on
    ALIVE = 0.0
    KEY_PAIRS = ()  # human controller: (plus, minus) key per joint
    max_steps = 600
    view_center = (0.0, 2.0)
    view_zoom = 3.5

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        kinds = JOINT_KINDS[: 3 if cls.FOOT else 2]
        joints = [f"{kind} {leg}".strip() for leg in cls.LEG_NAMES for kind in kinds]
        cls.action_names = tuple(joints)
        cls.action_space = Box(-1.0, 1.0, (len(joints),))
        cls.obs_names = ("hull angle", "hull ang. vel.", "vel. x", "vel. y", *joints,
                         *(f"{j} speed" for j in joints), *(f"foot {leg}".strip() for leg in cls.LEG_NAMES))
        cls.observation_space = Box(-np.inf, np.inf, (len(cls.obs_names),))

    def build(self):
        ground = self.world.create_static_body(position=(0, 0))
        self.scenery_shape(ground, b2d.box(200, 0.5, center=(190, -0.5)), friction=1.0)
        joints = len(self.action_names)
        hulls, parents, children = [], [[] for _ in range(joints)], [[] for _ in range(joints)]
        feet = [[] for _ in self.LEG_XS]
        for i in range(self.n):
            hull = self.agent_body(i, (0, self.HIP_Y + self.HULL[1]), [(b2d.box(*self.HULL), 5.0)])
            hulls.append(hull)
            j = 0
            for leg, x in enumerate(self.LEG_XS):
                thigh = self.agent_body(i, (x, self.HIP_Y - self.THIGH), [(b2d.box(0.07, self.THIGH), 3.0)])
                shin = self.agent_body(
                    i, (x, self.HIP_Y - 2 * self.THIGH - self.SHIN),
                    [(b2d.capsule((0, -self.SHIN + 0.05), (0, self.SHIN - 0.05), 0.05), 3.0)],
                )
                links = [(hull, thigh, (x, -self.HULL[1]), (0, self.THIGH), "hip"),
                         (thigh, shin, (0, -self.THIGH), (0, self.SHIN), "knee")]
                lowest = shin
                if self.FOOT:
                    y = self.HIP_Y - 2 * self.THIGH - 2 * self.SHIN - FOOT_H
                    foot = self.agent_body(i, (x, y), [(b2d.box(self.FOOT, FOOT_H), 3.0)])
                    links.append((shin, foot, (0, -self.SHIN), (0, FOOT_H), "ankle"))
                    lowest = foot
                for a, b, anchor_a, anchor_b, kind in links:  # joint j: parent a, child b
                    self._joint(a, b, anchor_a, anchor_b, self.LIMITS[kind])
                    parents[j].append(a)
                    children[j].append(b)
                    j += 1
                feet[leg].append(lowest)
        self.hulls = b2d.Bodies(hulls)
        self.parents = [b2d.Bodies(p) for p in parents]
        self.children = [b2d.Bodies(c) for c in children]
        self.feet = [b2d.Bodies(f) for f in feet]

    def _joint(self, a, b, anchor_a, anchor_b, limits):
        """a revolute joint with a limit and some friction (a motor holding speed 0
        with a small torque); the actions are torques on the two bodies"""
        self.world.create_revolute_joint(
            body_a=a,
            body_b=b,
            local_frame_a=b2d.transform(p=anchor_a),
            local_frame_b=b2d.transform(p=anchor_b),
            enable_limit=True,
            lower_angle=limits[0],
            upper_angle=limits[1],
            enable_motor=True,
            motor_speed=0.0,
            max_motor_torque=0.5,
        )

    def _reset(self, rng):
        velocity = np.zeros((self.n, 2), dtype=np.float32)
        velocity[:, 0] = rng.uniform(-0.2, 0.2, self.n)
        self.hulls.linear_velocity = velocity
        self.prev_x = self.hulls.position[:, 0].copy()
        self.effort = np.zeros(self.n)
        self.best_x = self.prev_x.copy()  # progress: the farthest x so far
        self.progress_step = np.zeros(self.n, dtype=int)  # step of the last progress

    def _apply(self, actions):
        a = np.clip(actions, -1, 1)
        for j in range(len(self.children)):
            torque = (a[:, j] * self.MAX_TORQUE).astype(np.float32)
            self.children[j].apply_torque(torque, True)
            self.parents[j].apply_torque(-torque, True)
        self.effort = np.square(a).sum(axis=1)

    def _observe(self):
        joints = range(len(self.children))
        angles = [self.children[j].angle - self.parents[j].angle for j in joints]
        speeds = [(self.children[j].angular_velocity - self.parents[j].angular_velocity) / 6 for j in joints]
        feet = []
        for f in self.feet:  # the lowest point of the leg near the ground
            pos, angle = f.position, f.angle
            low = pos[:, 1] - (FOOT_H if self.FOOT else self.SHIN * np.cos(angle))
            feet.append((low < 0.08).astype(np.float32))
        return np.column_stack(
            [self.hulls.angle, self.hulls.angular_velocity / 2, self.hulls.linear_velocity / 3,
             *angles, *speeds, *feet]
        ).astype(np.float32)

    def _outcome(self):
        pos = self.hulls.position
        reward = 10 * (pos[:, 0] - self.prev_x) - 0.005 * self.effort
        self.prev_x = pos[:, 0].copy()
        fallen = (pos[:, 1] < self.FALL_Y) | (np.abs(self.hulls.angle) > self.FALL_ANGLE)
        progressed = pos[:, 0] > self.best_x + STALL_DISTANCE
        self.best_x = np.where(progressed, pos[:, 0], self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        return reward + self.ALIVE * ~fallen - 100 * fallen, fallen | stalled

    def human_action(self, keys):
        """two keys per joint (KEY_PAIRS, see the env's docstring)"""
        return np.array([(plus in keys) - (minus in keys) for plus, minus in self.KEY_PAIRS], dtype=np.float32)

    def camera_target(self, i):
        """follow the hull to the right (the height stays: no bobbing)"""
        return float(self.hulls.position[i, 0]), self.view_center[1]


@register_env("Walker")
class Walker(Legged):
    keys = "left leg: Q / A hip, W / S knee · right leg: O / L hip, P / ; knee"
    KEY_PAIRS = (("q", "a"), ("w", "s"), ("o", "l"), ("p", ";"))
