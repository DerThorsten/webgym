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

HIP_Y = 1.2  # standing height of the hips
THIGH, SHIN = 0.32, 0.3  # half lengths
MAX_TORQUE = 12.0  # N m at full action
# no progress (the hull did not get STALL_DISTANCE further right than ever before)
# for STALL_SECONDS: the episode ends (no penalty), the population does not wait for it
STALL_SECONDS, STALL_DISTANCE = 1.5, 0.2
# hips: (lower, upper) angle of the thigh against the hull; knees: shin against thigh
HIP_LIMITS, KNEE_LIMITS = (-0.8, 1.1), (-1.5, 0.05)


@register_env("Walker")
class Walker(Env):
    observation_space = Box(-np.inf, np.inf, (14,))
    action_space = Box(-1.0, 1.0, (4,))
    max_steps = 600
    view_center = (0.0, 2.0)
    view_zoom = 3.5
    keys = "left leg: Q / A hip, W / S knee · right leg: O / L hip, P / ; knee"

    def build(self):
        ground = self.world.create_static_body(position=(0, 0))
        self.scenery_shape(ground, b2d.box(200, 0.5, center=(190, -0.5)), friction=1.0)
        hulls, parents, children, feet = [], [[] for _ in range(4)], [[] for _ in range(4)], [[], []]
        for i in range(self.n):
            hull = self.agent_body(i, (0, HIP_Y + 0.15), [(b2d.box(0.4, 0.15), 5.0)])
            hulls.append(hull)
            for leg in range(2):
                thigh = self.agent_body(i, (0, HIP_Y - THIGH), [(b2d.box(0.07, THIGH), 3.0)])
                shin = self.agent_body(
                    i, (0, HIP_Y - 2 * THIGH - SHIN),
                    [(b2d.capsule((0, -SHIN + 0.05), (0, SHIN - 0.05), 0.05), 3.0)],
                )
                self._joint(hull, thigh, (0, -0.15), (0, THIGH), HIP_LIMITS)
                self._joint(thigh, shin, (0, -THIGH), (0, SHIN), KNEE_LIMITS)
                # joint j = 2 * leg + (0: hip, 1: knee)
                parents[2 * leg].append(hull)
                children[2 * leg].append(thigh)
                parents[2 * leg + 1].append(thigh)
                children[2 * leg + 1].append(shin)
                feet[leg].append(shin)
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
        for j in range(4):
            torque = (a[:, j] * MAX_TORQUE).astype(np.float32)
            self.children[j].apply_torque(torque, True)
            self.parents[j].apply_torque(-torque, True)
        self.effort = np.square(a).sum(axis=1)

    def _observe(self):
        angles = [self.children[j].angle - self.parents[j].angle for j in range(4)]
        speeds = [(self.children[j].angular_velocity - self.parents[j].angular_velocity) / 6 for j in range(4)]
        feet = []
        for f in self.feet:  # the foot (lower end of the shin) near the ground
            pos, angle = f.position, f.angle
            feet.append((pos[:, 1] - SHIN * np.cos(angle) < 0.08).astype(np.float32))
        return np.column_stack(
            [self.hulls.angle, self.hulls.angular_velocity / 2, self.hulls.linear_velocity / 3,
             *angles, *speeds, *feet]
        ).astype(np.float32)

    def _outcome(self):
        pos = self.hulls.position
        reward = 10 * (pos[:, 0] - self.prev_x) - 0.005 * self.effort
        self.prev_x = pos[:, 0].copy()
        fallen = (pos[:, 1] < 0.55) | (np.abs(self.hulls.angle) > 1.0)
        progressed = pos[:, 0] > self.best_x + STALL_DISTANCE
        self.best_x = np.where(progressed, pos[:, 0], self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        return reward - 100 * fallen, fallen | stalled

    def human_action(self, keys):
        """two keys per joint (see the module docstring)"""
        pairs = [("q", "a"), ("w", "s"), ("o", "l"), ("p", ";")]
        return np.array([(plus in keys) - (minus in keys) for plus, minus in pairs], dtype=np.float32)
