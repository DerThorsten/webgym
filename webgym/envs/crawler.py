"""Crawler: a four-legged robot learns to crawl to the right.

A long, low hull with two leg pairs (back + front, thigh + shin each), torques at the
8 hips and knees (motors). Hard to tip over, it learns quickly.
obs: hull angle, angular velocity and velocity, the 8 joint angles and speeds,
the 4 foot contacts (24 values).
action: 8 joint torques in [-1, 1].
reward: forward progress, minus a small effort cost; -100 when it flips over.
Episodes last 10 s, or end when it makes no progress for 1.5 s.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import DARK_COLOR, LIGHT_COLOR, Env, bang_bang, draw_ruler_numbers, register_env, ruler

HULL = (0.6, 0.12)  # half size of the hull box
LEGS = {"back L": -0.5, "back R": -0.5, "front L": 0.5, "front R": 0.5}  # x of each leg's hip on the hull
HIP_Y = 0.8  # standing height of the hips
THIGH, SHIN = 0.2, 0.2  # half lengths
MAX_TORQUE = 8.0  # N m at full action
MAX_SPEED = 20.0  # rad/s the motors drive towards
LIMITS = {"hip": (-1.0, 1.0), "knee": (-1.5, 0.05)}  # joint angles (rad)
FALL_Y, FALL_ANGLE = 0.25, 1.5  # the hull lower / tilted more: fallen (only flipping over)
# no progress (the hull did not get STALL_DISTANCE further right than ever before) for
# STALL_SECONDS: the episode ends (no penalty), the population does not wait for it
STALL_SECONDS, STALL_DISTANCE = 1.5, 0.2
JOINTS = tuple(f"{kind} {leg}" for leg in LEGS for kind in ("hip", "knee"))  # ("hip back L", "knee back L", ...)


@register_env("Crawler")
class Crawler(Env):
    obs_names = ("hull angle", "hull ang. vel.", "vel. x", "vel. y", *JOINTS,
                 *(f"{joint} speed" for joint in JOINTS), *(f"foot {leg}" for leg in LEGS))
    action_names = JOINTS
    # (for DQN: the joints of a pair together at full torque, + or -: back hips, back
    # knees, front hips, front knees)
    discrete_actions, discrete_names = bang_bang([[0, 2], [1, 3], [4, 6], [5, 7]], 8)
    max_steps = 600
    view_center, view_zoom = (0.0, 1.5), 2.5
    suggested = {"quick": {"cem": {}, "es": {}},
                 "medium": {"es": {}, "cem": {}, "sac_torch": {"updates": 4}},
                 "long": {"ppo_torch": {"hidden": 64}, "sac_torch": {"updates": 4}, "es": {}}}

    def build_scenery(self):
        ruler(self)  # (the ground: a ruler, you see how far it crawled)

    def build_agent(self, i):
        hull = self.body((0, HIP_Y + HULL[1]), b2d.box(*HULL), density=5.0, color=DARK_COLOR)
        joints, feet = [], []
        for x in LEGS.values():
            # capsules whose round ends are centered on the joints (hip, knee): they line up
            thigh = self.body((x, HIP_Y - THIGH), b2d.capsule((0, -THIGH), (0, THIGH), 0.07), density=3.0)
            shin = self.body((x, HIP_Y - 2 * THIGH - SHIN), b2d.capsule((0, -SHIN + 0.05), (0, SHIN), 0.05),
                             density=3.0, color=LIGHT_COLOR)
            joints.append(self.joint(hull, thigh, (x, -HULL[1]), (0, THIGH), "hip"))
            joints.append(self.joint(thigh, shin, (0, -THIGH), (0, SHIN), "knee"))
            feet.append(self.sensor(shin, b2d.circle(radius=0.07, center=(0, -SHIN))))
        return {"hull": hull, "joints": joints, "feet": feet}

    def joint(self, a, b, anchor_a, anchor_b, kind):
        """a revolute joint (anchors: local points of a and b) with a limit and a motor"""
        lower, upper = LIMITS[kind]
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
        self.joints.target_motor_speed = (np.sign(actions) * MAX_SPEED).ravel()
        self.joints.max_motor_torque = np.maximum(np.abs(actions) * MAX_TORQUE, 0.5).ravel()
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
        fallen = (y < FALL_Y) | (np.abs(self.hull.angle) > FALL_ANGLE)
        progressed = x > self.best_x + STALL_DISTANCE
        self.best_x = np.where(progressed, x, self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        return reward - 100 * fallen, fallen | stalled

    def camera_target(self, i):
        """follow the hull to the right (the height stays: no bobbing)"""
        return float(self.hull.position[i, 0]), self.view_center[1]

    def draw(self, debug_draw):
        draw_ruler_numbers(debug_draw)
