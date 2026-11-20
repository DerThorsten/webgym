"""Ragdoll: a human figure (box2d's ragdoll) learns to walk to the right.

Head, torso, upper + lower arms, upper + lower legs, feet: capsules (and the feet)
connected by 11 joints, each with a motor: neck, shoulders, elbows, hips, knees and
ankles. Without a policy the motors are limp and it collapses.

Observations (30):
- the torso's angle, angular velocity and velocity x, y
- the 11 joints' angles and speeds
- the two feet's ground contacts
- the two heels' heights above the ground (m)

Actions (11):
- the torques at the 11 joints, each in [-1, 1] (each joint's strength: TORQUE)

Reward: forward progress, minus a small effort cost, a little for every step it stays
upright; -100 for a fall (the torso low or tilted).

Episode end: a fall, no progress for 1.5 s, or after 10 s.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import DARK_COLOR, LIGHT_COLOR, Env, draw_ruler_numbers, register_env, ruler

KINDS = ("hip", "knee", "ankle", "shoulder", "elbow")  # per side, then the neck
JOINTS = (*(f"{kind} L" for kind in KINDS), *(f"{kind} R" for kind in KINDS), "neck")
# N m at full action, per kind of joint
TORQUE = {"hip": 15.0, "knee": 15.0, "ankle": 8.0, "shoulder": 8.0, "elbow": 5.0, "neck": 3.0}
MAX_SPEED = 20.0  # rad/s the motors drive towards
# joint limits (angle of the lower part relative to the upper one; + = forward)
LIMITS = {"hip": (-0.6, 1.3), "knee": (-1.5, 0.0), "ankle": (-0.5, 0.5),
          "shoulder": (-1.0, 2.0), "elbow": (0.0, 1.5), "neck": (-0.3, 0.3)}
FALL_Y, FALL_ANGLE = 0.55, 1.0  # the torso lower (standing: 0.68) / tilted more: fallen
# reward per step with the torso upright (tilted less than UPRIGHT_ANGLE): standing pays a
# little, as in Gymnasium's Walker2d / Humanoid ("healthy reward"); CEM learns to walk far
# more often with it (8 seeds: median 8.6 m instead of 1.3 m, with the lighter top and the
# grip below)
UPRIGHT_BONUS, UPRIGHT_ANGLE = 0.05, 0.5
TOP_DENSITY = 6.0  # torso and head (the limbs: 10): a light top tips over less easily
GRIP = 1.5  # the feet's friction
# no progress (the torso did not get STALL_DISTANCE further right than ever before) for
# STALL_SECONDS: the episode ends (no penalty), the population does not wait for it
STALL_SECONDS, STALL_DISTANCE = 1.5, 0.2
# the feet: wider than box2d's ragdoll (easier to stand on), toes to the right
FOOT = b2d.polygon([(-0.06, -0.19), (0.16, -0.19), (0.16, -0.165), (-0.06, -0.14)], radius=0.015)
HEEL = np.array([-0.06, -0.205])  # (the foot's lowest back point, in the foot's frame)


@register_env("Ragdoll")
class Ragdoll(Env):
    obs_names = ("torso angle", "torso ang. vel.", "vel. x", "vel. y", *JOINTS,
                 *(f"{joint} speed" for joint in JOINTS), "foot L", "foot R", "heel L", "heel R")
    action_names = JOINTS
    # (no discrete actions, no DQN: 11 joints in a few full-torque groups are too coarse
    # to walk; DQN learned nothing in 3 min)
    max_steps = 600
    view_center, view_zoom = (0.0, 1.0), 2.0
    suggested = {"quick": {"cem": {}},
                 "medium": {"cem": {}, "ppo_torch": {"hidden": 64}},
                 "long": {"ppo_torch": {"hidden": 64}, "cem": {}, "neat": {}}}

    def build_scenery(self):
        ruler(self)  # (the ground: a ruler, you see how far it walked)

    def build_agent(self, i):
        # heights in m above the ground (about 1.35 m tall)
        torso = self.body((0, 0.68), b2d.capsule((0, 0), (0, 0.37), 0.1), density=TOP_DENSITY, color=DARK_COLOR)
        head = self.body((0, 1.2), b2d.capsule((0, -0.04), (0, 0.04), 0.075), density=TOP_DENSITY,
                         color=LIGHT_COLOR)
        joints, feet, soles = [], [], []
        for side in "LR":
            # (the legs' capsules: their round ends centered on the joints, hip 0.63, knee
            # 0.355, ankle 0.05: they line up; the arms' do already)
            thigh = self.body((0, 0.5), b2d.capsule((0, 0.355 - 0.5), (0, 0.63 - 0.5), 0.06), density=10.0)
            shin = self.body((0, 0.2), b2d.capsule((0, 0.05 - 0.2), (0, 0.355 - 0.2), 0.045), density=10.0)
            foot = self.body((0, 0.2), FOOT, density=10.0, friction=GRIP, color=DARK_COLOR)
            soles.append(foot)
            # (the sensor: the whole sole, heel to toes, and a little below it)
            feet.append(self.sensor(foot, b2d.box(0.13, 0.04, center=(0.05, -0.18))))
            upper = self.body((0, 0.955), b2d.capsule((0, -0.125), (0, 0.125), 0.035), density=10.0)
            lower = self.body((0, 0.705), b2d.capsule((0, -0.125), (0, 0.125), 0.03), density=10.0)
            # in the order of KINDS
            joints += [self.pin(torso, thigh, 0.63, "hip"), self.pin(thigh, shin, 0.355, "knee"),
                       self.pin(shin, foot, 0.05, "ankle"), self.pin(torso, upper, 1.08, "shoulder"),
                       self.pin(upper, lower, 0.83, "elbow")]
        joints.append(self.pin(torso, head, 1.13, "neck"))
        return {"torso": torso, "joints": joints, "feet": feet, "soles": soles}

    def pin(self, a, b, y, kind):
        """a joint of kind (LIMITS) between bodies a and b at height y, with a motor"""
        lower, upper = LIMITS[kind]
        return self.world.create_revolute_joint(
            body_a=a, body_b=b,
            local_frame_a=b2d.transform(p=a.local_point((0, y))), local_frame_b=b2d.transform(p=b.local_point((0, y))),
            enable_limit=True, lower_angle=lower, upper_angle=upper, enable_motor=True,
        )

    def randomize(self, rng):
        self.torso.linear_velocity = np.column_stack([rng.uniform(-0.2, 0.2, self.n), np.zeros(self.n)])
        self.effort = np.zeros(self.n)
        self.prev_x = self.torso.position[:, 0]
        self.best_x = self.prev_x.copy()  # progress: the farthest x so far
        self.progress_step = np.zeros(self.n, dtype=int)  # step of the last progress

    def act(self, actions):
        # each motor pushes towards +- MAX_SPEED with a torque up to |action| times its
        # TORQUE; limp at action 0 (no holding torque): untrained, it collapses
        torque = np.array([TORQUE[name.split()[0]] for name in JOINTS])
        self.joints.target_motor_speed = (np.sign(actions) * MAX_SPEED).ravel()
        self.joints.max_motor_torque = (np.abs(actions) * torque).ravel()
        self.effort = np.square(actions).sum(axis=1)

    def observe(self):
        n = self.n
        # the heels: the foot bodies' HEEL point in the world (the ground is at y = 0)
        y, angle = self.soles.position[:, 1], self.soles.angle
        heel = y + np.sin(angle) * HEEL[0] + np.cos(angle) * HEEL[1]
        return [self.torso.angle, self.torso.angular_velocity / 2, self.torso.linear_velocity / 3,
                self.joints.current_angle.reshape(n, -1), self.joints.current_speed.reshape(n, -1) / 6,
                self.feet.sensor_capacity.reshape(n, -1) > 0, heel.reshape(n, -1)]

    def reward(self):
        x, y = self.torso.position.T
        upright = np.abs(self.torso.angle) < UPRIGHT_ANGLE
        reward = 10 * (x - self.prev_x) - 0.005 * self.effort + UPRIGHT_BONUS * upright
        self.prev_x = x
        fallen = (y < FALL_Y) | (np.abs(self.torso.angle) > FALL_ANGLE)
        progressed = x > self.best_x + STALL_DISTANCE
        self.best_x = np.where(progressed, x, self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        return reward - 100 * fallen, fallen | stalled

    def camera_target(self, i):
        """follow the torso to the right (the height stays: no bobbing)"""
        return float(self.torso.position[i, 0]), self.view_center[1]

    def draw(self, debug_draw):
        draw_ruler_numbers(debug_draw)
