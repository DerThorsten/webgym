"""Ragdoll: a human figure (box2d's ragdoll) learns to walk to the right.

Head, torso, upper + lower arms, upper + lower legs, feet: capsules (and the feet)
connected by 11 joints, each with a motor: neck, shoulders, elbows, hips, knees and
ankles. Without a policy the motors are limp and it collapses.
obs: torso angle, angular velocity and velocity, the 11 joint angles and speeds,
foot contacts, the heels' heights above the ground in m (30 values).
action: 11 joint torques in [-1, 1] (each joint's strength: TORQUE).
reward: forward progress, minus a small effort cost; -100 when it falls (the torso
low or tilted). Episodes last 10 s, or end when it makes no progress for 1.5 s.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import register_env
from webgym.envs.walker import Legged

KINDS = ("hip", "knee", "ankle", "shoulder", "elbow")  # per side, then the neck
JOINTS = (*(f"{kind} L" for kind in KINDS), *(f"{kind} R" for kind in KINDS), "neck")
# the feet: wider than box2d's ragdoll (easier to stand on), toes to the right
FOOT = b2d.polygon([(-0.06, -0.19), (0.16, -0.19), (0.16, -0.165), (-0.06, -0.14)], radius=0.015)
HEEL = np.array([-0.06, -0.205])  # (the foot's lowest back point, in the foot's frame)


@register_env("Ragdoll")
class Ragdoll(Legged):
    obs_names = ("torso angle", "torso ang. vel.", "vel. x", "vel. y", *JOINTS,
                 *(f"{joint} speed" for joint in JOINTS), "foot L", "foot R", "heel L", "heel R")
    action_names = JOINTS
    # (no discrete actions, no DQN: 11 joints in a few full-torque groups are too coarse
    # to walk; DQN learned nothing in 3 min)
    # N m at full action, per kind of joint
    TORQUE = {"hip": 15.0, "knee": 15.0, "ankle": 8.0, "shoulder": 8.0, "elbow": 5.0, "neck": 3.0}
    # joint limits (angle of the lower part relative to the upper one; + = forward)
    LIMITS = {"hip": (-0.6, 1.3), "knee": (-1.5, 0.0), "ankle": (-0.5, 0.5),
              "shoulder": (-1.0, 2.0), "elbow": (0.0, 1.5), "neck": (-0.3, 0.3)}
    FALL_Y, FALL_ANGLE = 0.55, 1.0  # the torso lower (standing: 0.68) / tilted more: fallen
    view_center, view_zoom = (0.0, 1.0), 2.0
    suggested = {"quick": {"cem": {}},
                 "medium": {"cem": {}, "ppo_torch": {"hidden": 64}},
                 "long": {"ppo_torch": {"hidden": 64}, "cem": {}}}

    def build_agent(self, i):
        # heights in m above the ground (about 1.35 m tall); "hull" is the torso (the
        # name Legged uses for the main body)
        torso = self.body((0, 0.68), b2d.capsule((0, 0), (0, 0.37), 0.1), density=10.0)
        head = self.body((0, 1.2), b2d.capsule((0, -0.04), (0, 0.04), 0.075), density=10.0)
        joints, feet, soles = [], [], []
        for side in "LR":
            thigh = self.body((0, 0.5), b2d.capsule((0, -0.125), (0, 0.125), 0.06), density=10.0)
            shin = self.body((0, 0.2), b2d.capsule((0, -0.155), (0, 0.125), 0.045), density=10.0)
            foot = self.body((0, 0.2), FOOT, density=10.0, friction=1.0)
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
        return {"hull": torso, "joints": joints, "feet": feet, "soles": soles}

    def observe(self):
        # the heels: the foot bodies' HEEL point in the world (the ground is at y = 0)
        y, angle = self.soles.position[:, 1], self.soles.angle
        heel = y + np.sin(angle) * HEEL[0] + np.cos(angle) * HEEL[1]
        return [*super().observe(), heel.reshape(self.n, -1)]

    def pin(self, a, b, y, kind):
        """a joint of kind (LIMITS) between bodies a and b at height y"""
        return self.joint(a, b, a.local_point((0, y)), b.local_point((0, y)), kind)

    def act(self, actions):
        # like Legged, but limp at action 0 (no holding torque): untrained, it collapses
        torque = np.array([self.TORQUE[name.split()[0]] for name in JOINTS])
        self.joints.target_motor_speed = (np.sign(actions) * self.MAX_SPEED).ravel()
        self.joints.max_motor_torque = (np.abs(actions) * torque).ravel()
        self.effort = np.square(actions).sum(axis=1)
