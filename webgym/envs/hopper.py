"""Hopper: a one-legged robot learns to hop to the right.

A small hull on one leg (thigh, shin and a foot), torques at the hip, knee and
ankle. It cannot walk: it has to jump, and keep its balance while landing.
The hard one: agents learn to balance quickly, hopping takes long (ES got furthest).
obs: hull angle, angular velocity and velocity, the 3 joint angles and speeds,
foot contact (11 values).
action: 3 joint torques in [-1, 1].
reward: forward progress, a small bonus per step upright, minus a small effort
cost; -100 when it falls over.
Episodes last 10 s, or end when it makes no progress for 1.5 s.

Keys (human controller): q / a = hip, w / s = knee, e / d = ankle.
"""

from webgym.core import register_env
from webgym.envs.walker import FOOT_H, Legged


@register_env("Hopper")
class Hopper(Legged):
    HULL = (0.25, 0.12)
    LEG_XS = (0.0,)
    LEG_NAMES = ("",)
    THIGH, SHIN = 0.3, 0.3
    FOOT = 0.2
    HIP_Y = 2 * 0.3 + 2 * 0.3 + 2 * FOOT_H  # (standing on its foot)
    MAX_TORQUE = 6.0  # (more: random first policies kick it over at once, nothing to learn from)
    LIMITS = {"hip": (-1.2, 0.8), "knee": (-1.6, 0.05), "ankle": (-0.6, 0.6)}
    FALL_Y, FALL_ANGLE = 0.8, 0.8
    ALIVE = 0.05  # (without it, standing still is as good as anything)
    keys = "Q / A hip, W / S knee, E / D ankle"
    KEY_PAIRS = (("q", "a"), ("w", "s"), ("e", "d"))
