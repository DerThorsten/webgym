"""Crawler: a four-legged robot learns to crawl to the right.

A long, low hull with two leg pairs (back + front, thigh + shin each), torques at the
8 hips and knees (motors). Hard to tip over, it learns quickly.
obs: hull angle, angular velocity and velocity, the 8 joint angles and speeds,
the 4 foot contacts (24 values).
action: 8 joint torques in [-1, 1].
reward: forward progress, minus a small effort cost; -100 when it flips over.
Episodes last 10 s, or end when it makes no progress for 1.5 s.

Keys (human controller, both legs of a pair together): q / a = back hips,
w / s = back knees, o / l = front hips, p / ; = front knees.
"""

from webgym.core import register_env
from webgym.envs.walker import Legged, bang_bang, legged_names


@register_env("Crawler")
class Crawler(Legged):
    obs_names, action_names = legged_names(("back L", "back R", "front L", "front R"))
    # (for DQN: the joints of a pair together at full torque, + or -: back hips, back
    # knees, front hips, front knees)
    discrete_actions, discrete_names = bang_bang([[0, 2], [1, 3], [4, 6], [5, 7]], 8)
    HULL = (0.6, 0.12)
    LEG_XS = (-0.5, -0.5, 0.5, 0.5)
    HIP_Y = 0.8
    THIGH, SHIN = 0.2, 0.2
    MAX_TORQUE = 8.0
    LIMITS = {"hip": (-1.0, 1.0), "knee": (-1.5, 0.05)}
    FALL_Y, FALL_ANGLE = 0.25, 1.5  # (only flipping over ends it)
    view_center = (0.0, 1.5)
    view_zoom = 2.5
    keys = "back hips Q / A, back knees W / S · front hips O / L, front knees P / ;"
    KEY_PAIRS = (("q", "a"), ("w", "s")) * 2 + (("o", "l"), ("p", ";")) * 2
