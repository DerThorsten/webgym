"""Walker: a two-legged robot learns to walk to the right.

A hull with two legs (thigh + shin), driven by motors at the hips and knees.
obs: hull angle, angular velocity and velocity, the 4 joint angles and speeds,
foot contacts (14 values).
action: 4 motor torques in [-1, 1] (hips, knees).
reward: forward progress, minus a small effort cost; -100 when the hull hits
the ground. Episodes last 10 s.

Keys (human controller): q / a = left hip, w / s = left knee,
o / l = right hip, p / ; = right knee.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Box, Env, new_world, register_env

HIP_Y = 1.2  # standing height of the hips
THIGH, SHIN = 0.32, 0.3  # half lengths
MOTOR_SPEED = 6.0  # rad / s at full action
MAX_TORQUE = 60.0


@register_env("Walker")
class Walker(Env):
    observation_space = Box(-np.inf, np.inf, (14,))
    action_space = Box(-1.0, 1.0, (4,))
    max_steps = 600
    view_center = (0.0, 2.0)
    view_zoom = 3.5

    def __init__(self):
        world = new_world(gravity=(0, -10))
        ground = world.create_static_body(position=(0, 0))
        ground.create_shape(
            b2d.shape_def(material=b2d.surface_material(friction=1.0)),
            b2d.box(200, 0.5, center=(190, -0.5)),
        )
        # the walker's parts don't collide with each other (same negative group)
        parts = b2d.shape_def(density=1.0, filter=b2d.make_filter(group_index=-1),
                              material=b2d.surface_material(friction=1.0))
        self.hull = world.create_dynamic_body(position=(0, HIP_Y + 0.15))
        self.hull.create_shape(parts, b2d.box(0.4, 0.15))
        self.joints, self.feet = [], []
        for _ in range(2):
            thigh = world.create_dynamic_body(position=(0, HIP_Y - THIGH))
            thigh.create_shape(parts, b2d.box(0.07, THIGH))
            shin = world.create_dynamic_body(position=(0, HIP_Y - 2 * THIGH - SHIN))
            shin.create_shape(parts, b2d.capsule((0, -SHIN + 0.05), (0, SHIN - 0.05), 0.05))
            self.joints.append(self._motor(world, self.hull, thigh, (0, -0.15), (0, THIGH), -0.8, 1.1))
            self.joints.append(self._motor(world, thigh, shin, (0, -THIGH), (0, SHIN), -1.5, 0.05))
            self.feet.append(shin)
        self.world = world
        self.start = world.get_snapshot()

    @staticmethod
    def _motor(world, a, b, anchor_a, anchor_b, lower, upper):
        """a revolute joint with a limit and a motor (driven by the actions)"""
        return world.create_revolute_joint(
            body_a=a,
            body_b=b,
            local_frame_a=b2d.transform(p=anchor_a),
            local_frame_b=b2d.transform(p=anchor_b),
            enable_limit=True,
            lower_angle=lower,
            upper_angle=upper,
            enable_motor=True,
            max_motor_torque=MAX_TORQUE,
        )

    def focus(self):
        """the camera follows the hull"""
        return self.hull.position

    def _reset(self, rng):
        self.world.restore_from_snapshot(self.start)
        self.hull.linear_velocity = (rng.uniform(-0.2, 0.2), 0)
        self.prev_x = self.hull.position[0]
        self.effort = 0.0

    def _apply(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1, 1)
        # torque control via the motor: full speed in the direction of the action,
        # torque limited by its size
        for joint, ai in zip(self.joints, a):
            joint.target_motor_speed = MOTOR_SPEED * float(np.sign(ai))
            joint.max_motor_torque = MAX_TORQUE * float(abs(ai))
        self.effort = float(np.square(a).sum())

    def _feet_down(self):
        return [float(foot.world_point((0, -SHIN))[1] < 0.08) for foot in self.feet]

    def _observe(self):
        vx, vy = self.hull.linear_velocity
        return np.array(
            [self.hull.angle, self.hull.angular_velocity / 2, vx / 3, vy / 3,
             *(j.current_angle for j in self.joints),
             *(j.current_speed / MOTOR_SPEED for j in self.joints),
             *self._feet_down()],
            dtype=np.float32,
        )

    def _outcome(self):
        x = self.hull.position[0]
        reward = 10 * (x - self.prev_x) - 0.005 * self.effort
        self.prev_x = x
        fallen = self.hull.position[1] < 0.55 or abs(self.hull.angle) > 1.0
        if fallen:
            return reward - 100, True
        return reward, False

    def human_action(self, keys):
        """two keys per motor (see the module docstring)"""
        pairs = [("q", "a"), ("w", "s"), ("o", "l"), ("p", ";")]
        return np.array([(plus in keys) - (minus in keys) for plus, minus in pairs], dtype=np.float32)
