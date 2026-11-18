"""Car: drive to the flag over boxes, gaps, a gravel pit, a ledge and under a bar.

A car (a chassis on four sprung wheels with strong motors) drives over steps, loose
crates, gaps, through a deep gravel pit (loose pebbles of mixed shapes that give way
under the wheels), up a ramp onto a ledge and down over a gap, under a low bar (jump
there and it hits its roof) and over high boxes that only a jump gets it over.
Its motors are strong enough to wheelie over boxes and jump the gaps, but too much
throttle flips it over: the agent has to dose it. The car pushes the crates and the
gravel around (new places every episode), so the agents change their world and each one
needs a world of its own (not batchable: slower, fewer agents, the view shows the best
one).
It can also jump (once a second, with a wheel on something): a kick leg under the
chassis (a foot on a piston) shoots down against the ground, the ground pushes the car
up: over a crate, a box, a gap.
obs (14 values):
- angle: how far the chassis is tilted (radians, 0 = level, + = nose up),
- ang. vel.: how fast it tilts (radians per second),
- vel. x, vel. y: how fast it moves forward / up (m/s, divided by 5),
- sees 5° .. -80°: what lies ahead. 9 rays start at the front of the chassis, fanned out
  from a bit upward (5°) to steeply down (-80°), turning with the chassis; each
  value is how far the first thing it hits (ground, a box, a crate, gravel) is, as a
  share of 8 m: small before a box, larger when the ground falls away, 1 when it sees
  nothing (into a gap, or over the edge),
- can jump: 1 when a jump is possible now (a wheel on something, recharged), else 0.
action: [throttle (-1: full back .. 1: full forward), jump (> 0.5: jump, when it can)].
(Decoration on: the rays are drawn, you see what the car sees.)
reward: forward progress; +100 at the flag; -100 when it flips over or falls into a
gap. Episodes end there, after 40 s, or when it makes no progress for 3 s.

Keys (human controller): left / right = throttle, space = jump.
"""

import math

import numpy as np
import pyb2d3 as b2d

from webgym.core import DECORATION_FILTER, SCENERY, Env, register_env

GOAL = 105.0
# the course (it can jump: the higher boxes can't be climbed, only jumped)
GROUND = [(-5, 12, 0), (15, 26, 0), (35, 46, 0), (50, 58, 0), (58, 64, 1.5), (68, 84, 0), (88, 115, 0)]
# (from x, to x, height of the top): gaps / the pit between
RAMP = (52, 58, 1.5)  # up to the ledge (from x, to x, height); after the ledge: a gap and a drop
LOW_BAR = (71, 77, 1.55)  # a bar to drive under (from x, to x, its lower edge): don't jump there
PIT = (26, 35, 1.4)  # the gravel pit: from x, to x, depth
GRAVEL = 0.12  # the pebbles' size (radius)
PEBBLES = [  # their shapes (mixed: round ones roll, edged ones lock into each other)
    b2d.circle(radius=GRAVEL),
    b2d.box(0.8 * GRAVEL, 0.8 * GRAVEL),
    b2d.polygon([(-GRAVEL, -0.8 * GRAVEL), (GRAVEL, -0.8 * GRAVEL), (0, GRAVEL)]),
]
STEPS = [(8, 0.5), (22, 0.9), (40, 0.8), (93, 1.0), (99, 1.2)]  # (x, height) of the fixed boxes
CRATES = [17, 19, 37, 79, 81, 90, 103]  # x of the loose crates
WHEEL_RADIUS = 0.35
# (x, height) of the wheels' axles below the chassis; the front wheel sits higher: it
# climbs onto boxes and lifts the rest after it
WHEELS = [(-0.75, 0.3), (-0.25, 0.3), (0.25, 0.3), (0.85, 0.15)]
# the car's eyes: rays from the front of the chassis, fanned out from a bit upward to
# steeply down (degrees, relative to the chassis: they turn with it), each sees up to
# RAY_RANGE m
RAYS = np.array([5, -5, -15, -25, -35, -45, -55, -65, -80])
RAY_RANGE = 8.0
EYE = (1.0, 0.15)  # where they start (on the chassis)
MAX_SPEED, MAX_TORQUE = 30.0, 40.0  # of the wheel motors at full throttle
# the jump (the kick leg under the chassis): how far the foot goes out, how fast, how
# hard, for how long; then it recharges
KICK_LENGTH, KICK_SPEED, KICK_FORCE, KICK_SECONDS = 0.6, 8.0, 800.0, 0.12
JUMP_RECHARGE = 1.0  # s until the next jump
RAY_COLOR = 0xF2D24B  # (decoration: the rays drawn)
SEES = b2d.make_query_filter(category_bits=1, mask_bits=SCENERY)  # (the world, not the car itself)


@register_env("Car")
class Car(Env):
    obs_names = ("angle", "ang. vel.", "vel. x", "vel. y", *(f"sees {d:g}°" for d in RAYS), "can jump")
    action_names = ("throttle", "jump")
    discrete_actions = np.array([[0, 0], [0.3, 0], [0.6, 0], [1, 0], [-0.5, 0], [0.6, 1], [1, 1]])
    discrete_names = ("roll", "slow", "medium", "full", "back", "medium + jump", "full + jump")
    batchable = False  # (the car moves the crates)
    max_steps = 2400
    view_center, view_zoom = (4.0, 2.5), 5.0
    keys = "← / → throttle · space jump"

    def build_scenery(self):
        for start, end, top in GROUND:  # (3 m deep: also the walls of the gravel pit)
            self.scenery(b2d.box((end - start) / 2, (top + 3) / 2, center=((start + end) / 2, (top - 3) / 2)),
                         friction=1.0)
        start, end, height = RAMP
        self.scenery(b2d.polygon([(start, 0), (end, 0), (end, height)]), friction=1.0)
        start, end, low = LOW_BAR
        self.scenery(b2d.box((end - start) / 2, 0.15, center=((start + end) / 2, low + 0.15)))
        for x, height in STEPS:
            self.scenery(b2d.box(0.4, height / 2, center=(x, height / 2)))
        # the gravel pit: a floor below the ground, filled with loose pebbles up to the top
        start, end, depth = PIT
        self.scenery(b2d.box((end - start) / 2, 0.5, center=((start + end) / 2, -depth - 0.5)), friction=1.0)
        d = 2 * GRAVEL
        self.gravel_places = np.array([(start + GRAVEL + column * d + (row % 2) * GRAVEL, -depth + GRAVEL + row * d)
                                       for row in range(int(depth / d)) for column in range(int((end - start) / d))])
        self.gravel = b2d.Bodies([self.prop(tuple(p), PEBBLES[k % len(PEBBLES)], density=1.5, friction=0.4)
                                  for k, p in enumerate(self.gravel_places)])
        self.scenery(b2d.box(0.03, 1.0, center=(GOAL, 1.0)), solid=False)  # the flag
        self.scenery(b2d.polygon([(GOAL, 2.0), (GOAL, 1.6), (GOAL + 0.6, 1.8)]), solid=False)
        self.crates = b2d.Bodies([self.prop((x, 0.35), b2d.box(0.35, 0.35)) for x in CRATES])
        self.ray_lines = []  # (decorated: what the car sees, its rays drawn as lines)

    def build_agent(self, i):
        y = WHEEL_RADIUS + 0.3  # (the chassis' height: the rear wheels on the ground)
        chassis = self.body((0, y), b2d.box(1.0, 0.25), density=8.0)  # (heavy: it doesn't flip at once)
        self.shape(chassis, b2d.box(0.45, 0.15, center=(-0.1, 0.4)))  # the cabin
        axles, wheels, feet = [], [], []
        for x, below in WHEELS:
            wheel = self.body((x, y - below), b2d.circle(radius=WHEEL_RADIUS), density=0.5, friction=1.5)
            axles.append(self.world.create_wheel_joint(
                body_a=chassis, body_b=wheel,
                # the suspension axis: up (the x axis of frame a, turned by 90 degrees)
                local_frame_a=b2d.transform(p=(x, -below), q=b2d.make_rot(math.pi / 2)),
                local_frame_b=b2d.transform(p=(0, 0)),
                enable_spring=True, hertz=5.0, damping_ratio=0.7,
                enable_limit=True, lower_translation=-0.2, upper_translation=0.15,
                enable_motor=True,
            ))
            wheels.append(wheel)
            # (a sensor a bit larger than the wheel: it touches the ground, a box, gravel)
            feet.append(self.sensor(wheel, b2d.circle(radius=WHEEL_RADIUS + 0.05)))
        # the kick leg: a foot plate under the chassis on a piston (a prismatic joint
        # pointing down). To jump, its motor shoots the foot against the ground: the ground
        # pushes the car up. Otherwise it pulls the foot in.
        foot = self.body((0, y - 0.3), b2d.box(0.3, 0.05), density=2.0)
        down = b2d.make_rot(-math.pi / 2)  # (the piston's axis: the x axis of its frames, turned down)
        kicker = self.world.create_prismatic_joint(
            body_a=chassis, body_b=foot, local_frame_a=b2d.transform(p=(0, -0.3), q=down),
            local_frame_b=b2d.transform(p=(0, 0), q=down), enable_limit=True,
            lower_translation=0.0, upper_translation=KICK_LENGTH, enable_motor=True, max_motor_force=50.0,
        )
        if self.decorated:  # the rays, as lines (looks only: nothing hits them, rays don't see them)
            lines = self.world.create_static_body()
            line_def = b2d.shape_def(filter=DECORATION_FILTER, material=b2d.surface_material(custom_color=RAY_COLOR))
            self.ray_lines += [lines.create_shape(line_def, b2d.segment((0, 0), (0, -0.01))) for _ in RAYS]
        return {"chassis": chassis, "axles": axles, "wheels": wheels, "feet": feet, "kicker": kicker}

    def randomize(self, rng):
        x = np.array(CRATES) + rng.uniform(-1, 1, len(CRATES))
        self.crates.position = np.column_stack([x, np.full(len(CRATES), 0.35)])
        # the gravel: the pebbles in new places (another mix of shapes every episode)
        self.gravel.position = self.gravel_places[rng.permutation(len(self.gravel_places))]
        self.prev_x = self.chassis.position[:, 0]
        self.best_x, self.progress_step = self.prev_x.copy(), np.zeros(self.n, dtype=int)
        self.jumped_at = np.full(self.n, -JUMP_RECHARGE / self.dt)  # (step of the last jump)

    def can_jump(self):
        """(n,) bool: a wheel on something (its sensor) and the jump recharged"""
        on_ground = (self.feet.sensor_capacity.reshape(self.n, -1) > 0).any(axis=1)
        return on_ground & (self.steps - self.jumped_at >= JUMP_RECHARGE / self.dt)

    def act(self, actions):
        # the motors drive towards +- MAX_SPEED with a torque up to |throttle| * MAX_TORQUE
        throttle = np.repeat(actions[:, 0], len(WHEELS))
        self.axles.target_motor_speed = -np.sign(throttle) * MAX_SPEED  # (- : roll right)
        self.axles.max_motor_torque = np.abs(throttle) * MAX_TORQUE
        # the jump: the kick leg's motor shoots the foot down for KICK_SECONDS (hard, fast),
        # then pulls it back in (gently)
        jump = (actions[:, 1] > 0.5) & self.can_jump()
        self.jumped_at = np.where(jump, self.steps, self.jumped_at)
        kicking = self.steps - self.jumped_at < KICK_SECONDS / self.dt
        self.kicker.target_motor_speed = np.where(kicking, KICK_SPEED, -2.0)
        self.kicker.max_motor_force = np.where(kicking, KICK_FORCE, 50.0)

    def observe(self):
        return [self.chassis.angle, self.chassis.angular_velocity / 2, self.chassis.linear_velocity / 5,
                self._ground_ahead(), self.can_jump()]

    def _ground_ahead(self):
        """what the car sees: along each ray (from its front, turning with the chassis) how
        far the first thing of the world is, 0 .. 1 of RAY_RANGE (1: nothing, e.g. a gap
        below); the rays (start, end) are kept for the drawing"""
        seen = np.ones((self.n, len(RAYS)))
        self.rays = []
        for i, ((x, y), angle) in enumerate(zip(self.chassis.position, self.chassis.angle)):
            c, s = np.cos(angle), np.sin(angle)
            start = (float(x + c * EYE[0] - s * EYE[1]), float(y + s * EYE[0] + c * EYE[1]))
            for k, ray in enumerate(np.radians(RAYS) + angle):
                way = (RAY_RANGE * float(np.cos(ray)), RAY_RANGE * float(np.sin(ray)))
                hit = self.world.cast_ray_closest(start, way, SEES)
                if hit.hit:
                    seen[i, k] = hit.fraction
                end = (start[0] + seen[i, k] * way[0], start[1] + seen[i, k] * way[1])
                self.rays.append((start, end))
        return seen

    def decorate(self):
        """what the car sees: its rays as lines, from its eye to what they hit (cast
        again here: after an episode the world goes on without observations)"""
        self._ground_ahead()
        for line, (start, end) in zip(self.ray_lines, self.rays):
            line.segment = b2d.segment(start, end)

    def reward(self):
        x, y = self.chassis.position.T
        reward = x - self.prev_x
        self.prev_x = x
        crashed = (np.abs(self.chassis.angle) > 1.8) | (y < -2)  # flipped / in a gap
        arrived = x > GOAL
        progressed = x > self.best_x + 0.2
        self.best_x = np.where(progressed, x, self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > 3.0 / self.dt
        return reward - 100 * crashed + 100 * arrived, crashed | arrived | stalled

    def human_action(self, keys):
        return np.array([("right" in keys) - ("left" in keys), float("space" in keys)], dtype=np.float32)

    def camera_target(self, i):
        return float(self.chassis.position[i, 0]), self.view_center[1]
