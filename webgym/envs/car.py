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
obs (16 values):
- angle: how far the chassis is tilted (radians, 0 = level, + = nose up),
- ang. vel.: how fast it tilts (radians per second),
- vel. x, vel. y: how fast it moves forward / up (m/s, divided by 5),
- sees 5° .. -60°: what lies ahead. 4 rays (its eyes) start at the front of the chassis,
  fanned out from a bit upward (5°) to steeply down (-60°), turning with the chassis;
  each value is how far the first thing it hits (ground, a box, a crate, gravel) is, as
  a share of 8 m: small before a box, 1 when it sees nothing,
- ground +1.5 m .. +6 m: the terrain ahead. 7 probes look straight down (not turning
  with the car) every 0.75 m, 1.5 to 6 m ahead of it: the height of the ground there relative to
  the chassis, in units of 2 m: about -0.3 on flat ground, higher before a box or the
  ledge, about -2.5 over a gap (nothing below),
- can jump: 1 when a jump is possible now (a wheel on something, recharged), else 0.
action: [throttle (-1: full back .. 1: full forward), jump (> 0.5: jump, when it can)].
(Decoration on: eyes (yellow) and probes (blue) are drawn, you see what the car sees.)
reward: forward progress; +100 at the flag; -100 when it flips over or falls into a
gap. Episodes end there, after 40 s, or when it makes no progress for 3 s.
"""

import math

import numpy as np
import pyb2d3 as b2d

from webgym.core import DECORATION_FILTER, SCENERY, Env, register_env

GOAL = 105.0
# the course (it can jump: the higher boxes can't be climbed, only jumped)
GROUND = [(-5, 12, 0), (14, 26, 0), (32, 46, 0), (48.5, 58, 0), (58, 64, 1.0), (66.5, 84, 0), (86.5, 115, 0)]
# (from x, to x, height of the top): gaps / the pit between
RAMP = (52, 58, 1.0)  # up to the ledge (from x, to x, height); after the ledge: a gap and a drop
LOW_BAR = (71, 77, 1.55)  # a bar to drive under (from x, to x, its lower edge): don't jump there
PIT = (26, 32, 0.8)  # the gravel pit: from x, to x, depth
GRAVEL = 0.12  # the pebbles' size (radius)
PEBBLES = [  # their shapes (mixed: round ones roll, edged ones lock into each other)
    b2d.circle(radius=GRAVEL),
    b2d.box(0.8 * GRAVEL, 0.8 * GRAVEL),
    b2d.polygon([(-GRAVEL, -0.8 * GRAVEL), (GRAVEL, -0.8 * GRAVEL), (0, GRAVEL)]),
]
STEPS = [(8, 0.35), (22, 0.45), (40, 0.45), (93, 0.6), (99, 0.7)]  # (x, height) of the fixed boxes
CRATES = [17, 19, 37, 79, 81, 90, 103]  # x of the loose crates
WHEEL_RADIUS = 0.35
# (x, height) of the wheels' axles below the chassis; the front wheel sits higher: it
# climbs onto boxes and lifts the rest after it
WHEELS = [(-0.75, 0.3), (-0.25, 0.3), (0.25, 0.3), (0.85, 0.15)]
# the car's eyes: rays from the front of the chassis, fanned out from a bit upward to
# steeply down (degrees, relative to the chassis: they turn with it), each sees up to
# RAY_RANGE m
EYES = np.array([5, -15, -35, -60])
RAY_RANGE = 8.0
EYE = (1.0, 0.15)  # where they start (on the chassis)
# the ground probes: straight down (not turning with the car) at these distances ahead
# (m), from PROBE_UP m above the chassis, PROBE_DEPTH m deep: the height of the ground
# there (a box, the ledge, a gap)
PROBES = 1.5 + 0.75 * np.arange(7)  # (1.5 .. 6 m; closer than the narrowest box, 0.8 m)
PROBE_UP, PROBE_DEPTH = 0.9, 6.0
PROBE_COLOR = 0x7FD1F2  # (decoration: the probes drawn)
MAX_SPEED, MAX_TORQUE = 30.0, 30.0  # of the wheel motors at full throttle
STEADY = 3.0  # the chassis' angular damping: it resists tipping over (a wheelie, a jump)
# the jump (the kick leg under the chassis): how far the foot goes out, how fast, how
# hard, for how long; then it recharges
KICK_LENGTH, KICK_SPEED, KICK_FORCE, KICK_SECONDS = 0.6, 8.0, 500.0, 0.12  # (~0.7 m high)
JUMP_RECHARGE = 1.0  # s until the next jump
RAY_COLOR = 0xF2D24B  # (decoration: the rays drawn)
SEES = b2d.make_query_filter(category_bits=1, mask_bits=SCENERY)  # (the world, not the car itself)


@register_env("Car")
class Car(Env):
    obs_names = ("angle", "ang. vel.", "vel. x", "vel. y", *(f"sees {d:g}°" for d in EYES),
                 *(f"ground +{dx:g} m" for dx in PROBES), "can jump")
    action_names = ("throttle", "jump")
    discrete_actions = np.array([[0, 0], [0.3, 0], [0.6, 0], [1, 0], [-0.5, 0], [0.6, 1], [1, 1]])
    discrete_names = ("roll", "slow", "medium", "full", "back", "medium + jump", "full + jump")
    batchable = False  # (the car moves the crates)
    max_steps = 2400
    view_center, view_zoom = (4.0, 2.5), 5.0
    suggested = {"quick": {"cem": {}},
                 "medium": {"cem": {}, "neat": {}},
                 "long": {"cem": {}, "ppo_torch": {"hidden": 64}, "sac_torch": {}}}

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
        chassis = self.body((0, y), b2d.box(1.0, 0.25), density=8.0,  # (heavy, damped: it doesn't flip at once)
                            angular_damping=STEADY)
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
        # (no friction: a kick while rolling would brake the bottom and tip the car forward)
        foot = self.body((0, y - 0.3), b2d.box(0.3, 0.05), density=2.0, friction=0.0)
        down = b2d.make_rot(-math.pi / 2)  # (the piston's axis: the x axis of its frames, turned down)
        kicker = self.world.create_prismatic_joint(
            body_a=chassis, body_b=foot, local_frame_a=b2d.transform(p=(0, -0.3), q=down),
            local_frame_b=b2d.transform(p=(0, 0), q=down), enable_limit=True,
            lower_translation=0.0, upper_translation=KICK_LENGTH, enable_motor=True, max_motor_force=50.0,
        )
        if self.decorated:  # eyes and probes, as lines (looks only: nothing hits them, rays don't see them)
            lines = self.world.create_static_body()
            for color, count in ((RAY_COLOR, len(EYES)), (PROBE_COLOR, len(PROBES))):
                line_def = b2d.shape_def(filter=DECORATION_FILTER, material=b2d.surface_material(custom_color=color))
                self.ray_lines += [lines.create_shape(line_def, b2d.segment((0, 0), (0, -0.01))) for _ in range(count)]
        return {"chassis": chassis, "axles": axles, "wheels": wheels, "feet": feet, "kicker": kicker}

    def randomize(self, rng):
        x = np.array(CRATES) + rng.uniform(-1, 1, len(CRATES))
        self.crates.position = np.column_stack([x, np.full(len(CRATES), 0.35)])
        # the gravel: the pebbles in new places (another mix of shapes every episode)
        self.gravel.position = self.gravel_places[rng.permutation(len(self.gravel_places))]
        self.prev_x = self.chassis.position[:, 0]
        self.best_x, self.progress_step = self.prev_x.copy(), np.zeros(self.n, dtype=int)
        self.jumped_at = np.full(self.n, -JUMP_RECHARGE / self.dt)  # (step of the last jump)

    def on_ground(self):
        """(n,) bool: a wheel on something (its sensor)"""
        return (self.feet.sensor_capacity.reshape(self.n, -1) > 0).any(axis=1)

    def can_jump(self):
        """(n,) bool: a wheel on something and the jump recharged"""
        return self.on_ground() & (self.steps - self.jumped_at >= JUMP_RECHARGE / self.dt)

    def act(self, actions):
        # the motors drive towards +- MAX_SPEED with a torque up to |throttle| * MAX_TORQUE;
        # in the air they rest (spinning the wheels up there would tip the car over)
        throttle = np.repeat(actions[:, 0] * self.on_ground(), len(WHEELS))
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
                *self._senses(), self.can_jump()]

    def cast(self, i, start, way):
        """how far along `way` (from `start`) car i sees the first thing of the world:
        0 .. 1 of the way (1: nothing)"""
        hit = self.world.cast_ray_closest(start, way, SEES)
        return hit.fraction if hit.hit else 1.0

    def _senses(self):
        """what the cars see -> eyes (n, len(EYES)): along each eye ray (from the front,
        turning with the chassis) how far the first thing is, 0 .. 1 of RAY_RANGE; ground
        (n, len(PROBES)): the height of the ground ahead (probes straight down) relative to
        the chassis, in units of 2 m (a gap: about -2.5). The rays are kept for drawing."""
        eyes, ground = np.ones((self.n, len(EYES))), np.zeros((self.n, len(PROBES)))
        self.rays = []
        for i, ((x, y), angle) in enumerate(zip(self.chassis.position, self.chassis.angle)):
            c, s = np.cos(angle), np.sin(angle)
            start = (float(x + c * EYE[0] - s * EYE[1]), float(y + s * EYE[0] + c * EYE[1]))
            for k, ray in enumerate(np.radians(EYES) + angle):
                way = (RAY_RANGE * float(np.cos(ray)), RAY_RANGE * float(np.sin(ray)))
                eyes[i, k] = self.cast(i, start, way)
                self.rays.append((start, (start[0] + eyes[i, k] * way[0], start[1] + eyes[i, k] * way[1])))
            for k, dx in enumerate(PROBES):
                top = (float(x + dx), float(y + PROBE_UP))
                depth = self.cast(i, top, (0.0, -PROBE_DEPTH)) * PROBE_DEPTH
                ground[i, k] = (PROBE_UP - depth) / 2
                self.rays.append((top, (top[0], top[1] - depth)))
        return eyes, ground

    def decorate(self):
        """what the car sees: its eyes and probes as lines, to what they hit (cast again
        here: after an episode the world goes on without observations)"""
        self._senses()
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

    def camera_target(self, i):
        return float(self.chassis.position[i, 0]), self.view_center[1]
