"""Drone: a quadcopter (seen from the side) learns to fly through a course of gates.

A light frame with a rotor at each end; each rotor's thrust is an action. More thrust
on one side tilts the drone, and tilted, its thrust pushes it sideways, so it flies.
The gates hang at different heights. Their posts are solid, the ground too, and
touching them is a crash.

Observations (9):
- where the next gate is (x, y relative to the drone), and the one after it
- the drone's velocity x, y
- its tilt (sin, cos) and spin

Actions (2):
- the thrust of the left and of the right rotor (-1: off .. 1: full; both at 0 about
hold the drone in the air)

Reward: getting closer to the next gate, +10 for each gate passed, -100 for a crash; a
little for every second it stays in the air.

Episode end: a crash, the last gate passed, no closer to its gate for 3 s (hovering in
place), or after 20 s.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import DARK_COLOR, SENSOR_FILTER, Env, draw_ruler_numbers, register_env, ruler

GATES = 10
GATE_GAP = 6.0  # m between gates
OPENING = 1.6  # the gate's opening (m)
GATE_Y = np.random.default_rng(5).uniform(2.0, 7.0, GATES)  # the openings' centers (the same every episode)
GATE_X = GATE_GAP * (np.arange(GATES) + 1)
ARM = 0.35  # the rotors' distance from the center
MASS_FRAME = 0.8  # kg (with the rotors: about 1 kg)
MAX_THRUST = 2.0  # of the drone's weight, both rotors together at full power
TOP = 12.0  # above this: lost
STALL_SECONDS, STALL_DISTANCE = 3.0, 0.3  # no closer to the gate by this for that long: the end


@register_env("Drone")
class Drone(Env):
    obs_names = ("gate dx", "gate dy", "next gate dx", "next gate dy", "vel. x", "vel. y", "tilt sin", "tilt cos",
                 "spin")
    action_names = ("left rotor", "right rotor")
    # (no discrete actions, no DQN: flying needs fine thrust; with 4 fixed settings DQN
    # learned nothing in 3 min)
    max_steps = 1200
    view_center, view_zoom = (0.0, 4.3), 5.0  # (low enough for the ruler's numbers)
    # (PPO: mean 420 in 1 min natively; CEM: 60 in 2 min)
    suggested = {"quick": {"ppo_torch": {"hidden": 64}, "cem": {}},
                 "medium": {"ppo_torch": {"hidden": 64}, "cem": {}},
                 "long": {"ppo_torch": {"hidden": 64}, "sac_torch": {}, "cem": {}}}

    def build_scenery(self):
        ruler(self)  # (the ground: a ruler, you see how far it flew)
        for x, y in zip(GATE_X, GATE_Y):  # each gate: a post from the ground, one hanging from above
            low, high = y - OPENING / 2, y + OPENING / 2
            self.scenery(b2d.box(0.1, low / 2, center=(x, low / 2)))
            self.scenery(b2d.box(0.1, (TOP - high) / 2, center=(x, (TOP + high) / 2)))

    def build_agent(self, i):
        frame = self.body((0, 3.0), b2d.box(ARM, 0.05), density=MASS_FRAME / (4 * ARM * 0.05), color=DARK_COLOR)
        for x in (-ARM, ARM):  # the rotors (looks: small round heads)
            self.shape(frame, b2d.circle(center=(x, 0.08), radius=0.07), density=0.1 / (np.pi * 0.07**2))
        # (a crash: the frame touches the ground or a gate; a sensor around it)
        hit = self.sensor(frame, b2d.box(ARM + 0.08, 0.17, center=(0, 0.03)), filter=SENSOR_FILTER, density=0.0)
        return {"frame": frame, "hit": hit}

    def randomize(self, rng):
        n = self.n
        self.frame.linear_velocity = np.column_stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n)])
        self.frame.angular_velocity = rng.uniform(-0.3, 0.3, n)
        self.gate = np.zeros(n, dtype=int)  # the next gate
        self.prev_dist = self._dist()
        self.best_dist, self.progress_step = self.prev_dist.copy(), np.zeros(n)

    def act(self, actions):
        weight = self.frame[0].mass * -self.gravity[1]
        # (action 0: half of the full thrust per rotor = hover)
        thrust = (actions + 1) / 2 * MAX_THRUST * weight / 2  # (n, 2): left, right
        angle = self.frame.angle
        up = np.column_stack([-np.sin(angle), np.cos(angle)])
        pos = self.frame.position
        side = np.column_stack([np.cos(angle), np.sin(angle)]) * ARM
        for k, sign in enumerate((-1, 1)):
            self.frame.apply_force(up * thrust[:, k:k + 1], pos + sign * side)

    def _next(self, k=0):
        """(n, 2): where gate `gate + k` is (the last gate after the end)"""
        g = np.minimum(self.gate + k, GATES - 1)
        return np.column_stack([GATE_X[g], GATE_Y[g]])

    def _dist(self):
        return np.linalg.norm(self._next() - self.frame.position, axis=1)

    def observe(self):
        pos = self.frame.position
        angle = self.frame.angle
        return [(self._next() - pos) / 5, (self._next(1) - pos) / 5, self.frame.linear_velocity / 5,
                np.sin(angle), np.cos(angle), self.frame.angular_velocity / 5]

    def reward(self):
        pos = self.frame.position
        progress = self.prev_dist - self._dist()  # (towards the gate it was flying to)
        # passing a gate: crossing its x (through the opening: else the post was hit)
        passed = (pos[:, 0] > GATE_X[np.minimum(self.gate, GATES - 1)]) & (self.gate < GATES)
        self.gate = self.gate + passed
        reward = 5 * progress + 10 * passed + 0.01
        self.prev_dist = self._dist()
        # (stalled: no closer to its gate for STALL_SECONDS; a new gate starts afresh)
        closer = passed | (self.prev_dist < self.best_dist - STALL_DISTANCE)
        self.best_dist = np.where(closer, self.prev_dist, self.best_dist)
        self.progress_step = np.where(closer, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        crashed = (self.hit.sensor_capacity > 0) | (pos[:, 1] > TOP) | (pos[:, 0] < -5)
        done = crashed | stalled | (self.gate >= GATES)
        return np.where(crashed, -100.0, reward), done

    def camera_target(self, i):
        """follow the drone sideways (the height stays: the course fits)"""
        return float(self.frame.position[i, 0]) + 2.0, self.view_center[1]

    def draw(self, debug_draw):
        draw_ruler_numbers(debug_draw)
