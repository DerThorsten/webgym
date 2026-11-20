"""Blob: a soft ring of balls learns to roll, crawl and hop to the right.

A ring of 10 small balls, held together by springs: each ball to its neighbours and to
the next but one (it keeps its round shape), and five springs across the ring, each
from a ball to the one opposite. The policy sets the length of the five springs across.
Squeezing the ring one way and stretching it the other tips it over, it rolls, hops or
crawls, whatever works. Drag it around with the mouse, it is soft.
obs: the balls' positions relative to the ring's middle, the middle's velocity, each
ball touching the ground, the lowest ball's height (in ball radii: 1 touching the
ground, more in the air) and the balls' mean height (0 at rest, + higher).
action: per spring across its length (-1: short .. 1: long).
reward: forward progress of the ring's middle, a bonus for every moment in the air (no
ball touching the ground, so jumps pay), minus a small effort cost.
Episodes last 10 s, or end when it makes no progress for 1.5 s.
"""

import math

import numpy as np
import pyb2d3 as b2d

from webgym.core import AGENT, SCENERY, Env, register_env
from webgym.envs.walker import STALL_DISTANCE, STALL_SECONDS, bang_bang, ground_with_ticks

BALLS = 10
ACROSS = BALLS // 2  # the springs across the ring (the actions)
RADIUS = 0.8  # the ring's
BALL = 0.15  # a ball's radius
SPAN = (0.75, 1.25)  # a spring across at action -1 / 1, times the ring's diameter
SPRING = (12.0, 0.4)  # hertz, damping ratio of the springs across (soft, but quick)
RIM_SPRING = (16.0, 0.5)  # of the ring: neighbours, and the next but one (it keeps its shape)
# the balls grip and roll with losses, it spins down: without, a blob once rolling rolls on
# by itself (no effort needed)
FRICTION, ROLLING, DAMPING = 1.5, 0.2, 2.0  # friction, rolling resistance, angular damping
LINE_COLOR, GHOST_LINE_COLOR = 0xF0C080, 0x4FA3FF  # the springs (looks: lines between the balls)
AIR_BONUS = 0.2  # reward per step with no ball on the ground (rolling well: ~0.3 per step)


@register_env("Blob")
class Blob(Env):
    obs_names = (*(f"ball {k + 1} {c}" for k in range(BALLS) for c in ("x", "y")), "vel. x", "vel. y",
                 *(f"ball {k + 1} down" for k in range(BALLS)), "lowest height", "mean height")
    action_names = tuple(f"across {k + 1}" for k in range(ACROSS))
    # (for DQN: every spring across long (+) or short (-), all 32 combinations: "+-+-+")
    discrete_actions, discrete_names = bang_bang([[k] for k in range(ACROSS)], ACROSS)
    max_steps = 600
    view_center, view_zoom = (0.0, 1.5), 3.0
    suggested = {"quick": {"cem": {}, "es": {}},
                 "medium": {"cem": {}, "es": {}, "ppo_torch": {"hidden": 64}},
                 "long": {"ppo_torch": {"hidden": 64}, "cem": {}}}

    def build_scenery(self):
        ground_with_ticks(self)
        # the springs, as pairs of balls (of one blob): drawn as lines
        self.pairs = [(k, (k + step) % BALLS) for step in (1, 2) for k in range(BALLS)]
        self.pairs += [(k, k + ACROSS) for k in range(ACROSS)]

    def build_agent(self, i):
        y0 = RADIUS + BALL + 0.05
        # (its balls bump into each other, not into the other blobs: a group per blob)
        own = b2d.make_filter(category_bits=AGENT, mask_bits=SCENERY, group_index=i + 1)
        balls, down = [], []
        for k in range(BALLS):
            a = 2 * math.pi * k / BALLS
            ball = self.body((RADIUS * math.cos(a), y0 + RADIUS * math.sin(a)), b2d.circle(radius=BALL),
                             density=2.0, friction=FRICTION, filter=own, angular_damping=DAMPING,
                             material={"rolling_resistance": ROLLING})
            down.append(self.sensor(ball, b2d.circle(radius=1.2 * BALL), density=0.0))
            balls.append(ball)
        rim = []
        for step in (1, 2):
            chord = 2 * RADIUS * math.sin(step * math.pi / BALLS)  # (their distance on the ring)
            rim += [self.spring(balls[k], balls[(k + step) % BALLS], chord, *RIM_SPRING) for k in range(BALLS)]
        across = [self.spring(balls[k], balls[k + ACROSS], 2 * RADIUS, *SPRING) for k in range(ACROSS)]
        return {"balls": balls, "rim": rim, "across": across, "down": down}

    def spring(self, a, b, length, hertz, damping):
        """a springy distance joint between the centers of a and b"""
        return self.world.create_distance_joint(
            body_a=a, body_b=b, length=length, enable_spring=True, hertz=hertz, damping_ratio=damping,
            enable_limit=True, min_length=0.3 * length, max_length=1.6 * length,
        )

    def _middle(self):
        """(n, 2): the middle of each ring (the mean of its balls)"""
        return self.balls.position.reshape(self.n, BALLS, 2).mean(axis=1)

    def _middle_velocity(self):
        return self.balls.linear_velocity.reshape(self.n, BALLS, 2).mean(axis=1)

    def randomize(self, rng):
        push = np.repeat(rng.uniform(-0.3, 0.3, self.n), BALLS)
        self.balls.linear_velocity = np.column_stack([push, np.zeros(self.n * BALLS)])
        self.prev_x = self._middle()[:, 0].copy()
        self.best_x, self.progress_step = self.prev_x.copy(), np.zeros(self.n)
        self.effort = np.zeros(self.n)

    def act(self, actions):
        low, high = SPAN
        self.across.rest_length = (2 * RADIUS * (low + (actions + 1) / 2 * (high - low))).ravel()
        self.effort = np.mean(actions**2, axis=1)

    def observe(self):
        n = self.n
        rel = self.balls.position.reshape(n, BALLS, 2) - self._middle()[:, None]
        down = self.down.sensor_capacity.reshape(n, BALLS) > 0
        y = self.balls.position.reshape(n, BALLS, 2)[:, :, 1]  # (the ground is at 0)
        return [rel.reshape(n, -1), self._middle_velocity() / 5, down,
                y.min(axis=1) / BALL, y.mean(axis=1) - (RADIUS + BALL)]  # (a round ring at rest: 0)

    def reward(self):
        x = self._middle()[:, 0]
        airborne = ~(self.down.sensor_capacity.reshape(self.n, BALLS) > 0).any(axis=1)
        reward = 10 * (x - self.prev_x) + AIR_BONUS * airborne - 0.005 * self.effort
        self.prev_x = x
        progressed = x > self.best_x + STALL_DISTANCE
        self.best_x = np.where(progressed, x, self.best_x)
        self.progress_step = np.where(progressed, self.steps, self.progress_step)
        stalled = self.steps - self.progress_step > STALL_SECONDS / self.dt
        return reward, stalled

    def camera_target(self, i):
        """follow the ring to the right (the height stays)"""
        return float(self._middle()[i, 0]), self.view_center[1]

    def draw(self, debug_draw):
        """the springs as plain lines between the balls (the current policy's in color,
        the ghosts' blue)"""
        pos = self.balls.position.reshape(self.n, BALLS, 2)
        for i in range(self.n):
            color = LINE_COLOR if i == 0 else GHOST_LINE_COLOR
            for a, b in self.pairs:
                debug_draw.draw_segment(tuple(pos[i, a]), tuple(pos[i, b]), color)
