"""MyBallPit: a starting point for an environment with a world full of loose things (edit me!).

A ball learns to push its way through a heap of 60 small balls to the target on the
other side (the white disc). The small balls are not part of the agent, they belong to
the world (made with self.prop), and the agent shoves them around.

Why this environment is not batchable (batchable = False): the other examples put all
agents into ONE world, side by side, which is much faster. Here every agent changes its
world, so every agent needs its own 60 balls. Each agent therefore gets a world of its
own. That is simple, and a little slower.

To be fair: this one could also be batched. Every agent would get its own 60 balls in a
collision group of its own (as the Blob's rings), so they never meet the others. But then
one world holds 60 balls per agent, 3840 for 64 agents, and the simulation has to sort
out all of them every step. For worlds with many loose things, separate worlds are the
simpler code and often just as fast.

Some ideas:
- more or bigger balls, or boxes instead
- a target up on a ledge
- a second agent ball (in the same world: they push each other)

Observations (3):
- the distance to the target (x)
- the ball's velocity (x, y)

Actions (1):
- push the ball (-1: left .. 1: right)

Reward: minus the distance to the target, every step.

Episode end: the ball reaches the target, or after 8 s.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Env, register_env

TARGET = 3.5  # m: where the ball has to get to (x)
ROWS, COLUMNS = 6, 10  # the heap of small balls: 6 x 10 = 60
SMALL = 0.12  # m: a small ball's radius
PUSH = 6.0  # N: the agent's strongest push


# Not batchable: each agent gets its own world. Inside
# this class there is only ever ONE agent (self.n is 1), but the code is written for n
# agents anyway, as in a batchable environment: one row per agent.
@register_env("MyBallPit")
class MyBallPit(Env):
    obs_names = ("to target", "velocity x", "velocity y")  # what the agent sees
    action_names = ("push",)  # what it controls, in [-1, 1]
    max_steps = 480  # 8 s (60 steps per second)
    batchable = False  # (the agent changes its world: one world per agent)
    view_center, view_zoom = (0.0, 1.5), 3.0  # the camera: the middle, half the height (m)
    suggested = {"quick": {"cem": {}, "my_algorithm": {}}}  # (the algorithms the page picks)

    def build_scenery(self):
        """the world, once: the ground, a wall at each end, and the heap"""
        self.scenery(b2d.box(5.0, 0.1))  # the ground: 10 m wide, its top at y = 0.1
        self.scenery(b2d.box(0.1, 1.5, center=(-5.0, 1.5)))  # the left wall
        self.scenery(b2d.box(0.1, 1.5, center=(5.0, 1.5)))  # the right wall
        # The heap: self.prop makes a LOOSE body. It is part of the world, not of the
        # agent: it falls, rolls, and the agent pushes it around. (self.body would make a
        # part of the agent instead.) Light and slippery, so the ball gets through.
        for row in range(ROWS):
            for col in range(COLUMNS):
                x = -0.5 + col * 2.1 * SMALL  # (side by side, a little gap between)
                x += (row % 2) * SMALL  # (every other row half a ball over: they settle into a heap)
                y = 0.1 + SMALL + row * 2 * SMALL  # (rows on top of each other)
                self.prop((x, y), b2d.circle(radius=SMALL), density=0.2, friction=0.3)

    def build_agent(self, i):
        """one agent: a ball, left of the heap (self.body: a part of the agent)"""
        ball = self.body((-3.0, 0.4), b2d.circle(radius=0.3))
        return {"ball": ball}  # from here on self.ball are the balls of all agents

    def randomize(self, rng):
        """a new episode (the world, heap included, is back at its start): a small random
        push to start with, so no two episodes are the same"""
        speed = rng.uniform(-0.5, 0.5, self.n)  # (x, one per agent)
        self.ball.linear_velocity = np.column_stack([speed, np.zeros(self.n)])

    def act(self, actions):
        """actions: one row per agent; push the ball sideways"""
        force = np.column_stack([PUSH * actions[:, 0], np.zeros(self.n)])  # (x, y) in newton
        self.ball.apply_force_to_center(force, True)

    def observe(self):
        """what the agents see: one value per agent for each name in obs_names"""
        x = self.ball.position[:, 0]
        velocity = self.ball.linear_velocity  # (one row per agent: x, y)
        return [TARGET - x, velocity[:, 0], velocity[:, 1]]

    def reward(self):
        """after each step: the reward of every agent, and whether its episode is over"""
        distance = np.abs(TARGET - self.ball.position[:, 0])
        reward = -distance * self.dt  # (the closer, the better; soon there, the better)
        arrived = distance < 0.3  # (there: the episode is over)
        return reward, arrived

    def draw(self, debug_draw):
        """looks only: the target as a white disc on the ground"""
        debug_draw.draw_solid_circle(b2d.transform(p=(TARGET, 0.1)), 0.1, 0xFFFFFF)
