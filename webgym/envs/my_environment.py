"""MyEnvironment: a starting point for your own environment (edit me!).

A ball learns to roll to a target (the white disc) and to stay there. It is the
smallest environment here, made to be changed: open the Code, edit, press Apply
(Ctrl/Cmd+Enter) and train again. Some ideas:
- a target in the air: the ball has to jump
- a wall or a slope in the way
- a second ball, or a box on top of the ball
- another reward: get there fast, or stop exactly on the target

The other environments (CartPole is the next simplest) show more: joints, motors,
sensors, settings.

Observations (2):
- the distance to the target (x)
- the ball's velocity (x)

Actions (1):
- push the ball (-1: left .. 1: right)

Reward: minus the distance to the target, every step (be there soon, stay there).

Episode end: after 5 s.
"""

import numpy as np
import pyb2d3 as b2d

from webgym.core import Env, register_env


# The page trains many agents at once, all in ONE box2d world (they don't touch each
# other). build_agent builds one agent; everything after it works on all agents at
# once: numpy arrays with one row per agent (self.n agents).
@register_env("MyEnvironment")
class MyEnvironment(Env):
    obs_names = ("to target", "velocity")  # what the agent sees (observe returns them)
    action_names = ("push",)  # what it controls, each in [-1, 1]
    max_steps = 300  # 5 s (60 steps per second)
    view_center, view_zoom = (0.0, 1.5), 3.0  # the camera: the middle, half the height (m)
    suggested = {"quick": {"my_algorithm": {}, "cem": {}}}  # (the algorithms the page picks)

    def build_scenery(self):
        """the world, once: the ground (10 m wide, static: it never moves)"""
        self.scenery(b2d.box(5.0, 0.1))

    def build_agent(self, i):
        """one agent (i: which one): a ball, a dynamic body that falls, rolls, collides"""
        ball = self.body((0, 0.4), b2d.circle(radius=0.3))
        return {"ball": ball}  # from here on self.ball are the balls of all agents

    def randomize(self, rng):
        """a new episode (the world is back at its start): a new target, the same for
        all agents (a fair race: rng.uniform(-3, 3, self.n) would give each its own)"""
        self.target = np.full(self.n, rng.uniform(-3, 3))

    def act(self, actions):
        """actions: one row per agent, one column per action name"""
        force = np.column_stack([2.0 * actions[:, 0], np.zeros(self.n)])  # (x, y) in newton
        self.ball.apply_force_to_center(force, True)

    def observe(self):
        """what the agents see: one array (one value per agent) per observation name"""
        x = self.ball.position[:, 0]
        return [self.target - x, self.ball.linear_velocity[:, 0]]

    def reward(self):
        """after each step: the reward of every agent, and whether its episode is over"""
        distance = np.abs(self.target - self.ball.position[:, 0])
        return -distance * self.dt, np.zeros(self.n, dtype=bool)  # (never over early)

    def draw(self, debug_draw):
        """looks only: the target as a white disc (of agent 0, the one you watch)"""
        debug_draw.draw_solid_circle(b2d.transform(p=(float(self.target[0]), 0.1)), 0.1, 0xFFFFFF)
