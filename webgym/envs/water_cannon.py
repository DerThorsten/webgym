"""WaterCannon: keep balls in the air with water jets.

A cart with water cannons (barrels on hinges) drives under light balls and holds them
up with the jets. The balls can't touch the cart: only the water carries them; the jets
also push them sideways, so the cart has to follow, aim and dose the pressure. The water
drops are real bodies (an emitter at the end of each barrel), their recoil pushes the
cart. Each cart's water only hits its own balls (collision groups): many carts share one
world. Setting "balls": keep 1 or 2 up (they bump into each other); the cart has as many
cannons as balls (two: a wider cart, a cannon at each end).
obs: per ball its position (relative to the cart) and velocity; the cart's velocity;
per cannon its angle, pressure and spread.
action: drive (-1: left .. 1: right), then per cannon: angle (-1: left .. 1: right),
pressure (<= 0: off, 1: full), spread (-1: a narrow jet .. 1: a wide shower).
reward: 1 per step while the balls are in the air, up to 0.5 more for keeping them high
(the lowest at 4 m: all of it), minus the water used (all cannons at full pressure:
0.2): high balls with little water, more reward. The episode ends when a ball touches
the ground or flies away, or after 20 s.
"""

import numpy as np
import pyb2d3 as b2d
from pyb2d3 import extensions as ext

from webgym.core import PARTICLE, SCENERY, Env, register_env

BARREL = 1.0  # a barrel's length (the water leaves at its end)
ANGLE_MAX = 0.8  # a barrel's angle from straight up, either way (radians)
WHEEL_SPEED, MOTOR_TORQUE = 8.0, 60.0  # at full drive
JET_SPEED = 14.0  # m/s of the drops at full pressure
SPREAD_MIN, SPREAD_MAX = 0.02, 0.4  # half angle (radians) of the spray: a jet .. a shower
MAX_RATE = 120.0  # drops per second at full pressure (fewer at less)
DROP_RADIUS = 0.05
# how long a drop lives (s): long enough to reach the balls (at full pressure 6 m in
# ~0.6 s); the drops alive cost the time, not their collisions
DROP_LIFE = (0.5, 0.8)
WATER_COST = 0.2  # reward per step lost with all cannons at full pressure
# reward per step for the lowest ball's height (full at HEIGHT_FULL and above): each
# spray that lifts one pays at once (step-based learners like PPO need that; without it
# they learn "water costs" long before "water keeps the ball up")
HEIGHT_BONUS, HEIGHT_FULL = 0.5, 4.0
BALL_RADIUS = 0.25
BALL_GRAVITY = 0.45  # of the normal gravity: the balls float a little
CART_HALF_WIDTH = {1: 1.0, 2: 1.5}  # per number of balls (two: room for a cannon at each end)
BALL = 32  # the balls' collision category
GUN_ACTIONS = ("angle", "pressure", "spread")  # each cannon's actions


def own(category, i):
    """a filter for agent i's water / balls: they touch the ground and each other (the
    same positive group always collides), not the cart, not the other agents' water and
    balls (their masks leave them out): many agents fit in one world"""
    return b2d.make_filter(category_bits=category, mask_bits=SCENERY, group_index=i + 1)


def numbered(names, count):
    """names for `count` things: "angle" (one), "angle 1", "angle 2", ... (more)"""
    return tuple(name if count == 1 else f"{name} {k + 1}" for k in range(count) for name in names)


@register_env("WaterCannon")
class WaterCannon(Env):
    settings = {"balls": (1, (1, 2), "How many balls to keep in the air, with as many cannons "
                                     "on the cart (harder with more).")}
    max_steps = 1200
    view_center, view_zoom = (0.0, 4.0), 6.0
    suggested = {"quick": {"cem": {}},
                 "medium": {"cem": {}, "ppo_torch": {"hidden": 64}},
                 "long": {"ppo_torch": {"hidden": 64}, "sac_torch": {}, "cem": {}}}

    @property
    def obs_names(self):
        return (*numbered(("ball x", "ball y"), self.balls),
                *numbered(("ball vel. x", "ball vel. y"), self.balls),
                "cart vel.", *numbered(GUN_ACTIONS, self.balls))

    @property
    def action_names(self):
        return ("drive", *numbered(GUN_ACTIONS, self.balls))

    # (no discrete actions, no DQN: driving, aiming and dosing the water together need more
    # than a few fixed settings)

    @property
    def guns_x(self):
        """where the cannons sit on top of the cart (x along it): one in the middle, two
        at the ends"""
        edge = CART_HALF_WIDTH[self.balls] - 0.15
        return [0.0] if self.balls == 1 else [-edge, edge]

    @property
    def ball_starts(self):
        """where the balls start: above the cannons, one above the other"""
        return [(x, 4.0 + 1.5 * b) for b, x in enumerate(self.guns_x)]

    def build_scenery(self):
        self.scenery(b2d.box(40, 0.5, center=(0, -0.5)))

    def build_agent(self, i):
        half = CART_HALF_WIDTH[self.balls]
        cart = self.body((0, 1.0), b2d.box(half, 0.3), density=5.0)
        axles = []
        for x in (-(half - 0.3), half - 0.3):
            wheel = self.body((x, 0.4), b2d.circle(radius=0.4), density=5.0)
            axles.append(self.world.create_revolute_joint(
                body_a=cart, body_b=wheel, local_frame_a=b2d.transform(p=(x, -0.6)),
                local_frame_b=b2d.transform(p=(0, 0)), enable_motor=True, max_motor_torque=MOTOR_TORQUE,
            ))
        # the cannons: barrels on hinges on top of the cart, pointing up; motors turn them
        barrels, hinges = [], []
        for x in self.guns_x:
            barrel = self.body((x, 1.3), b2d.box(0.1, BARREL / 2, center=(0, BARREL / 2)), density=1.0)
            hinges.append(self.world.create_revolute_joint(
                body_a=cart, body_b=barrel, local_frame_a=b2d.transform(p=(x, 0.3)),
                local_frame_b=b2d.transform(p=(0, 0)), enable_limit=True, lower_angle=-ANGLE_MAX,
                upper_angle=ANGLE_MAX, enable_motor=True, max_motor_torque=200.0,
            ))
            barrels.append(barrel)
        # the balls: light (gravity pulls them only a little, they fall slowly)
        balls = [self.body((x, y), b2d.circle(radius=BALL_RADIUS), density=0.2,
                           filter=own(BALL, i), gravity_scale=BALL_GRAVITY)
                 for x, y in self.ball_starts]
        # (self.ball / self.barrel / self.hinge: all of them, agent after agent;
        # self.balls: how many each agent has)
        return {"cart": cart, "axles": axles, "barrel": barrels, "hinge": hinges, "ball": balls}

    def water_cannon(self, k, seed):
        """an emitter of water drops at the end of barrel k (of all agents): it follows the
        barrel, the barrel takes the recoil (made anew each episode: its random drops
        start from `seed`)"""
        agent = k // self.balls
        return ext.RadialEmitter(
            self.world,
            b2d.shape_def(density=1.0, filter=own(PARTICLE, agent), material=b2d.surface_material(friction=0.1)),
            b2d.circle(radius=DROP_RADIUS),
            body=self.barrel[k], seed=seed, local_frame=b2d.transform(p=(0, BARREL), q=b2d.make_rot(np.pi / 2)),
            rate=MAX_RATE, capacity=int(MAX_RATE * DROP_LIFE[1]) + 1, angle_min=-0.04, angle_max=0.04,
            lifetime=ext.uniform(*DROP_LIFE), max_lifetime=DROP_LIFE[1],
            colors=[(0.0, 0xE0F4FF), (1.0, 0x2B6CB0)],  # by age
        )

    def randomize(self, rng):
        guns = self.n * self.balls
        seeds = rng.integers(2**31, size=guns // self.balls)  # (one per agent: the same start for all alike)
        self.cannon = [self.water_cannon(k, int(seeds[k // self.balls]) + k % self.balls) for k in range(guns)]
        # the balls start above the cannons (ball_starts), all a little aside, drifting
        n, starts = self.n, np.array(self.ball_starts)
        x = rng.uniform(-0.3, 0.3, n)[:, None] + starts[:, 0]
        y = rng.uniform(-0.5, 0.5, n)[:, None] + starts[:, 1]
        drift = rng.uniform(-0.5, 0.5, n)[:, None] + np.zeros(self.balls)
        self.ball.position = np.column_stack([x.ravel(), y.ravel()])
        self.ball.linear_velocity = np.column_stack([drift.ravel(), np.zeros(n * self.balls)])
        self.pressure, self.spread = np.zeros(guns), np.full(guns, SPREAD_MIN)

    def act(self, actions):
        self.axles.target_motor_speed = np.repeat(-actions[:, 0] * WHEEL_SPEED, 2)  # (- : roll right)
        # each cannon's actions, one row per cannon (agent after agent)
        angle, pressure, spread = actions[:, 1:].reshape(-1, 3).T
        target = -angle * ANGLE_MAX  # (+ action: aim right = turn clockwise)
        self.hinge.target_motor_speed = 6.0 * (target - self.hinge.current_angle)  # (turn towards it)
        self.pressure = np.maximum(pressure, 0.0)
        self.spread = SPREAD_MIN + (spread + 1) / 2 * (SPREAD_MAX - SPREAD_MIN)
        for k, cannon in enumerate(self.cannon):
            p = float(self.pressure[k])
            cannon.enabled = p > 0.05
            cannon.rate = MAX_RATE * p  # (less pressure: less water)
            cannon.speed = ext.uniform(0.9 * JET_SPEED * p, JET_SPEED * p)
            cannon.angle_min, cannon.angle_max = -self.spread[k], self.spread[k]
            impulse, angular_impulse = cannon.step(self.dt)  # the recoil
            self.barrel[k].apply_linear_impulse_to_center(impulse)
            self.barrel[k].apply_angular_impulse(angular_impulse)

    def observe(self):
        n = self.n
        balls = self.ball.position.reshape(n, self.balls, 2) - self.cart.position[:, None]  # (a row per agent)
        guns = np.column_stack([self.hinge.current_angle / ANGLE_MAX, self.pressure,
                                (self.spread - SPREAD_MIN) / (SPREAD_MAX - SPREAD_MIN)])  # (one row per cannon)
        return [balls.reshape(n, -1) / 5, self.ball.linear_velocity.reshape(n, -1) / 5,
                self.cart.linear_velocity[:, 0] / 5, guns.reshape(n, -1)]

    def reward(self):
        x, y = self.ball.position.reshape(self.n, self.balls, 2).transpose(2, 0, 1)
        dropped = (y < 1.5 * BALL_RADIUS) | (np.abs(x - self.cart.position[:, :1]) > 10)  # (on the ground)
        water = self.pressure.reshape(self.n, self.balls).mean(axis=1)
        height = np.clip(y.min(axis=1) / HEIGHT_FULL, 0, 1)
        return 1.0 + HEIGHT_BONUS * height - WATER_COST * water, dropped.any(axis=1)

    def camera_target(self, i):
        """follow the cart sideways (the height stays)"""
        return float(self.cart.position[i, 0]), self.view_center[1]
