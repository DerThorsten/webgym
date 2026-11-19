"""Toys: things you add to the world you watch, to tease a trained agent (the Toys tab).

A click on a body grabs it (drag it around), always. A click elsewhere into the world
uses the chosen tool (TOOLS): drop a box, a ball or a plank; set off an
explosion; put a ramp, a step or a seesaw on the ground (for the walkers to deal with:
heavy, but dynamic too: they can be pushed and tipped over). Weather: rain, wind, more or less gravity. They are real bodies / forces of the
world: the agent bumps into them, they push it. A reset of the episode (R) removes them
(the world goes back to its start). Never in training: only the env you watch. Their
settings (PARAMS) are sliders in the Toys tab, in groups.
"""

import time

import numpy as np
import pyb2d3 as b2d

from webgym.core import SCENERY, SCENERY_FILTER

TOY_COLOR, RAIN_COLOR, GROUND_COLOR = 0xD9A441, 0x5DA9E9, 0x8FBF6F

# the sliders of the Toys tab: {name: (default, min, max, step, group, explanation)}
PARAMS = {
    "size": (0.4, 0.1, 1.5, 0.05, "Things to drop", "Boxes and balls: half the side of a box / the radius of a ball (m)."),
    "weight": (1.0, 0.1, 10.0, 0.1, "Things to drop", "Boxes, balls and planks: their density (kg per m²): heavy ones knock the agent over."),
    "bounce": (0.2, 0.0, 0.95, 0.05, "Things to drop", "Boxes, balls and planks: how bouncy they are (0: a thud, 0.95: a rubber ball)."),
    "plank length": (2.0, 0.5, 6.0, 0.25, "Things to drop", "A plank: its length (m)."),
    "blast radius": (4.0, 1.0, 10.0, 0.5, "Explosion", "How far it reaches (m)."),
    "blast strength": (6.0, 1.0, 40.0, 1.0, "Explosion", "How hard it pushes (impulse per meter of surface facing it)."),
    "ramp length": (3.0, 0.5, 8.0, 0.25, "Ramps and steps", "A ramp: how long it is (m); it rises to the right."),
    "ramp height": (0.6, 0.1, 3.0, 0.05, "Ramps and steps", "A ramp: its height at the high end (m)."),
    "step height": (0.3, 0.05, 2.0, 0.05, "Ramps and steps", "A step: its height (m); 1 m wide."),
    "seesaw length": (3.0, 1.0, 8.0, 0.25, "Ramps and steps", "A seesaw: the length of its plank (m)."),
    "heaviness": (20.0, 1.0, 100.0, 1.0, "Ramps and steps",
                  "Ramps, steps and seesaws: their density. Heavy ones stay where they are (but can be "
                  "pushed or tipped over), light ones slide away under the agent."),
    "wind": (0.0, -20.0, 20.0, 0.5, "Weather", "Pushes everything sideways (m/s², + = to the right): a storm at 10."),
    "gravity": (1.0, 0.1, 2.5, 0.05, "Weather", "Gravity, times the world's own: 0.17 is the moon, 2.5 a heavy planet."),
    "rain rate": (120.0, 10.0, 600.0, 10.0, "Weather", "Rain: drops per second."),
    "drop size": (0.06, 0.02, 0.3, 0.01, "Weather", "Rain: the radius of a drop (m)."),
    "drop weight": (0.5, 0.1, 10.0, 0.1, "Weather", "Rain: the density of a drop: light drizzle .. hail."),
    "drop life": (4.0, 0.5, 15.0, 0.5, "Weather", "Rain: how long a drop stays (s) before it disappears."),
}
GROUPS = ("Things to drop", "Explosion", "Ramps and steps", "Weather")  # (in this order)


def defaults():
    return {name: spec[0] for name, spec in PARAMS.items()}


def thing(world, position, geometry, p, density=None, color=TOY_COLOR):
    """a loose body at `position`: the agents and the scenery collide with it"""
    body = world.create_dynamic_body(position=position)
    shape_def = b2d.shape_def(density=p["weight"] if density is None else density, filter=SCENERY_FILTER,
                              enable_sensor_events=True,
                              material=b2d.surface_material(friction=0.6, restitution=p["bounce"], custom_color=color))
    body.create_shape(shape_def, geometry)
    return body


def empty(world, position):
    """is there nothing at `position` (no body of the world)? the tools act only there"""
    near = []
    world.overlap_aabb(b2d.aabb((position[0] - 0.01, position[1] - 0.01), (position[0] + 0.01, position[1] + 0.01)),
                       near.append)
    return not any(shape.test_point(position) for shape in near if not shape.is_sensor)


def ground_below(world, position):
    """the point of the ground (the scenery) straight below `position` (or `position`)"""
    hit = world.cast_ray_closest(position, (0.0, -100.0), b2d.make_query_filter(category_bits=1, mask_bits=SCENERY))
    return (position[0], hit.point[1]) if hit.hit else position


def piece(world, position, geometry, p):
    """a piece of ground to put in the agent's way: a dynamic body, as heavy as
    p["heaviness"] says (heavy: it stays; light: it slides away), grippy"""
    body = world.create_dynamic_body(position=position)
    shape_def = b2d.shape_def(density=p["heaviness"], filter=SCENERY_FILTER, enable_sensor_events=True,
                              material=b2d.surface_material(friction=0.9, custom_color=GROUND_COLOR))
    body.create_shape(shape_def, geometry)
    return body


# -- the tools of a click (each: (world, position, params) -> a blast to draw or None) -----

def drop_box(world, position, p):
    thing(world, position, b2d.box(p["size"], p["size"]), p)


def drop_ball(world, position, p):
    thing(world, position, b2d.circle(radius=p["size"]), p)


def drop_plank(world, position, p):
    thing(world, position, b2d.box(p["plank length"] / 2, 0.05), p)


def explode(world, position, p):
    """push everything near `position` away from it -> the blast to draw (see draw_blasts)"""
    world.explode(position=position, radius=p["blast radius"], impulse_per_length=p["blast strength"])
    return {"position": position, "radius": p["blast radius"], "start": time.perf_counter()}


def ramp(world, position, p):
    """a ramp (a wedge) on the ground below the click, rising to the right"""
    x, y = ground_below(world, position)
    length, height = p["ramp length"], p["ramp height"]
    piece(world, (x, y + 0.01), b2d.polygon([(-length / 2, 0), (length / 2, 0), (length / 2, height)]), p)


def step(world, position, p):
    """a step (a block) on the ground below the click"""
    x, y = ground_below(world, position)
    piece(world, (x, y + p["step height"] / 2 + 0.01), b2d.box(0.5, p["step height"] / 2), p)


def seesaw(world, position, p):
    """a seesaw on the ground below the click: a wedge with a plank on a hinge at its tip"""
    x, y = ground_below(world, position)
    base = piece(world, (x, y + 0.01), b2d.polygon([(-0.4, 0), (0.4, 0), (0, 0.5)]), p)
    plank = piece(world, (x, y + 0.6), b2d.box(p["seesaw length"] / 2, 0.06), p)
    world.create_revolute_joint(body_a=base, body_b=plank, local_frame_a=b2d.transform(p=(0, 0.5)),
                                local_frame_b=b2d.transform(p=(0, -0.08)))


# {name: (button label, the tool, its group in the Toys tab)}; "grab": no tool (a click
# on a body always grabs it, whatever tool is chosen)
TOOLS = {
    "grab": ("Grab", None, None),
    "box": ("Box", drop_box, "Things to drop"),
    "ball": ("Ball", drop_ball, "Things to drop"),
    "plank": ("Plank", drop_plank, "Things to drop"),
    "explosion": ("💥 Explosion", explode, "Explosion"),
    "ramp": ("Ramp", ramp, "Ramps and steps"),
    "step": ("Step", step, "Ramps and steps"),
    "seesaw": ("Seesaw", seesaw, "Ramps and steps"),
}


# -- explosions, drawn ---------------------------------------------------------------

BLAST_SECONDS = 0.35  # how long a blast is drawn


def draw_blasts(draw, blasts):
    """the blasts of the last BLAST_SECONDS (as in pyb2d3's tank sample): rings growing to
    the blast's radius, a flash in the middle -> the blasts still to draw"""
    now, left = time.perf_counter(), []
    for blast in blasts:
        t = (now - blast["start"]) / BLAST_SECONDS  # 0 .. 1
        if t >= 1:
            continue
        draw.draw_circle(blast["position"], blast["radius"] * t, b2d.HexColor.OrangeRed)
        draw.draw_circle(blast["position"], blast["radius"] * t * 0.6, b2d.HexColor.Orange)
        draw.draw_point(blast["position"], 30.0 * (1 - t), b2d.HexColor.Yellow)
        left.append(blast)
    return left


# -- weather -------------------------------------------------------------------------

def weather(world, base_gravity, p):
    """wind and gravity: the world's gravity tilted by the wind, scaled"""
    world.gravity = (p["wind"], base_gravity[1] * p["gravity"])


class Rain:
    """drops falling from above the view; each lives p["drop life"] seconds"""

    def __init__(self):
        self.drops = []  # [body, seconds left]
        self.world = None  # (the world the drops are in)
        self.rng = np.random.default_rng()

    def step(self, world, left, right, top, dt, p):
        """new drops between x = left .. right at height `top` (p["rain rate"] per second,
        on average); old ones go"""
        if world is not self.world:
            # another world (the watched env was rebuilt): the old drops went with the old
            # one. Forget them: box2d body ids don't name their world, an old id may point
            # to a body of the new one (destroying it crashed the page)
            self.world, self.drops = world, []
        for _ in range(self.rng.poisson(p["rain rate"] * dt)):
            x = float(self.rng.uniform(left, right))
            drop = thing(world, (x, top), b2d.circle(radius=p["drop size"]), p, p["drop weight"], RAIN_COLOR)
            self.drops.append([drop, p["drop life"]])
        alive = []
        for body, life in self.drops:
            if not body.is_valid:  # (gone: the world was reset)
                continue
            if life - dt <= 0:
                body.destroy()
            else:
                alive.append([body, life - dt])
        self.drops = alive
