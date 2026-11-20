"""Toys: things you add to the world you watch, to tease a trained agent (the Toys tab).

A click on a body grabs it (drag it around), always. A click elsewhere into the world
uses the chosen tool (TOOLS): drop a box, a ball or a plank; set off an
explosion; put a ramp, a step or a seesaw on the ground (for the walkers to deal with:
heavy, but dynamic too: they can be pushed and tipped over). Gravity: its vector (sideways:
wind; up-down: the moon .. upside down). Weather: rain. They are real bodies / forces of the
world: the agent bumps into them, they push it. A reset of the episode (R) removes them
(the world goes back to its start). Never in training: only the env you watch. Their
settings (PARAMS) are sliders in the Toys tab, in groups.
"""

import time

import numpy as np
import pyb2d3 as b2d

from webgym.core import SCENERY, SCENERY_FILTER

TOY_COLOR, RAIN_COLOR, GROUND_COLOR = 0xD9A441, 0x5DA9E9, 0x8FBF6F
RAIN_WATER = (0.45, 0.85, 0.95)  # (the rain drawn as a liquid: a light cyan, not the water cannon's blue)

# the sliders of the Toys tab: {name: (default, min, max, step, group, explanation)};
# step None: a free slider (no steps). The Gravity group's act while they are dragged
# (LIVE), the others on the next toy
PARAMS = {
    "size": (0.4, 0.1, 1.5, None, "Things to drop", "Boxes and balls: half the side of a box / the radius of a ball (m)."),
    "weight": (1.0, 0.1, 10.0, None, "Things to drop", "Boxes, balls and planks: their density (kg per m²): heavy ones knock the agent over."),
    "bounce": (0.2, 0.0, 0.95, None, "Things to drop", "Boxes, balls and planks: how bouncy they are (0: a thud, 0.95: a rubber ball)."),
    "plank length": (2.0, 0.5, 6.0, None, "Things to drop", "A plank: its length (m)."),
    "blast radius": (4.0, 1.0, 10.0, None, "Explosion", "How far it reaches (m)."),
    "blast strength": (6.0, 1.0, 40.0, None, "Explosion", "How hard it pushes (impulse per meter of surface facing it)."),
    "ramp length": (3.0, 0.5, 8.0, None, "Ramps and steps", "A ramp: how long it is (m); it rises to the right."),
    "ramp height": (0.6, 0.1, 3.0, None, "Ramps and steps", "A ramp: its height at the high end (m)."),
    "step height": (0.3, 0.05, 2.0, None, "Ramps and steps", "A step: its height (m); 1 m wide."),
    "seesaw length": (3.0, 1.0, 8.0, None, "Ramps and steps", "A seesaw: the length of its plank (m)."),
    "heaviness": (20.0, 1.0, 100.0, None, "Ramps and steps",
                  "Ramps, steps and seesaws: their density. Heavy ones stay where they are (but can be "
                  "pushed or tipped over), light ones slide away under the agent."),
    "gravity x": (0.0, -2.5, 2.5, None, "Gravity",
                  "The gravity vector's sideways part, in the world's own g (+ = to the right): a wind that pushes "
                  "everything; a storm at 1. (The arrow in the view's corner; its ring: 1 g.)"),
    "gravity y": (1.0, -2.5, 2.5, None, "Gravity",
                  "The gravity vector's downward part, in the world's own g: 1 as usual, 0.17 is the moon, 2.5 a "
                  "heavy planet, 0 floats, below 0 everything falls up."),
    "rain rate": (150.0, 10.0, 600.0, 10.0, "Weather", "Rain: drops per second."),
    "drop size": (0.01, 0.005, 0.3, None, "Weather", "Rain: the radius of a drop (m)."),
    "drop weight": (0.5, 0.1, 10.0, None, "Weather", "Rain: the density of a drop: light drizzle .. hail."),
    "drop life": (15.0, 0.5, 15.0, None, "Weather", "Rain: how long a drop stays (s) before it disappears."),
}
GROUPS = ("Things to drop", "Explosion", "Ramps and steps", "Gravity", "Weather")  # (in this order)
LIVE = ("Gravity",)  # the groups whose sliders act while dragged (the watched world follows at once)


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


# -- gravity, weather -----------------------------------------------------------------

def gravity(world, base_gravity, p):
    """the world's gravity vector (for everything in it), both parts in the world's own g:
    sideways (gravity x, + = to the right) and down (gravity y, 1 = as usual). Bodies at
    rest sleep and would not notice another gravity: while it is not the world's own,
    nothing sleeps"""
    g = abs(base_gravity[1])
    world.gravity = (p["gravity x"] * g, -p["gravity y"] * g)
    own = p["gravity x"] == 0.0 and p["gravity y"] == 1.0
    if world.sleeping_enabled != own:
        world.sleeping_enabled = own  # (off: box2d wakes every body)


GRAVITY_COLOR = 0xEFE2C8


def draw_gravity(draw, camera, p):
    """the gravity vector as an arrow in the view's top right corner; the ring: 1 g. Far
    enough from the edges that the longest arrow (both sliders at their limit) fits"""
    r = 0.06 * camera.zoom  # (1 g, in world units: the same size on screen at any zoom)
    reach = max(abs(PARAMS[k][i]) for k in ("gravity x", "gravity y") for i in (1, 2)) * r  # (per axis)
    corner = camera.convert_screen_to_world(b2d.Vec2(camera.width, 0))
    center = (corner[0] - reach - 0.04 * camera.zoom, corner[1] - reach - 0.04 * camera.zoom)
    draw.draw_circle(center, r, GRAVITY_COLOR)
    dx, dy = p["gravity x"] * r, -p["gravity y"] * r
    length = np.hypot(dx, dy)
    if length < 1e-6:  # (no gravity: a dot)
        draw.draw_point(center, 6.0, GRAVITY_COLOR)
        return
    tip = (center[0] + dx, center[1] + dy)
    draw.draw_segment(center, tip, GRAVITY_COLOR)
    ux, uy, head = dx / length, dy / length, min(0.3 * r, 0.5 * length)
    for side in (1, -1):  # (the arrow's head)
        draw.draw_segment(tip, (tip[0] - head * (ux - side * 0.6 * uy), tip[1] - head * (uy + side * 0.6 * ux)),
                          GRAVITY_COLOR)


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

    def clear(self, world):
        """remove all drops (the rain stops); only from the world they are in (see step)"""
        if world is self.world:
            for body, _ in self.drops:
                if body.is_valid:
                    body.destroy()
        self.drops = []

    def draw(self, draw, p):
        """the drops as water (a liquid: they merge into puddles; the drop bodies
        themselves, in RAIN_COLOR, are not drawn)"""
        drops = [body for body, _ in self.drops if body.is_valid]
        if drops:
            # (the blobs bigger than the drops: they flow together into puddles)
            draw.draw_liquid(b2d.Bodies(drops).position, 3.5 * p["drop size"], RAIN_WATER)
