# this code is a modified version of  https://github.com/giorgosg/box2d-py/blob/main/src/box2d_testbed/debug_draw_gl.py
# The code has been adapted to work with the pyb2d3 library and its frontend system.
# The original code is licensed under the MIT License with the following copyright:
#
# MIT License
# Copyright 2025 Giorgos Giagas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# (copied from pyb2d3/examples/pyb2d3_sandbox_opengl/debug_draw_gl.py; no sandbox
# dependency: plain b2d.DebugDraw base, text drawing injected)
import pyb2d3 as b2d
from OpenGL.GL import *  # noqa: F403
from pyb2d3 import Vec2, AABB
from .draw import GLBackground, GLCircles, GLLiquid, GLPoints, GLLines
from .draw import GLSolidPolygons, GLSolidCircles, GLSolidCapsules
from .draw import CAPSULE_DTYPE, CIRCLE_DTYPE, LINE_VERTEX_DTYPE, POINT_DTYPE
from .draw import SOLID_CIRCLE_DTYPE, make_rgba8, polygon_dtype
import numpy as np
import time
import weakref

# (PyOpenGL's error checking: switched off in this package's __init__, before the
# first OpenGL.GL import)


class Camera:
    def __init__(self):
        self.width = 1280
        self.height = 800
        self.reset_view()
        self._matrix = None

    def reset_view(self):
        """Reset camera to initial position and zoom"""
        self.center = Vec2(0.0, 0.0)
        self.zoom = 1.0
        self._matrix = None

    def set_size(self, width, height):
        """Set camera size"""
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self._matrix = None

    def set_view(self, center, zoom, width, height):
        """Set camera view parameters"""
        if (
            self.center != Vec2(*center)
            or self.zoom != zoom
            or self.width != width
            or self.height != height
        ):
            self.center = Vec2(*center)
            self.zoom = zoom
            self.width = width
            self.height = height
            self._matrix = None

    def screen_delta_to_world_delta(self, delta):
        w0 = self.convert_screen_to_world(Vec2(0, 0))
        w1 = self.convert_screen_to_world(Vec2(delta[0], delta[1]))
        return Vec2(w1.x - w0.x, w1.y - w0.y)

    def convert_screen_to_world(self, ps):
        """Convert from screen coordinates to world coordinates"""
        w = float(self.width)
        h = float(self.height)
        u = ps.x / w
        v = (h - ps.y) / h

        ratio = w / h
        extents = Vec2(self.zoom * ratio, self.zoom)

        lower = self.center - extents
        upper = self.center + extents

        pw = Vec2((1.0 - u) * lower.x + u * upper.x, (1.0 - v) * lower.y + v * upper.y)
        return pw

    def convert_world_to_screen(self, pw):
        """Convert from world coordinates to screen coordinates"""
        w = float(self.width)
        h = float(self.height)
        ratio = w / h

        extents = Vec2(self.zoom * ratio, self.zoom)

        # Vec2 operations
        lower = self.center - extents
        upper = self.center + extents

        u = (pw.x - lower.x) / (upper.x - lower.x)
        v = (pw.y - lower.y) / (upper.y - lower.y)

        ps = Vec2(u * w, (1.0 - v) * h)
        return ps

    def build_projection_matrix(self, z_bias=0.0):
        """Build projection matrix for rendering"""
        if self._matrix is not None:
            return self._matrix
        ratio = float(self.width) / float(self.height)
        extents = Vec2(self.zoom * ratio, self.zoom)

        # Vec2 operations
        lower = self.center - extents
        upper = self.center + extents

        w = upper.x - lower.x
        h = upper.y - lower.y

        matrix = np.zeros(16, dtype=np.float32)

        # Column-major order
        matrix[0] = 2.0 / w
        matrix[5] = 2.0 / h
        matrix[10] = -1.0
        matrix[12] = -2.0 * self.center.x / w
        matrix[13] = -2.0 * self.center.y / h
        matrix[14] = z_bias
        matrix[15] = 1.0
        self._matrix = matrix
        return matrix

    def get_view_bounds(self):
        """Get AABB in world coordinates of current view"""
        lower = self.convert_screen_to_world(Vec2(0.0, float(self.height)))
        upper = self.convert_screen_to_world(Vec2(float(self.width), 0.0))
        return AABB(lower, upper)


ensure_hex = b2d.ensure_hex_color


def _rgb(colors):
    """packed RGBA8 0xRRGGBB colors -> their packed rgb part, to compare with"""
    return [make_rgba8(c, 0) & 0x00FFFFFF for c in colors]


def _faded(block, dtype, alpha, only=(), hide=(), show=None):
    """a float32 copy of a BatchDebugDraw block; the shapes drawn in one of the `only`
    colors (0xRRGGBB) get `alpha` (0..255; None: nothing changes), those in the `hide`
    colors are left out; `show` (colors): only those are kept"""
    out = np.array(block, dtype=np.float32, copy=True)
    if len(out) == 0:
        return out
    rows = out.view(dtype).reshape(-1)
    keep = None
    if hide:
        keep = ~np.isin(rows["color"] & 0x00FFFFFF, _rgb(hide))
    if show is not None:
        shown = np.isin(rows["color"] & 0x00FFFFFF, _rgb(show))
        keep = shown if keep is None else keep & shown
    if keep is not None:
        kept = rows[keep]
        out, rows = kept.view(np.float32).reshape((-1, *out.shape[1:])), kept
    if alpha is not None and only and len(out):
        colors = rows["color"]
        change = np.isin(colors & 0x00FFFFFF, _rgb(only))
        colors[change] = (colors[change] & 0x00FFFFFF) | np.uint32(alpha << 24)
    return out


class GLDebugDraw(b2d.DebugDraw):
    def __init__(self, camera, draw_text=None):
        """`draw_text(screen_pos, text, (r, g, b))` draws the debug strings
        (None: they are dropped)."""
        super().__init__()

        self.camera = camera
        self.draw_text = draw_text
        self.background = GLBackground(self.camera)
        self.circles = GLCircles(self.camera)
        self.solid_circles = GLSolidCircles(self.camera)
        self.solid_capsules = GLSolidCapsules(self.camera)
        self.solid_polygons = GLSolidPolygons(self.camera)
        self.points = GLPoints(self.camera)
        self.lines = GLLines(self.camera)
        self.liquids = {}  # (webgym: water from drops, one liquid per color, see draw_liquid)
        # Add storage for debug strings
        self.debug_strings = []
        # the world is drawn in C++ into flat arrays (no python call per shape); the
        # draw_* callbacks below are for what samples draw themselves. The string
        # callback holds this debug draw weakly: a strong reference from the C++
        # object back to self would be a cycle the garbage collector cannot see
        self_ref = weakref.ref(self)

        def on_string(position, text, color):
            """body names etc. (rare): into this debug draw's strings"""
            debug_draw = self_ref()
            if debug_draw is not None:
                debug_draw.draw_string(position, text, color)

        self.batch = b2d.BatchDebugDraw(on_string=on_string)

    def draw_world(self, world, alpha=None, only=(), hide=(), show=None):
        """draw `world` with this debug draw's options: collected in C++
        (BatchDebugDraw), handed to the GL batches as whole arrays. Several worlds
        can be drawn between begin_draw / end_draw (overlaid); the shapes in the `only`
        colors (0xRRGGBB) get `alpha` (0..255; webgym: the ghosts), those in the `hide`
        colors are not drawn (webgym: sensors); `show` (colors): only those are drawn"""
        batch = self.batch
        batch.copy_options(self)
        batch.clear()
        world._draw(batch)
        targets = (
            (self.solid_circles.circles, batch.solid_circles, SOLID_CIRCLE_DTYPE),
            (self.solid_capsules.capsules, batch.capsules, CAPSULE_DTYPE),
            (self.solid_polygons, batch.polygons, polygon_dtype),
            (self.circles.circles, batch.circles, CIRCLE_DTYPE),
            (self.points.points, batch.points, POINT_DTYPE),
            (self.lines.lines, batch.lines, LINE_VERTEX_DTYPE),
        )
        for target, block, dtype in targets:
            # a copy: the batch reuses its arrays for the next world
            target.add_block(_faded(block, dtype, alpha, only, hide, show))

    def draw_liquid(self, positions, radius, color=None):
        """drops at `positions` ((k, 2) world points) drawn as one liquid (metaballs,
        blobs of `radius` that merge); `color`: the water's (r, g, b), 0..1 (liquids of
        different colors are drawn apart: they don't merge)"""
        color = tuple(color or (0.3, 0.6, 0.95))
        if color not in self.liquids:
            self.liquids[color] = GLLiquid(self.camera)
        self.liquids[color].add(np.asarray(positions, dtype=np.float32).reshape(-1, 2), radius, color)

    def flush(self):
        """draw what is queued so far (each batch empties itself): what is drawn after
        this lies on top of it (webgym: the ghosts first, behind the current policy)"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.solid_circles.draw()
        self.solid_capsules.draw()
        self.solid_polygons.draw()
        self.circles.draw()
        self.lines.draw()
        self.points.draw()

    def begin_draw(self):
        self._draw_start_time = time.perf_counter()  # start timing
        # self.update_settings()
        self.background.draw()

    def end_draw(self):
        # the GL state is set once per frame, not per shape type (every GL call
        # costs, especially python -> wasm -> JS in the browser): blending for all
        # shapes, a reset at the end (the imgui frontend draws after us)
        self.flush()
        for liquid in self.liquids.values():  # (over everything else)
            liquid.draw()
        glDisable(GL_BLEND)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

        # Draw debug strings
        for pos, text, color in self.debug_strings if self.draw_text else ():
            screen_pos = self.camera.convert_world_to_screen(Vec2(*pos))
            # Convert hex color to RGB float values
            r = ((color >> 16) & 0xFF) / 255.0
            g = ((color >> 8) & 0xFF) / 255.0
            b = (color & 0xFF) / 255.0
            self.draw_text(screen_pos, text, (r, g, b))
        self.debug_strings.clear()

        # Calculate and update draw performance metrics
        # elapsed = (time.perf_counter() - self._draw_start_time) * 1000.0  # elapsed ms
        # smoothing = 0.5  # state.perf.smoothing_avg
        # state.perf.draw_ms = elapsed
        # state.perf.draw_ms_avg *= smoothing
        # state.perf.draw_ms_avg += elapsed * (1 - smoothing)
        # state.perf.draw_ms_max = max(state.perf.draw_ms_max, elapsed)

    # colors are passed on as they come (int, HexColor, (r, g, b)): the add_* methods
    # convert them once with the cached draw.rgba8

    def draw_polygon(self, points, color):
        # Draw polygon outlines by connecting points in order
        n = len(points)
        for i in range(n):
            self.lines.add_line(points[i], points[(i + 1) % n], color)

    def draw_solid_polygon(self, transform, points, radius: float, color):
        # Delegate to solid_polygons; pass the raw b2Transform from the Transform wrapper
        self.solid_polygons.add_polygon(transform, points, len(points), radius, color)

    def draw_circle(self, center, radius: float, color):
        # Queue border circle drawing
        self.circles.add_circle(center, radius, color)

    def draw_segment(self, p1, p2, color):
        # Draw a line segment between two points
        self.lines.add_line(p1, p2, color)

    def draw_point(self, p, size: float, color):
        # Draw a point as a small circle
        self.points.add_point(p, size, color)

    def draw_string(self, p, s: str, color=1):
        """Store debug string for rendering during end_frame"""
        self.debug_strings.append((p, s, ensure_hex(color)))

    def draw_capsule(self, p1, p2, radius: float, color):
        # Draw capsule outline by adding a capsule (the same as solid capsule here)
        self.solid_capsules.add_capsule(p1, p2, radius, color)

    def draw_solid_capsule(self, p1, p2, radius: float, color):
        # Draw a filled capsule
        self.solid_capsules.add_capsule(p1, p2, radius, color)

    def draw_solid_circle(self, transform, radius: float, color):
        # Queue solid circle drawing; pass the underlying b2Transform
        self.solid_circles.add_circle(transform, radius, color)

    def draw_transform(self, transform):
        # Draw coordinate axes. Use a fixed scale.
        scale = 0.5
        p = transform.p
        x_axis = transform.transform_point((scale, 0))
        y_axis = transform.transform_point((0, scale))
        self.lines.add_line(p, x_axis, 0xFF0000)  # red for x-axis
        self.lines.add_line(p, y_axis, 0x00FF00)  # green for y-axis
