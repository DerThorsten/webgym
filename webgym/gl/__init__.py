# GL drawing of pyb2d3 worlds, copied from pyb2d3/examples/pyb2d3_sandbox_opengl
# (draw.py, shader.py, debug_draw_gl.py, shaders/). The same code runs natively
# (OpenGL 3.3) and in the browser (WebGL2 via PyOpenGL, also on an OffscreenCanvas).

import OpenGL

# No glGetError after every GL call: PyOpenGL reads the flag when OpenGL.GL is
# imported the first time, so import this package before OpenGL.GL elsewhere.
# In the browser every glGetError is a sync with the GPU process (~35 us per call).
OpenGL.ERROR_CHECKING = False
