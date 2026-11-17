"""The shared app without GL: pointer events, commands, training via the frame logic."""

import pyb2d3 as b2d

from webgym.app import App


def screen_of(app, world_pos):
    """drawing-buffer px of a world point"""
    p = app.camera.convert_world_to_screen(b2d.Vec2(*world_pos))
    return p.x, p.y


def make_app():
    app = App()
    app.camera.set_size(800, 600)
    return app


def test_drag_the_cart_in_watch_mode():
    """grab the cart, move the pointer right: the cart follows"""
    app = make_app()
    start = tuple(app.env.cart.position)
    app.event("down", *screen_of(app, start))
    assert app.mouse_drag.active
    app.event("move", *screen_of(app, (start[0] + 1.5, start[1])))
    for _ in range(30):
        app._watch_step()
    assert app.env.cart.position[0] > start[0] + 0.5
    app.event("up", 0, 0)
    assert not app.mouse_drag.active


def test_pan_and_zoom():
    """dragging empty space pans the camera, the wheel zooms about the pointer"""
    app = make_app()
    sky = screen_of(app, (0, 3.5))
    center = tuple(app.camera.center)
    app.event("down", *sky)
    assert not app.mouse_drag.active
    app.event("move", sky[0] + 100, sky[1])
    app.event("up", 0, 0)
    assert app.camera.center[0] < center[0]
    zoom = app.camera.zoom
    app.event("wheel", 400, 300, delta=-200)
    assert app.camera.zoom < zoom


def test_train_then_watch():
    """train via commands: history grows, the watched policy is set; stop -> watch,
    no dragging while training"""
    app = make_app()
    app.command("speed", 6)  # fastest
    app.command("train")
    assert app.training and app.state()["mode"] == "train"
    app.event("down", *screen_of(app, tuple(app.envs[0].cart.position)))
    assert not app.mouse_drag.active  # pan only while training
    app.event("up", 0, 0)
    while len(app.history) < 5:
        app._train_steps()
    assert app.params is not None
    app.command("stop")
    assert not app.training
    for _ in range(100):
        app._watch_step()
    assert {c["name"] for c in app.controls()} >= {"env", "algo", "train", "speed"}
