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


def test_human_controller_and_delayed_reset():
    """human: the right arrow pushes the cart right; after the episode ends, the next
    one starts only after a pause"""
    app = make_app()
    app.command("controller", "human")
    app.event("keydown", key="right")
    steps = 0
    while app.end_timer is None:
        app._watch_step()
        steps += 1
    assert app.env.cart.position[0] > 0.05
    app.event("keyup", key="right")
    assert app.keys == set()
    app._watch_step()
    assert app.ended and app.episodes == 1  # paused, no reset yet
    for _ in range(int(60 * 1.1)):
        app._watch_step()
    assert app.episodes == 2 and not app.ended  # a new episode


def test_without_auto_reset_the_env_goes_on():
    """auto reset off: after the end the agent / human keeps control, no new episode"""
    app = make_app()
    app.command("auto_reset", False)
    for _ in range(600):
        app._watch_step()
    assert app.ended and app.episodes == 1 and app.end_timer is None
    app.command("auto_reset", True)  # switched back on: restarts after the pause
    for _ in range(int(60 * 1.1)):
        app._watch_step()
    assert app.episodes == 2


def test_no_world_leak_at_max_population():
    """box2d has at most 128 worlds: train / stop / switch envs at the max population
    many times, every world must be valid (populations are freed right away)"""
    from webgym.app import MAX_POPULATION

    app = make_app()
    app.command("population", 10_000)  # capped
    assert app.population == MAX_POPULATION
    for name in ["CartPole", "LunarLander", "Walker", "CartPole"] * 2:
        app.command("env", name)
        app.command("train")
        app._train_steps()
        assert all(env.world.is_valid for env in app.envs)
        app.command("stop")
    assert app.env.world.is_valid


def test_too_many_worlds_raise():
    """an invalid world (all worlds in use) is an error, not silent memory corruption"""
    import pytest
    import pyb2d3 as b2d

    from webgym.core import new_world

    worlds = []
    with pytest.raises(RuntimeError, match="too many box2d worlds"):
        for _ in range(b2d.MAX_WORLDS + 1):
            worlds.append(new_world())
    for w in worlds:
        w.destroy()


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
