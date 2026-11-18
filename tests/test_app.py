"""The shared app without GL: pointer / key events, commands, training via the frame
logic, resume, the leader."""

import json

import numpy as np
import pyb2d3 as b2d
import pytest

from webgym.app import App
from webgym.core import new_world


def screen_of(app, world_pos):
    """drawing-buffer px of a world point"""
    p = app.camera.convert_world_to_screen(b2d.Vec2(*world_pos))
    return p.x, p.y


def make_app(**kwargs):
    app = App(**kwargs)
    app.camera.set_size(800, 600)
    return app


def cart_x(app):
    return float(app.env.carts.position[0, 0])


def test_drag_the_cart_in_watch_mode():
    """grab the cart, move the pointer right: the cart follows"""
    app = make_app()
    app.command("show_n", 1)  # (one cart: the grabbed one is agent 0)
    start = tuple(app.env.carts.position[0])
    app.event("down", *screen_of(app, start))
    assert app.mouse_drag.active
    app.event("move", *screen_of(app, (start[0] + 1.5, start[1])))
    for _ in range(30):
        app._watch_step()
    assert cart_x(app) > start[0] + 0.5
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
    while app.end_timer is None:
        app._watch_step()
    assert cart_x(app) > 0.05
    app.event("keyup", key="right")
    assert app.keys == set()
    app._watch_step()
    assert app.ended and app.episodes == 1  # paused, no reset yet
    assert len(app.scores) == 1 and app.state()["scores"] == [round(app.scores[0], 1)]
    assert not app.state()["new_best"]  # the first score is no "new best"
    for _ in range(int(60 * 1.1)):
        app._watch_step()
    assert app.episodes == 2 and not app.ended  # a new episode


def test_without_auto_reset_the_env_goes_on():
    """auto reset off: after the end the agent / human keeps control, no new episode;
    R starts a new one"""
    app = make_app()
    app.command("auto_reset", False)
    for _ in range(600):
        app._watch_step()
    assert app.ended and app.episodes == 1 and app.end_timer is None
    app.command("controller", "human")
    app.command("auto_reset", False)
    for _ in range(100):
        app._watch_step()
    app.event("keydown", key="right")  # still controls the cart after the end
    x = cart_x(app)
    for _ in range(20):
        app._watch_step()
    assert cart_x(app) > x
    app.event("keyup", key="right")
    app.event("keydown", key="r")  # R: a new episode
    assert not app.ended and app.env.steps == 0


def test_train_stop_continue_new_run():
    """train; Stop keeps the state: Train continues the iterations and the history;
    New run starts over; no dragging while training"""
    app = make_app()
    app.command("train")
    assert app.training and app.state()["mode"] == "train"
    app.event("down", *screen_of(app, tuple(app.pop_env.carts.position[0])))
    assert not app.mouse_drag.active  # pan only while training
    app.event("up", 0, 0)
    while len(app.history) < 3:
        app._train_steps()
    app.command("stop")
    assert not app.training and app.params is not None
    assert any(c.get("label") == "Continue training" for c in app.controls())
    for _ in range(100):
        app._watch_step()
    app.command("train")
    while len(app.history) < 5:
        app._train_steps()
    iterations = [h["iteration"] for h in app.history]
    assert iterations == list(range(1, len(iterations) + 1))  # continued, no restart
    app.command("new_run")
    assert app.history == [] and app.params is None and app.train_state is None


def test_checkpoint_restore_continue():
    """a checkpoint restores env, policy, history and the trainer's state: training
    continues with the next iteration (also in a new app, as after a reload)"""
    app = make_app(env="LunarLander", population=16)
    app.command("train")
    while len(app.history) < 2:
        app._train_steps()
    app.command("stop")
    data = app.checkpoint()
    assert data[:8] and len(data) < 50_000  # small

    fresh = make_app()  # CartPole: restore switches the env
    fresh.restore(data)
    assert fresh.env_name == "LunarLander" and fresh.population == 16
    assert np.array_equal(fresh.params, app.params)
    assert fresh.history == app.history
    assert np.array_equal(fresh.train_state["mean"], app.train_state["mean"])
    fresh.command("train")
    while len(fresh.history) < 3:
        fresh._train_steps()
    assert fresh.history[-1]["iteration"] == 3


def test_display_takes_snapshots():
    """two workers: the display app takes the trainer's policy (a checkpoint) without
    restarting its episode; a snapshot of another env switches the env"""
    trainer = make_app(population=16)
    trainer.command("train")
    while len(trainer.history) < 2:
        trainer._train_steps()
    display = make_app()
    for _ in range(10):
        display._watch_step()
    steps = display.env.steps
    display.load_policy(trainer.checkpoint())
    assert np.array_equal(display.params, trainer.params) and display.env.steps == steps
    assert "iteration" in display.state()["policy"]
    lander = make_app(env="LunarLander", population=8)
    display.load_policy(lander.checkpoint())  # (no policy yet: zero actions)
    assert display.env_name == "LunarLander" and display.params is None


def test_show_n_best_policies():
    """watching shows the current policy + the trainer's best candidates (in the
    snapshot); the score is the current policy's; a human plays alone"""
    trainer = make_app(population=16)
    trainer.command("train")
    while len(trainer.history) < 2:
        trainer._train_steps()
    assert trainer.top_params is not None and len(trainer.top_params) >= 8
    display = make_app()
    display.load_policy(trainer.checkpoint())
    assert display.env.n == 8
    params = display._watch_params()
    assert np.array_equal(params[0], trainer.params) and np.array_equal(params[1], trainer.top_params[0])
    while not display.ended:
        display._watch_step()
    assert len(display.scores) == 1 and display.scores[0] == display.env.episode_return[0]
    display.command("controller", "human")
    assert display.env.n == 1
    display.command("controller", "agent")
    display.command("show_n", 3)
    assert display.env.n == 3


def test_network_view():
    """the network view: sizes, weights, activations of agent 0 (the output = what the
    policy computes), names; None without a policy"""
    app = make_app()
    assert app.network() is None
    trainer = make_app(population=16)
    trainer.command("train")
    while not trainer.history:
        trainer._train_steps()
    app.load_policy(trainer.checkpoint())
    net = app.network()
    assert net["sizes"] == app.policy.sizes and len(net["weights"]) == len(net["sizes"]) - 1
    assert [len(a) for a in net["activations"]] == net["sizes"]
    expected = app.policy(app.params, app.obs[:1])[0]
    assert np.allclose(net["activations"][-1], expected, atol=1e-3)
    assert net["inputs"][2] == "pole angle" and net["outputs"] == ["force"]


def test_code_view_and_apply():
    """the Code section: the env's / algo's source; applying it reloads the module (the
    env registers again, is rebuilt); a syntax error leaves the file alone"""
    app = make_app()
    code = app.code()
    assert code["env"]["path"].endswith("cartpole.py") and "register_env" in code["env"]["text"]
    assert code["algo"]["path"].endswith("cem.py")
    old_class = type(app.env)
    app.command("code", json.dumps({"kind": "env", "text": code["env"]["text"]}))  # (unchanged)
    assert type(app.env) is not old_class and type(app.env).__name__ == old_class.__name__
    with pytest.raises(SyntaxError):
        app.command("code", json.dumps({"kind": "algo", "text": "def train(:"}))
    assert app.code() == code


def test_non_batchable_env_disables_population(monkeypatch):
    """an env whose agents change the world (batchable = False): no population /
    show-policies slider to move"""
    app = make_app()
    disabled = lambda: {c["name"] for c in app.controls() if c.get("disabled")}  # noqa: E731
    assert not {"population", "show_n"} & disabled()
    monkeypatch.setattr(type(app.env), "batchable", False)
    assert {"population", "show_n"} <= disabled()


def test_gpu_checkbox_only_for_gpu_algorithms():
    """'Use GPU' can only be ticked for an algorithm that trains on the GPU (PPO jax)"""
    app = make_app()
    use_gpu = lambda: next(c for c in app.controls() if c["name"] == "use_gpu")  # noqa: E731
    assert use_gpu()["disabled"]  # (CEM)
    app.command("algo", "ppo_jax")
    assert not use_gpu()["disabled"]


def test_camera_follows_the_walker():
    """'Camera follows': the camera glides to the walker the env suggests; off: back to
    the fixed view; CartPole suggests nothing (the option is disabled)"""
    app = make_app()
    assert next(c for c in app.controls() if c["name"] == "follow")["disabled"]
    app.command("env", "Walker")
    app.command("follow", True)
    app.env.hulls.position = app.env.hulls.position + np.array([5.0, 0.0])  # (walked away)
    for _ in range(100):
        app._update_view()
    assert abs(app.camera.center[0] - app.env.camera_target(0)[0]) < 0.01
    app.command("follow", False)
    assert tuple(app.camera.center) == app.env.view_center


def test_throw_things_at_the_agent():
    """torture: a thrown ball / box flies at the watched agent and hits it (its hull is
    pushed); the next episode has none of them; automatic throwing every few seconds"""
    app = make_app(env="Walker")
    app.command("show_n", 1)
    for _ in range(30):  # (let it stand)
        app._watch_step()
    body = app.throw()
    before = np.array(app.env.hulls.linear_velocity[0])
    hit = False
    for _ in range(60):
        app._watch_step()
        hit |= np.abs(np.array(app.env.hulls.linear_velocity[0]) - before).max() > 0.3
    assert hit, "the thrown thing did not hit the walker"
    assert body.is_valid
    app.command("reset")
    assert not body.is_valid  # (gone with the snapshot reset)
    app.command("throw_every", 3)  # every 0.3 s
    for _ in range(60):
        app._watch_step()
    assert app._throw_timer < 0.3


def test_replay_speed():
    """replay speed: 0.25x steps every 4th frame, 4x four times per frame"""
    app = make_app()
    app.debug_draw = None
    for speed, frames, steps in ((0.25, 8, 2), (4, 2, 8)):
        app.command("reset")
        app.command("replay_speed", [0.25, 0.5, 1, 2, 4, 8].index(speed))
        app._draw = lambda w, h: None  # (no GL in the tests)
        for _ in range(frames):
            app.frame(800, 600)
        assert app.env.steps == steps


def test_safetensors_roundtrip():
    """the checkpoint format: arrays + metadata survive, the header is standard"""
    import json
    import struct

    from webgym import checkpoints

    tensors = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.array([1.5, -2.0]),
               "scalar": np.array(0.25)}  # (0-d: e.g. DQN's epsilon)
    data = checkpoints.dumps(tensors, {"x": [1, 2]})
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    assert header["a"] == {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}
    back, meta = checkpoints.loads(data)
    assert meta == {"x": [1, 2]}
    assert all(np.array_equal(back[k], tensors[k]) and back[k].dtype == tensors[k].dtype for k in tensors)
    assert back["scalar"].shape == () and float(back["scalar"]) == 0.25


def test_population_change_keeps_training():
    """changing the population while training goes on training (same run, new size)"""
    app = make_app(population=16)
    app.command("train")
    while len(app.history) < 2:
        app._train_steps()
    app.command("population", 32)
    assert app.training and app.pop_env.n == 32
    while len(app.history) < 4:
        app._train_steps()
    assert [h["iteration"] for h in app.history][:4] == [1, 2, 3, 4]


def test_leader_is_sticky():
    """within an iteration the leader keeps its highlight until its episode ends (no
    flicker); a new iteration (all agents reset) picks a new one"""
    app = make_app(env="LunarLander", population=16)
    app.command("train")
    changes = 0
    previous, iteration = None, 0
    for _ in range(300):
        app._train_steps(max_steps=1)
        app._update_view()
        if len(app.history) == iteration and previous is not None and app.leader != previous:
            changes += 1
            assert app.pop_env.done[previous]  # only when the old leader's episode ended
        previous, iteration = app.leader, len(app.history)
    assert changes > 0  # (leaders do end)


def test_switching_envs_frees_worlds():
    """box2d has at most 128 worlds: switching envs / training many times must not leak"""
    app = make_app(population=64)
    for name in ["CartPole", "LunarLander", "Walker"] * 20:
        app.command("env", name)
        app.command("train")
        app._train_steps()
        app.command("stop")
    worlds = [new_world() for _ in range(b2d.MAX_WORLDS - 10)]
    assert all(w.is_valid for w in worlds)
    for w in worlds:
        w.destroy()


def test_too_many_worlds_raise():
    """an invalid world (all worlds in use) is an error, not silent memory corruption"""
    worlds = []
    with pytest.raises(RuntimeError, match="too many box2d worlds"):
        for _ in range(b2d.MAX_WORLDS + 1):
            worlds.append(new_world())
    for w in worlds:
        w.destroy()


def test_view_logic_without_gl():
    """the view update (leader, camera) runs in both modes; the walker is followed"""
    app = make_app(env="Walker", population=8)
    app._update_view()
    app.command("train")
    for _ in range(30):
        app._train_steps()
        app._update_view()
    assert app.leader is not None and np.isfinite(app.camera.center[0])
