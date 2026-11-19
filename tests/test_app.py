"""The shared app without GL: pointer / key events, commands, training via the frame
logic, resume, the leader."""

import json

import numpy as np
import pyb2d3 as b2d
import pytest

from webgym.app import App


def screen_of(app, world_pos):
    """drawing-buffer px of a world point"""
    p = app.camera.convert_world_to_screen(b2d.Vec2(*world_pos))
    return p.x, p.y


def make_app(**kwargs):
    app = App(**kwargs)
    app.camera.set_size(800, 600)
    return app


def cart_x(app):
    return float(app.env.cart.position[0, 0])


def test_drag_the_cart_in_watch_mode():
    """grab the cart, move the pointer right: the cart follows"""
    app = make_app()
    app.command("show_n", 1)  # (one cart: the grabbed one is agent 0)
    start = tuple(app.env.cart.position[0])
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


def test_delayed_reset():
    """with auto reset the next episode starts only after a pause"""
    app = make_app()
    assert app.auto_reset
    while app.end_timer is None:
        app._watch_step()
    app._watch_step()
    assert app.ended and app.episodes == 1  # paused, no reset yet
    assert len(app.scores) == 1 and app.state()["scores"] == [round(app.scores[0], 1)]
    assert not app.state()["new_best"]  # the first score is no "new best"
    for _ in range(int(60 * 1.1)):
        app._watch_step()
    assert app.episodes == 2 and not app.ended  # a new episode


def test_without_auto_reset_the_env_goes_on():
    """auto reset off: after the end the agent keeps control, no new episode; R starts
    a new one"""
    app = make_app()
    app.command("auto_reset", False)
    for _ in range(600):
        app._watch_step()
    assert app.ended and app.episodes == 1 and app.end_timer is None
    app.event("keydown", key="r")  # R: a new episode
    assert not app.ended and app.env.steps == 0


def test_train_stop_continue_new_run():
    """train; Stop keeps the state: Train continues the iterations and the history;
    New run starts over; no dragging while training"""
    app = make_app()
    app.command("train")
    assert app.training and app.state()["mode"] == "train"
    app.event("down", *screen_of(app, tuple(app.pop_env.cart.position[0])))
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
    app.command("new_run")  # (starts over and trains right away)
    assert app.history == [] and app.params is None and app.train_state is None
    assert app.training
    app.command("algo", "es")  # (switched while training: the new algorithm trains at once)
    assert app.algo == "es" and app.training and app.history == []
    app.command("stop")
    app.command("algo", "cem")  # (not training: it only switches)
    assert app.algo == "cem" and not app.training


def test_checkpoint_restore_continue():
    """a checkpoint restores env, policy, history and the trainer's state: training
    continues with the next iteration (also in a new app, as after a reload)"""
    app = make_app(env="LunarLander", population=16)
    app.command("train")
    while len(app.history) < 2:
        app._train_steps()
    app.command("stop")
    data = app.checkpoint()
    assert data[:8] and len(data) < 100_000  # small

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
    snapshot); the score is the current policy's"""
    trainer = make_app(population=16)
    trainer.command("train")
    while len(trainer.history) < 2:
        trainer._train_steps()
    assert trainer.top_params is not None and len(trainer.top_params) >= 8
    display = make_app()
    assert display.env.n == 1  # (by default: the current policy alone)
    display.command("show_n", 8)
    display.load_policy(trainer.checkpoint())
    assert display.env.n == 8
    params = display._watch_params()
    assert np.array_equal(params[0], trainer.params) and np.array_equal(params[1], trainer.top_params[0])
    while not display.ended:
        display._watch_step()
    assert len(display.scores) == 1 and display.scores[0] == display.env.episode_return[0]
    display.command("show_n", 3)
    assert display.env.n == 3


def test_watched_policies_start_alike():
    """the policies you watch all get the same (random) start, a new one each episode;
    in training every agent gets its own"""
    app = App(env="LunarLander", seed=0)
    starts = []
    for _ in range(2):
        app.env.reset()
        pos = app.env.lander.position
        assert np.allclose(pos, pos[0])
        starts.append(pos[0].copy())
    assert not np.allclose(starts[0], starts[1])
    app.command("train")
    app.pop_env.reset(seed=0)
    pos = app.pop_env.lander.position
    assert len(np.unique(pos[:, 0])) == app.population


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


def test_non_batchable_env_trains_in_separate_worlds(monkeypatch):
    """an env whose agents change the world (here CartPole, marked so): one agent in
    view, training with a population of separate worlds (at most MAX_SEPARATE); the
    leader's world is drawn"""
    from webgym.core import ENVS, MAX_SEPARATE

    monkeypatch.setattr(ENVS["CartPole"], "batchable", False)
    app = make_app(env="CartPole", population=64)
    assert app.env.n == 1
    app.command("train")
    assert app.pop_env.n == MAX_SEPARATE
    while not app.history:
        app._train_steps()
        app._update_view()
    assert app.pop_env.world is app.pop_env.envs[app.leader or 0].world
    app.command("stop")
    assert app.env.n == 1


@pytest.mark.parametrize("env", ["WaterCannon", "LunarLander", "Walker"])
def test_show_several_policies(env):
    """several policies watched side by side start alike (randomize with the same-start
    rng: every kind of draw must work)"""
    app = make_app(env=env)
    app.command("show_n", 3)
    for _ in range(5):
        app._watch_step()


def test_toys():
    """the Toys tab: a toy turns auto reset off and shows one policy; a click drops a box
    / a ball or sets off an explosion; rain adds drops; a reset clears them all"""
    app = make_app(env="LunarLander")
    app.command("show_n", 3)
    app.command("toy_mode", True)  # (the page: the Toys tab was opened)
    assert app.show_n == 1 and not app.auto_reset and app.env.n == 1
    assert app.state()["toys"]  # (no scores: no time limit applies)
    while not app.env.done[0]:
        app._watch_step()
    app._watch_step()
    assert app.scores == []
    app.command("toy_mode", False)  # (another tab: scores again)
    assert not app.state()["toys"]
    app.command("show_n", 3)
    app.command("auto_reset", True)
    app.command("toy", "box")  # (a toy: the same)
    assert app.show_n == 1 and not app.auto_reset and app.env.n == 1
    start = app.env.world.counters.body_count
    app.event("down", 400, 100)
    app.event("up", 400, 100)
    app.command("toy", "ball")
    app.event("down", 300, 100)
    assert app.env.world.counters.body_count == start + 2
    # a tool acts only in empty space: a click on the ground does nothing, a click on a
    # body grabs it (the mouse joint), whatever tool is chosen
    count = app.env.world.counters.body_count
    ground = app.camera.convert_world_to_screen(b2d.Vec2(0.0, -0.5))
    app.event("down", ground[0], ground[1])
    app.event("up", ground[0], ground[1])
    lander = app.camera.convert_world_to_screen(b2d.Vec2(*map(float, app.env.lander.position[0])))
    app.event("down", lander[0], lander[1])
    assert app.mouse_drag.active  # (grabbed: no box dropped)
    app.event("up", lander[0], lander[1])
    assert app.env.world.counters.body_count == count  # (nothing new: not on the ground, not on the lander)
    app.command("toy", "explosion")
    app.event("down", 400, 300)
    assert len(app.blasts) == 1  # (drawn for a moment: rings and a flash)
    app.command("rain", True)
    for _ in range(10):
        app._watch_step()
    assert app.env.world.counters.body_count > start + 2
    app.command("reset")  # (the world back at its start: the toys are gone)
    assert app.env.world.counters.body_count == start
    for _ in range(3):
        app._watch_step()  # (rain goes on; its old drops are gone)
    from webgym import toys as toy_module

    toys = [c for c in app.controls() if c["section"] == "toys"]
    assert {c["name"] for c in toys} == {"toy", "rain", "toys_hint", *(f"toy.{k}" for k in toy_module.PARAMS)}
    app.command("toy.size", 1.2)  # (the sliders: a bigger, heavier box)
    app.command("toy.weight", 3)
    big = toy_module.thing(app.env.world, (0, 5), b2d.box(1.2, 1.2), app.toy_params)
    assert big.mass == pytest.approx(3 * 2.4 * 2.4, rel=1e-3)
    # the other tools: a plank, a ramp, a step and a seesaw (wedge + plank): all dynamic
    for tool, x, bodies in (("plank", 100, 1), ("ramp", 300, 1), ("step", 500, 1), ("seesaw", 700, 2)):
        count = app.env.world.counters.body_count
        app.command("toy", tool)
        app.event("down", x, 150)  # (each in empty space)
        app.event("up", x, 150)
        assert app.env.world.counters.body_count == count + bodies, tool
    # weather: wind and gravity
    app.command("toy.wind", 10)
    app.command("toy.gravity", 0.5)
    app._watch_step()
    assert tuple(app.env.world.gravity) == pytest.approx((10, app.env.gravity[1] * 0.5))


def test_watched_episodes_get_new_seeds():
    """every watched episode starts anew (another random start)"""
    app = make_app(env="Walker")
    first = app.env.hull.linear_velocity.copy()
    app.command("reset")
    assert not np.array_equal(first, app.env.hull.linear_velocity)


def test_suggested_algorithms():
    """opening an env takes its suggestion for the time budget (algorithm + parameters);
    an algorithm given explicitly (the URL's) is kept; the budget switches; ★ marks"""
    app = make_app(env="LunarLander")
    assert app.algo == "cem"
    app.command("budget", "long")
    assert app.algo == "ppo_torch" and app.hp["ppo_torch"]["hidden"] == 64
    algo = next(c for c in app.controls() if c["name"] == "algo")
    labels = dict(algo["options"])
    assert labels["td3_torch"].startswith("★") and not labels["es"].startswith("★")
    app.command("env", "Walker")  # (another env: its suggestion for the budget, with its parameters)
    assert app.algo == "ppo_torch" and app.hp["ppo_torch"]["hidden"] == 64
    assert make_app(env="LunarLander", algo="es").algo == "es"


def test_checkpoint_of_an_unknown_env():
    """a checkpoint of an env webgym no longer has (renamed, removed): refused with a
    message (the page puts it in the log), the app goes on"""
    from webgym import checkpoints

    app = make_app()
    data = app.checkpoint()
    tensors, meta = checkpoints.loads(data)
    with pytest.raises(ValueError, match="no longer has"):
        app.restore(checkpoints.dumps(tensors, dict(meta, env="Tank")))
    assert app.env_name == "CartPole"


def test_env_settings():
    """an env setting (the water cannon's objects) is a select in the page's World tab; a
    change rebuilds the env (more objects: more observations) as a new run; checkpoints
    carry the settings and bring them back; the env key names them"""
    app = make_app(env="WaterCannon")
    control = next(c for c in app.controls() if c["name"] == "setting.objects")
    assert control["type"] == "select" and control["options"] == [1, 2, 3] and control["value"] == 1
    # (the World tab: the env and its settings; the page's icons above the tabs: training,
    # the switches, how you watch)
    sections = {c["name"]: c["section"] for c in app.controls()}
    assert sections["env"] == sections["setting.objects"] == "environment"
    assert {n for n, s in sections.items() if s == "bar"} == {"auto_reset", "train", "new_run", "reset", "follow",
                                                              "decorate", "show_n", "replay_speed"}
    assert "watch" not in sections.values()
    assert app.env_key == "WaterCannon · objects 1"
    app.command("train")
    while not app.history:
        app._train_steps()
    one_object = app.checkpoint()
    app.command("setting.objects", "2")  # (the page sends text)
    assert app.settings == {"objects": 2} and app.env.objects == 2 and not app.history
    assert len(app.env.obs_names) == 15 and app.policy.sizes[0] == 15  # (2 objects, 2 cannons)
    assert app.env_key == "WaterCannon · objects 2"
    app.command("train")
    while not app.history:
        app._train_steps()
    assert app.pop_env.objects == 2
    app.restore(one_object)  # (back to one object, with its run)
    assert app.settings == {"objects": 1} and app.history and app.policy.sizes[0] == 8
    display = make_app()
    display.load_policy(app.checkpoint())
    assert display.env_key == "WaterCannon · objects 1"
    for _ in range(3):
        display._watch_step()


def test_simulation_helpers():
    """CEM with simulation helpers: the population is split (this env runs its share, the
    helpers the rest: connected directly here, in the browser through the page); the
    helpers' returns come back and training learns as without them; a helper that never
    answers: its candidates count as failed, training goes on"""
    from webgym import helpers as h

    answers = []
    pool = h.Pool(lambda message: answers.append(h.evaluate(message)))
    pool.count = 3
    app = make_app(population=64, helpers=pool)
    app.command("train")
    assert app.pop_env.n == 16 and app.pop_env.population == 64
    while len(app.history) < 12:
        app._train_steps()
        while answers:  # (the page would deliver them between the steps)
            pool.receive(answers.pop())
    first, last = app.history[0]["mean_return"], app.history[-1]["mean_return"]
    assert last > 3 * first  # (CartPole learns, with all 64)
    app.command("helpers", 0)  # (the helpers went away: all in this env again)
    assert app.pop_env.n == 64

    from webgym.algos.cem import rollout
    from webgym.core import make
    from webgym.policies import policy_for

    silent = h.Pool(lambda message: None)  # (a helper that never answers)
    silent.count = 1
    env = make("CartPole", n=4)
    env.helpers, env.population = silent, 8
    policy = policy_for(env)
    h.TIMEOUT, timeout = 0.0, h.TIMEOUT
    try:
        episode = rollout(env, policy, np.zeros((8, policy.num_params)), 0)
        while True:
            next(episode)
    except StopIteration as end:
        returns = end.value
    finally:
        h.TIMEOUT = timeout
    assert len(returns) == 8 and (returns[4:] == returns[:4].min()).all()


def test_pretrained_from_the_command_line(tmp_path):
    """python -m webgym.cli ENV --save FILE trains headless and writes a page checkpoint:
    the app restores it like its own (the page's pretrained runs)"""
    from webgym.cli import train_and_save

    path = tmp_path / "pretrained" / "CartPole-cem.safetensors"
    train_and_save(path, "CartPole", iterations=3, population=16, log=lambda line: None)
    app = make_app()
    app.restore(path.read_bytes())
    n = len(app.history)  # (at least 3: a training chunk may run a few more)
    assert n >= 3 and [h["iteration"] for h in app.history] == list(range(1, n + 1)) and app.params is not None
    app.command("train")  # (it goes on from there)
    while len(app.history) < n + 1:
        app._train_steps()
    assert app.history[n]["iteration"] == n + 1  # (continued: no restart at 1)


def test_checkpoint_of_another_env_version_is_refused():
    """a checkpoint whose policy doesn't fit the env as it is now (the env's code changed:
    other observations) is refused with a clear error; the app goes on as it was"""
    from webgym import checkpoints

    app = make_app()
    app.command("train")
    while not app.history:
        app._train_steps()
    tensors, meta = checkpoints.loads(app.checkpoint())
    meta["policy_sizes"] = [5, *meta["policy_sizes"][1:]]  # (CartPole sees 4 values)
    other = make_app()
    with pytest.raises(ValueError, match="another version of the env"):
        other.restore(checkpoints.dumps(tensors, meta))
    with pytest.raises(ValueError, match="another version of the env"):
        other.load_policy(checkpoints.dumps(tensors, meta))
    for _ in range(5):
        other._watch_step()


def test_checkpoint_of_an_unknown_algorithm_is_refused():
    """a checkpoint of an algorithm webgym no longer has (e.g. ppo_jax) gives a clear
    error, the app stays as it was"""
    from webgym import checkpoints

    app = make_app()
    app.command("train")
    while not app.history:
        app._train_steps()
    app.command("stop")
    tensors, meta = checkpoints.loads(app.checkpoint())
    data = checkpoints.dumps(tensors, dict(meta, algo="ppo_jax"))
    with pytest.raises(ValueError, match="no longer has"):
        make_app().restore(data)


def test_camera_follows_the_walker():
    """'Camera follows': the camera glides to the walker the env suggests; off: back to
    the fixed view; CartPole suggests nothing (the option is disabled)"""
    app = make_app()
    assert next(c for c in app.controls() if c["name"] == "follow")["disabled"]
    app.command("env", "Walker")
    app.command("follow", True)
    app.env.hull.position = app.env.hull.position + np.array([5.0, 0.0])  # (walked away)
    for _ in range(100):
        app._update_view()
    assert abs(app.camera.center[0] - app.env.camera_target(0)[0]) < 0.01
    app.command("follow", False)
    assert tuple(app.camera.center) == app.env.view_center


def test_layers():
    """'layers' x 'hidden' make the network (a new run); the controls show its shape"""
    app = make_app()
    assert app.policy.sizes == [4, 16, 16, 1]  # (CEM: two layers by default)
    app.command("hp.layers", 3)
    app.command("hp.hidden", 8)
    assert app.policy.sizes == [4, 8, 8, 8, 1]
    hint = next(c for c in app.controls() if c["name"] == "network")
    assert "4 → 8 → 8 → 8 → 1" in hint["text"] and f"{app.policy.num_params:,} weights" in hint["text"]
    app.command("train")
    while not app.history:
        app._train_steps()
    app.command("stop")
    fresh = make_app()  # (a restored run shows its network in the sliders)
    fresh.restore(app.checkpoint())
    assert fresh.hp["cem"]["layers"] == 3 and fresh.hp["cem"]["hidden"] == 8


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
    worlds = [b2d.World() for _ in range(b2d.MAX_WORLDS - 10)]  # (raises if they leaked)
    for w in worlds:
        w.destroy()


def test_neat_in_the_app():
    """NEAT: trains (also with simulation helpers), its network view is a graph whose
    outputs are what the policy computes; checkpoint -> restore in a fresh app continues
    the run; the display takes its snapshots"""
    from webgym import helpers as h

    answers = []
    pool = h.Pool(lambda message: answers.append(h.evaluate(message)))
    pool.count = 1
    app = make_app(population=32, helpers=pool)
    app.command("algo", "neat")
    app.command("hp.add_node", 0.5)
    assert app.policy.kind == "graph" and "hidden neurons" in app._network_hint()
    app.command("train")
    assert app.pop_env.n == 16
    while len(app.history) < 6:
        app._train_steps()
        while answers:
            pool.receive(answers.pop())
    app.command("stop")
    net = app.network()
    assert net["graph"] and net["inputs"][2] == "pole angle" and net["outputs"] == ["force"]
    output = next(n for n in net["nodes"] if n["key"] == "o0")
    assert np.isclose(output["value"], app.policy(app.params, app.obs[:1])[0, 0], atol=1e-3)
    assert all(0 <= a < len(net["nodes"]) and 0 <= b < len(net["nodes"]) for a, b, _ in net["edges"])
    assert max(len(g["nodes"]) for g in app.train_state["genomes"]) > 1  # (neurons have grown)

    fresh = make_app()
    fresh.restore(app.checkpoint())
    assert fresh.algo == "neat" and fresh.policy.kind == "graph" and fresh.hp["neat"]["max_hidden"] == 16
    assert np.array_equal(fresh.params, app.params) and fresh.network()["edges"] == net["edges"]
    fresh.command("train")
    while len(fresh.history) < 8:
        fresh._train_steps()
    assert fresh.history[-1]["iteration"] == 8

    display = make_app()
    display.load_policy(app.checkpoint())
    assert display.policy.kind == "graph" and np.array_equal(display.params, app.params)
    display.load_policy(make_app(population=8).checkpoint())  # (back to an MLP)
    assert display.policy.kind == "mlp"


def test_view_logic_without_gl():
    """the view update (leader, camera) runs in both modes; the walker is followed"""
    app = make_app(env="Walker", population=8)
    app._update_view()
    app.command("train")
    for _ in range(30):
        app._train_steps()
        app._update_view()
    assert app.leader is not None and np.isfinite(app.camera.center[0])
