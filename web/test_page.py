"""Load the built page (web/dist) in headless chromium: cold, then a cached reload.
Fails on console errors / page errors or if python does not start in time.
(Adapted from pyb2d3/examples/web/test_page.py.)

    python web/test_page.py [--screenshot out.png] [--install pytorch]
"""

import argparse
import functools
import http.server
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

DIST = Path(__file__).resolve().parent / "dist"
READY = "webgym loading times"


def serve(directory):
    """serve `directory` on a free port in a thread; returns the url"""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/"


def load(page, url, errors):
    """open `url`, wait for the loading-times console line; returns its text"""
    with page.expect_console_message(lambda m: READY in m.text, timeout=120_000) as msg:
        page.goto(url)
    if errors:
        raise SystemExit("errors:\n" + "\n".join(errors))
    return msg.value.text


def state(page):
    """the state panel as one line"""
    return page.inner_text("#state").replace("\n", " | ")


def iteration(page):
    """the trained policy's iteration (0: none)"""
    text = page.inner_text("#state")
    return int(text.split("iteration ")[1].split()[0]) if "iteration " in text else 0


def check_checkpoints(page, url, errors):
    """autosave + named checkpoint survive a reload; Continue training goes on; New
    run drops the autosave, the named one stays restorable"""
    page.click("button:text('Stop')")  # (the training of the default checks)
    page.wait_for_timeout(500)
    trained = iteration(page)
    assert trained > 0, state(page)
    page.fill("#ckpt-name", "my run")
    page.click("#ckpt-save")
    page.wait_for_function("document.getElementById('ckpt-list').textContent.includes('my run')", timeout=10_000)
    load(page, url, errors)  # reload: the autosave comes back
    page.wait_for_function(
        f"document.getElementById('state').textContent.includes('iteration {trained}')", timeout=60_000
    )
    print("restored after reload:", state(page))
    assert "my run" in page.inner_text("#ckpt-list")
    page.click("button:text('Continue training')")
    page.wait_for_function(
        f"(() => {{ const m = document.getElementById('state').textContent.match(/iteration (\\d+)/);"
        f" return m && Number(m[1]) > {trained}; }})()",
        timeout=30_000,
    )
    page.click("button:text('Stop')")
    print("continued:", state(page))
    page.click("button:text('New run')")
    page.wait_for_timeout(500)
    load(page, url, errors)  # no autosave any more: nothing restored
    page.wait_for_timeout(1500)
    assert "none yet" in page.inner_text("#state"), state(page)
    page.locator(".ckpt", has_text="my run").locator("button:text('Restore')").click()
    page.wait_for_function(
        f"document.getElementById('state').textContent.includes('iteration {trained}')", timeout=20_000
    )
    print("restored by name:", state(page))


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--packages", action="store_true", help="wait for pytorch / jax / webgpu")
    parser.add_argument("--query", default="", help="URL query, e.g. env=LunarLander")
    parser.add_argument("--stress", type=float, default=0, help="seconds per env at max population / speed")
    parser.add_argument("--crash", action="store_true", help="crash python (wasm trap), expect a restart")
    parser.add_argument("--converge", type=float, help="time training until the mean return reaches this")
    parser.add_argument("--checkpoints", action="store_true", help="autosave / save / restore / reload")
    parser.add_argument("--gpu-train", action="store_true", help="train PPO (jax) on the GPU (WebGPU)")
    args = parser.parse_args()
    base = serve(DIST)
    # (the page starts training by itself: the checks below start watching, train=0;
    # the autostart is checked at the end)
    url = base + "?" + "&".join(q for q in ("train=0", args.query) if q)
    # (the architecture is printed with the loading times: wasm64 by default in chromium)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--enable-unsafe-webgpu",
                  "--enable-features=Vulkan"]
        )
        page = browser.new_page(viewport={"width": 1200, "height": 700})
        errors = []
        # (python warnings go to stderr = console errors: not failures)
        page.on(
            "console",
            lambda m: m.type == "error" and not any(w in m.text for w in ("Warning", "import filelock"))
            and errors.append(m.text),
        )
        page.on("console", lambda m: ("webgym.gpu" in m.text or "GPU_ERROR" in m.text) and print(m.text[:400]))
        page.on("pageerror", lambda e: errors.append(str(e)))
        print("cold:", load(page, url, errors))
        print("reload:", load(page, url, errors))
        page.wait_for_timeout(1000)
        print("watch:", page.inner_text("#state"))
        # explanations: a click on a control's name shows it, another click hides it
        page.click("[data-help] >> text=Show policies")
        assert page.is_visible("#help-popup") and "ghosts" in page.inner_text("#help-popup")
        page.click("[data-help] >> text=Show policies")
        assert not page.is_visible("#help-popup")
        # the Code section: edit the env's source (a comment), apply: the env is rebuilt
        # with it in the worker(s), no error
        page.click("#sec-code > summary")
        page.wait_for_function("document.getElementById('code').textContent.includes('register_env')", timeout=10_000)
        page.click("#code .cm-content")
        page.keyboard.press("ControlOrMeta+End")
        page.keyboard.type("\n# edited in the browser\n")
        page.keyboard.press("ControlOrMeta+Enter")  # (= Apply)
        page.wait_for_timeout(1500)
        assert "# edited in the browser" in page.inner_text("#code")  # (the file, sent back)
        assert "applying the edited" in page.text_content("#log")  # (collapsed: no inner_text)
        page.wait_for_timeout(1000)
        assert not errors, errors
        page.click("button:text('Train')")
        page.wait_for_function(
            "document.getElementById('state').textContent.includes('mean/best')", timeout=60_000
        )
        page.wait_for_timeout(3000)
        print("train:", page.inner_text("#state"))
        if args.converge:
            # seconds of training in the browser until the population's mean return
            # reaches `converge` (from a fresh run)
            page.click("button:text('Stop')")
            page.wait_for_timeout(500)
            start = time.perf_counter()
            page.click("button:text('Train')")
            page.wait_for_function(
                """(target) => {
                const m = document.getElementById('state').textContent.match(/mean\\/best (-?\\d+)/);
                return m && Number(m[1]) >= target;
                }""",
                arg=args.converge,
                timeout=600_000,
            )
            print(f"converged (mean >= {args.converge}) after {time.perf_counter() - start:.1f} s:",
                  page.inner_text("#state").replace("\n", " | "))
        if args.gpu_train:
            # PPO (jax) on the GPU: jax loads, the update is compiled for WebGPU (compile
            # worker), training runs; the policy line says where it ran
            page.click("button:text('Stop')")
            page.select_option("#controls-training select >> nth=0", "ppo_jax")
            page.wait_for_timeout(500)
            page.locator(".control.checkbox", has_text="Use GPU").locator("input").check()
            page.wait_for_timeout(500)
            start = time.perf_counter()
            page.click("button:text('Train')")
            page.wait_for_function(
                "/ppo_jax, iteration [3-9]/.test(document.getElementById('state').textContent)"
                " || document.getElementById('status').textContent.startsWith('error')",
                timeout=600_000,
            )
            assert not page.inner_text("#status").startswith("error"), (
                page.inner_text("#state") + "\n" + page.inner_text("#status"))
            text = page.inner_text("#state").replace("\n", " | ")
            print(f"GPU training ({time.perf_counter() - start:.0f} s):", text)
            assert "(GPU: " in text and "ms/minibatch" in text, "trained on the CPU, not on the GPU"
            # the same run on the CPU (the box off): the policy line says so, with its time
            page.locator(".control.checkbox", has_text="Use GPU").locator("input").uncheck()
            page.wait_for_function("/ppo_jax, iteration \\d+ \\(CPU, /.test(document.getElementById('state').textContent)",
                                   timeout=300_000)
            print("CPU training:", page.inner_text("#state").splitlines()[1])
        if args.checkpoints:
            check_checkpoints(page, url, errors)
        if args.crash:
            # a wasm trap in the worker: the page restarts it, python runs again
            errors.clear()  # the crash itself is logged as an error on purpose
            with page.expect_console_message(lambda m: READY in m.text, timeout=60_000):
                page.evaluate("webgymCrashTest()")
            print("after crash:", page.inner_text("#status").splitlines()[0])
            assert "restarted" in page.inner_text("#status")
            page.wait_for_timeout(1000)
            errors.clear()
        if args.stress:
            # max population + max speed, train every env a while (box2d world limit,
            # memory): no errors allowed
            for env in ("CartPole", "LunarLander", "Walker", "CartPole"):
                page.select_option("#controls-environment select >> nth=0", env)
                page.wait_for_timeout(300)
                for name in ("Population", "Training speed"):
                    page.evaluate(
                        """(name) => {
                        const label = [...document.querySelectorAll('.controls .control')]
                            .find((c) => c.textContent.includes(name));
                        const input = label.querySelector('input');
                        input.value = input.max;
                        input.dispatchEvent(new Event('change'));
                        }""",
                        name,
                    )
                    page.wait_for_timeout(300)
                page.click("button:text('Train')")
                page.wait_for_timeout(args.stress * 1000)
                print(f"stress {env}:", page.inner_text("#state").replace("\n", " | "))
                page.click("button:text('Stop')")
        if args.packages:
            # pytorch, jax, webgpu (IREE) load in the background by themselves; their
            # checks (incl. the GPU probe) end up in the packages panel
            page.wait_for_function(
                "['pytorch', 'jax', 'webgpu'].every((n) => document.getElementById('package-' + n))",
                timeout=900_000,
            )
            for name in ("pytorch", "jax", "webgpu"):  # (collapsed section: text_content)
                print(page.text_content(f"#package-{name}"))
        if args.screenshot:
            page.screenshot(path=str(args.screenshot))
            section = page.locator("#network-section")
            if section.is_visible():  # (the network view, below the fold: its own picture)
                section.screenshot(path=str(args.screenshot.with_name(args.screenshot.stem + "_network.png")))
        # torture: the Crawler, three things thrown at it (no errors; a picture)
        page.select_option("#controls-environment select >> nth=0", "Crawler")
        page.wait_for_timeout(1500)
        for _ in range(3):
            page.click("button:text('Throw (T)')")
            page.wait_for_timeout(250)
        if args.screenshot:
            page.screenshot(path=str(args.screenshot.with_name(args.screenshot.stem + "_throw.png")))
        page.select_option("#controls-environment select >> nth=0", "CartPole")
        page.wait_for_timeout(500)
        # the plain page starts training by itself
        load(page, base + (f"?{args.query}" if args.query else ""), errors)
        page.wait_for_function("document.getElementById('state').textContent.includes('running')", timeout=60_000)
        print("autostart:", page.inner_text("#state").splitlines()[0])
        if errors or page.inner_text("#status").startswith("error"):
            raise SystemExit("errors:\n" + "\n".join(errors) + page.inner_text("#status"))
        browser.close()


if __name__ == "__main__":
    main()
