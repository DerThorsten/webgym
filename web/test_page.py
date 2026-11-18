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
    run starts training from scratch, the named one stays restorable"""
    page.click("button:text('Stop')")  # (the training of the default checks)
    # settled: training stopped, the last autosave (in the list) and the watched policy are
    # the same iteration
    page.wait_for_function(
        """() => {
        const state = document.getElementById('state').textContent;
        const saved = document.getElementById('ckpt-list').textContent.match(/autosaveiteration (\\d+)/);
        const shown = state.match(/policy\\s+\\w+, iteration (\\d+)/);
        return state.includes('stopped') && saved && shown && saved[1] === shown[1];
        }""",
        timeout=30_000,
    )
    trained = iteration(page)
    assert trained > 0, state(page)
    page.click("#tabs button:text('Policy')")  # (the checkpoints' tab; remembered over the reload)
    page.fill("#ckpt-name", "my run")
    page.click("#ckpt-save")
    page.wait_for_function("document.getElementById('ckpt-list').textContent.includes('my run')", timeout=10_000)
    load(page, url, errors)  # reload: the autosave comes back
    try:
        page.wait_for_function(
            f"document.getElementById('state').textContent.includes('iteration {trained}')", timeout=60_000
        )
    except Exception:
        raise SystemExit(f"not restored (iteration {trained} expected): {state(page)}\n"
                         f"checkpoints: {page.text_content('#ckpt-list')}\nstatus: {page.inner_text('#status')}")
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
    # New run: a fresh network, trained right away (the iterations start over)
    page.click("button:text('New run')")
    page.wait_for_function(
        f"(() => {{ const m = document.getElementById('state').textContent.match(/running, iteration (\\d+)/);"
        f" return m && Number(m[1]) < {trained} / 2; }})()",
        timeout=30_000,
    )
    print("new run:", state(page))
    page.click("button:text('Stop')")
    page.wait_for_timeout(1000)
    page.locator(".ckpt", has_text="my run").locator("button:text('Restore')").click()
    page.wait_for_function(
        f"document.getElementById('state').textContent.includes('iteration {trained}')", timeout=20_000
    )
    print("restored by name:", state(page))
    page.click("#tabs button:text('World')")


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--packages", action="store_true", help="wait for pytorch, then train PPO (torch)")
    parser.add_argument("--query", default="", help="URL query, e.g. env=LunarLander")
    parser.add_argument("--stress", type=float, default=0, help="seconds per env at max population / speed")
    parser.add_argument("--crash", action="store_true", help="crash python (wasm trap), expect a restart")
    parser.add_argument("--converge", type=float, help="time training until the mean return reaches this")
    parser.add_argument("--checkpoints", action="store_true", help="autosave / save / restore / reload")
    args = parser.parse_args()
    base = serve(DIST)
    # (the page starts training by itself: the checks below start watching, train=0;
    # the autostart is checked at the end)
    url = base + "?" + "&".join(q for q in ("train=0", args.query) if q)
    # (the architecture is printed with the loading times: wasm64 by default in chromium)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1200, "height": 700})
        errors = []
        # (python warnings go to stderr = console errors: not failures)
        page.on(
            "console",
            lambda m: m.type == "error" and not any(w in m.text for w in ("Warning", "import filelock"))
            and errors.append(m.text),
        )
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
        # the tours (not by themselves in automated tests: the links), through to the end:
        # the quick one, then from its last step the detailed one
        page.click("#tour-link")
        for _ in range(5):
            assert page.is_visible("#tour-card") and page.is_visible("#tour-frame")
            page.click("#tour-card .next")
        page.click("#tour-card .more")  # (the last step: "Detailed tour")
        steps = 0
        while page.is_visible("#tour-card"):
            assert "Detailed tour" in page.inner_text("#tour-card .step")
            page.click("#tour-card .next")
            steps += 1
        assert steps >= 18 and not page.is_visible("#tour-frame"), steps
        page.click("#tabs button:text('World')")  # (the tour opened other tabs)
        # the code panel (right, closed): edit the env's source (a comment), apply: the env
        # is rebuilt with it in the worker(s), no error
        assert not page.is_visible("#code-panel")
        page.click("#code-toggle")
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
        # kept in this browser (IndexedDB): after a reload the workers start with the edit
        load(page, url, errors)
        page.click("#code-toggle")
        page.wait_for_function("document.getElementById('code').textContent.includes('# edited in the browser')",
                               timeout=10_000)
        assert "edited (kept in this browser)" in page.inner_text("#code-path")
        # Revert: the original comes back (the page reloads)
        page.once("dialog", lambda dialog: dialog.accept())
        with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
            page.click("#code-revert")
        page.click("#code-toggle")
        page.wait_for_function("document.getElementById('code').textContent.includes('register_env')", timeout=10_000)
        assert "# edited in the browser" not in page.inner_text("#code")
        page.click("#code-toggle")
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
            # pytorch loads in the background by itself (pinned, cached after the first
            # visit); its check ends up in the packages panel; then PPO (torch) trains
            page.wait_for_function("document.getElementById('package-pytorch')", timeout=900_000)
            print(page.text_content("#package-pytorch"))  # (collapsed section: text_content)
            assert "failed" not in page.text_content("#package-pytorch")
            if page.is_visible("button:text('Stop')"):
                page.click("button:text('Stop')")
            page.click("#tabs button:text('Training')")
            page.select_option("#controls-training select >> nth=0", "ppo_torch")
            page.click("#tabs button:text('World')")
            page.wait_for_timeout(500)
            page.click("button:text('Train')")
            page.wait_for_function("/ppo_torch, iteration [2-9]/.test(document.getElementById('state').textContent)",
                                   timeout=300_000)
            print("PPO (torch):", page.inner_text("#state").replace("\n", " | "))
            page.click("button:text('Stop')")
        if args.screenshot:
            page.screenshot(path=str(args.screenshot))
            page.click("#tabs button:text('Policy')")  # (the network view: its own picture)
            page.wait_for_timeout(1000)
            section = page.locator("#network-section")
            if section.is_visible():
                section.screenshot(path=str(args.screenshot.with_name(args.screenshot.stem + "_network.png")))
            page.click("#tabs button:text('World')")
        # every env loads and runs in the browser (no errors)
        for env in ("LunarLander", "Walker", "Crawler", "Car"):
            page.select_option("#controls-environment select >> nth=0", env)
            page.wait_for_timeout(800)
        # the pretrained walker that comes with the page (public/pretrained/): restorable
        page.select_option("#controls-environment select >> nth=0", "Walker")
        page.click("#tabs button:text('Policy')")
        pretrained = page.locator(".ckpt", has_text="pretrained")
        pretrained.wait_for(timeout=20_000)
        pretrained.locator("button:text('Restore')").click()
        page.wait_for_function("document.getElementById('state').textContent.includes('cem, iteration 300')",
                               timeout=20_000)
        print("pretrained walker:", state(page))
        page.click("#tabs button:text('World')")
        page.select_option("#controls-environment select >> nth=0", "Car")
        page.wait_for_timeout(800)
        # the car (not batchable: one world per agent) trains too
        page.wait_for_function("document.getElementById('state').textContent.includes('none yet')", timeout=30_000)
        page.click("button:text('Train')")
        page.wait_for_function("document.getElementById('state').textContent.includes('cem, iteration')",
                               timeout=60_000)  # (the display watches the trained car policy)
        print("car:", state(page))
        page.click("button:text('Stop')")
        page.select_option("#controls-environment select >> nth=0", "CartPole")
        page.wait_for_timeout(800)
        # the plain page starts training by itself
        load(page, base + (f"?{args.query}" if args.query else ""), errors)
        page.wait_for_function("document.getElementById('state').textContent.includes('running')", timeout=60_000)
        print("autostart:", page.inner_text("#state").splitlines()[0])
        if errors or page.inner_text("#status").startswith("error"):
            raise SystemExit("errors:\n" + "\n".join(errors) + page.inner_text("#status"))
        browser.close()


if __name__ == "__main__":
    main()
