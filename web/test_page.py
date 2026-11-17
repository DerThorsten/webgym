"""Load the built page (web/dist) in headless chromium: cold, then a cached reload.
Fails on console errors / page errors or if python does not start in time.
(Adapted from pyb2d3/examples/web/test_page.py.)

    python web/test_page.py [--screenshot out.png] [--install pytorch]
"""

import argparse
import functools
import http.server
import threading
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


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--install", help="also install this package (button) and wait for it")
    parser.add_argument("--query", default="", help="URL query, e.g. env=LunarLander")
    parser.add_argument("--stress", type=float, default=0, help="seconds per env at max population / speed")
    parser.add_argument("--crash", action="store_true", help="crash python (wasm trap), expect a restart")
    args = parser.parse_args()
    url = serve(DIST) + (f"?{args.query}" if args.query else "")
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1200, "height": 700})
        errors = []
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        print("cold:", load(page, url, errors))
        print("reload:", load(page, url, errors))
        page.wait_for_timeout(1000)
        print("watch:", page.inner_text("#state"))
        page.click("button:text('Train')")
        page.wait_for_function(
            "document.getElementById('state').textContent.includes('mean/best')", timeout=60_000
        )
        page.wait_for_timeout(3000)
        print("train:", page.inner_text("#state"))
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
                page.select_option("#controls select >> nth=0", env)
                page.wait_for_timeout(300)
                for name in ("Population", "Training speed"):
                    page.evaluate(
                        """(name) => {
                        const label = [...document.querySelectorAll('#controls .control')]
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
        if args.install:
            button = "#load-torch" if args.install == "pytorch" else "#load-jax"
            with page.expect_console_message(lambda m: "installed" in m.text, timeout=600_000) as msg:
                page.click(button)
            print(msg.value.text, page.inner_text("#status"))
        if args.screenshot:
            page.screenshot(path=str(args.screenshot))
        if errors or page.inner_text("#status").startswith("error"):
            raise SystemExit("errors:\n" + "\n".join(errors) + page.inner_text("#status"))
        browser.close()


if __name__ == "__main__":
    main()
