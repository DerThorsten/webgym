"""The screenshots of the help page (help.html), from the built page (web/dist):

    pixi run -e web-test python web/help_shots.py

writes web/public/help/*.png (then rebuild the page).
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

from test_page import DIST, READY, serve

OUT = Path(__file__).resolve().parent / "public" / "help"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    url = serve(DIST) + "?train=0&env=LunarLander"
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
            page.goto(url)
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "overview.png")
        # training: the population (ghosts), the leader solid
        page.click("button:text('Train')")
        page.wait_for_timeout(12_000)
        page.screenshot(path=OUT / "training.png")
        page.locator("#plot-strip").screenshot(path=OUT / "progress.png")
        page.click("button:text('Stop')")
        # watching the trained policy (solid) and the best candidates (ghosts, 8 of them):
        # all start alike, after a while they part
        page.locator(".control", has_text="Show policies").locator("input").evaluate(
            "(input) => { input.value = 8; input.dispatchEvent(new Event('change')); }")
        page.wait_for_timeout(6500)
        page.locator("#stage").screenshot(path=OUT / "watch.png")
        page.click("#tabs button:text('Policy')")  # (the network and the checkpoints)
        page.wait_for_timeout(1000)
        page.locator("#tab-policy").screenshot(path=OUT / "policy.png")
        page.click("#tabs button:text('World')")
        page.click("#code-toggle")
        page.wait_for_timeout(1000)
        page.screenshot(path=OUT / "code.png")
        page.close()
        walker(browser.new_page(viewport={"width": 1280, "height": 800}), url.replace("LunarLander", "Walker"))
        browser.close()
    print("written:", ", ".join(sorted(f.name for f in OUT.glob("*.png"))))


def walker(page, url):
    """"Having fun with agents": a trained walker (the pretrained run, if there is one)
    walks (one policy, no auto reset), the mouse flips it over, it reacts"""
    with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
        page.goto(url)
    page.wait_for_timeout(1500)
    page.click("#tabs button:text('Policy')")
    pretrained = page.locator(".ckpt", has_text="pretrained")
    if pretrained.count():  # (public/pretrained/, see build_page.py)
        pretrained.locator("button:text('Restore')").click()
        page.wait_for_timeout(2000)
    else:
        page.click("button:text('Train')")
        page.wait_for_timeout(40_000)  # (CEM, with the simulation workers: it walks by then)
        page.click("button:text('Stop')")
    page.click("#tabs button:text('World')")
    page.locator(".control", has_text="Auto reset").locator("input").uncheck()
    page.wait_for_timeout(3000)  # (walking)
    page.locator("#stage").screenshot(path=OUT / "walker_walks.png")
    # the hull: in the middle (the camera follows it), at y = 1.35 m in a view of
    # center y 2 m, half height 3.5 m
    box = page.locator("#canvas").bounding_box()
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2 + (2.0 - 1.35) / 3.5 * box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x - 60, y - 160, steps=8)  # (up and back: it tips over)
    page.wait_for_timeout(300)
    page.locator("#stage").screenshot(path=OUT / "walker_grabbed.png")
    page.mouse.move(x - 120, y + 40, steps=8)
    page.mouse.up()
    page.wait_for_timeout(2500)  # (how it reacts)
    page.locator("#stage").screenshot(path=OUT / "walker_reacts.png")
    page.close()


if __name__ == "__main__":
    main()
