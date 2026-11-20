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
        page.wait_for_function("document.querySelectorAll('#chips .chip').length > 1", timeout=30_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "overview.png")
        # training: the current policy in view, the progress plot growing
        page.click("#quick-train[title='Train']")
        page.wait_for_timeout(12_000)
        page.screenshot(path=OUT / "training.png")
        page.locator("#plot-strip").screenshot(path=OUT / "progress.png")
        page.click("#quick-train[title='Stop']")
        # watching the trained policy
        page.wait_for_timeout(6500)
        page.locator("#stage").screenshot(path=OUT / "watch.png")
        page.click("#tabs button:text('Policy')")  # (the network and the checkpoints)
        page.wait_for_timeout(1000)
        page.locator("#tab-policy").screenshot(path=OUT / "policy.png")
        page.click("#tabs button:text('World')")
        page.click("#code-toggle")
        # (Monaco loads; the world's canvas got narrower: the worker draws at the new size
        # a moment later, before that the old picture is squeezed into it)
        page.wait_for_function("document.querySelectorAll('#code .view-line').length > 5", timeout=30_000)
        page.wait_for_timeout(3000)
        page.screenshot(path=OUT / "code.png")
        page.close()
        base = url.split("?")[0]
        walker(browser.new_page(viewport={"width": 1280, "height": 800}), base + "?train=0&env=Walker&algo=ppo_torch")
        toys(browser, base)
        worlds(browser, base)
        browser.close()
    print("written:", ", ".join(sorted(f.name for f in OUT.glob("*.png"))))


def pretrained(page, url):
    """open the page at `url` and restore the pretrained run of its env + algo"""
    with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
        page.goto(url)
    page.wait_for_timeout(1500)
    page.click("#tabs button:text('Policy')")
    page.locator(".ckpt", has_text="pretrained").first.locator("button:text('Restore')").click()
    page.wait_for_timeout(2000)


def to_screen(page, dx, y, center_y, zoom):
    """a world point (dx m right of the followed agent, height y) in page pixels (the
    camera follows the agent: it is in the middle; the view is `zoom` m high above and
    below center_y)"""
    box = page.locator("#canvas").bounding_box()
    scale = box["height"] / 2 / zoom
    return box["x"] + box["width"] / 2 + dx * scale, box["y"] + box["height"] / 2 - (y - center_y) * scale


# where the walking (pretrained PPO) walker's hull is in the view: a bit right of the
# middle (the camera trails it), low (it walks crouched); m
HULL = (1.2, 0.95)


def walker(page, url):
    """"Having fun with agents": the trained walker (pretrained) walks (one policy, no
    auto reset), the mouse flips it over, it reacts"""
    pretrained(page, url)
    page.click("#tabs button:text('Toys')")  # (auto reset off, one policy, no score card)
    page.wait_for_timeout(3000)  # (walking)
    page.locator("#stage").screenshot(path=OUT / "walker_walks.png")
    x, y = to_screen(page, HULL[0], HULL[1], 2.0, 3.5)  # (grab the hull)
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


def open_fold(page, fold):
    """the Toys tab: open the fold (a <details>) if it is closed"""
    if page.locator("details.group", has_text=fold).get_attribute("open") is None:
        page.locator("summary", has_text=fold).click()


def tool(page, fold, name):
    """the Toys tab: choose the tool"""
    open_fold(page, fold)
    page.click(f".tools button:text-is('{name}')")
    page.wait_for_timeout(300)


def reset(page):
    """a new episode (the button: R would go to a focused widget)"""
    page.click("#quick-episode")


def toys(browser, base):
    """the Toys tab: the walker meets a ramp and an explosion, the ragdoll walks in the rain"""
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    pretrained(page, base + "?train=0&env=Walker&algo=ppo_torch")
    page.click("#tabs button:text('Toys')")  # (auto reset off, one policy)
    reset(page)
    page.wait_for_timeout(1500)
    tool(page, "Ramps and steps", "Ramp")
    page.mouse.click(*to_screen(page, HULL[0] + 2.5, 1.0, 2.0, 3.5))  # (ahead of it: the ramp lands on the ground)
    page.wait_for_timeout(300)  # (it walks up; a moment later it may well fall back)
    page.locator("#stage").screenshot(path=OUT / "toys_ramp.png")
    reset(page)
    page.wait_for_timeout(2000)
    tool(page, "Explosion", "💥 Explosion")
    page.mouse.click(*to_screen(page, HULL[0] - 0.7, 0.4, 2.0, 3.5))  # (just behind it, low)
    page.wait_for_timeout(250)
    page.locator("#stage").screenshot(path=OUT / "toys_explosion.png")
    page.close()
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    pretrained(page, base + "?train=0&env=Ragdoll&algo=ppo_torch")
    page.click("#tabs button:text('Toys')")
    open_fold(page, "Weather")
    page.get_by_role("checkbox", name="Rain").check()
    reset(page)
    page.wait_for_timeout(3000)
    page.locator("#stage").screenshot(path=OUT / "toys_rain.png")
    page.close()


def worlds(browser, base):
    """the Blob (pretrained ES) and the Drone (pretrained PPO) on their way"""
    for env, algo, seconds in (("Blob", "es", 4000), ("Drone", "ppo_torch", 5000)):
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        pretrained(page, f"{base}?train=0&env={env}&algo={algo}")
        page.click("#tabs button:text('World')")
        reset(page)
        page.wait_for_timeout(seconds)
        page.locator("#stage").screenshot(path=OUT / f"{env.lower()}.png")
        page.close()


if __name__ == "__main__":
    main()
