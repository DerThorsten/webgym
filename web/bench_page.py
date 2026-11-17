"""Run the worker benchmark (web/dist/bench.html) in headless chromium, print the result.

    python web/bench_page.py [--workers 1,2,3,4,6]
"""

import argparse
import json

from playwright.sync_api import sync_playwright

from test_page import DIST, serve


def main():
    """command line"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", default="1,2,3,4,6")
    args = parser.parse_args()
    url = serve(DIST) + f"bench.html?workers={args.workers}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("console", lambda m: m.type == "error" and print("console error:", m.text))
        with page.expect_console_message(lambda m: "bench result" in m.text, timeout=600_000) as msg:
            page.goto(url)
        result = json.loads(msg.value.text.split("bench result", 1)[1])
        print(page.inner_text("#out"))
        print(json.dumps(result))
        browser.close()


if __name__ == "__main__":
    main()
