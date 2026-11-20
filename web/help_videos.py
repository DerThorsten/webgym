"""Short clips of the page for the help page and the README: pretrained runs, a fun
failure (an untrained ragdoll learning with NEAT), the network view, the whole page. Each scene: the page opens at a
URL, the browser records it (playwright, real time), ffmpeg cuts out the world's band
where the agent is -> public/help/videos/<name>.mp4 (help page), <name>.jpg (its poster:
the first frame, shown until the video plays) and <name>.gif (README).

Not in git: the files are assets of the GitHub release "videos" (as the pretrained runs,
see web/pretrain.py); the Pages workflow downloads the mp4 + jpg, the README shows the
gifs from the release.

    pixi run -e web-test web-build
    pixi run -e web-test python web/help_videos.py [name ...]   # record (all: no names)
    pixi run python web/help_videos.py --upload                 # -> the release
    pixi run python web/help_videos.py --download               # <- the release (mp4 + jpg)
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "public" / "help" / "videos"
RELEASE = "videos"  # (one release for all versions: the README links stay valid)
README_GIFS = ("ragdoll_pretrained", "ragdoll_neat_fresh")  # (the only gifs uploaded: the README's)
VIEW = {"width": 1280, "height": 800}
SIZE = (960, 540)  # every clip: 16:9, the same size (side by side on the help page)
PANEL = "0x1b1b1b"  # (the side panel's color: around a target narrower than 16:9)

# each scene: the URL query; seconds before the clip ("lead") and of the clip; what is
# recorded ("target": the world "#stage", the network "#network", "page": all of it, then
# in a 16:9 window) with the tab open ("tab"); where the 16:9 cut sits in the target
# ("center": its middle, as a fraction of the target's height) or all of the target, the
# rest filled with the panel's color ("fit": the network); more CSS for the recording
# ("style"). No score card, each clip starts a new episode; its poster: the
# frame at "poster" seconds (the first may be empty)
DEFAULTS = {"lead": 1.0, "target": "#stage", "tab": "world", "center": 0.5, "fit": False, "style": "",
            "poster": 1.0}
SCENES = {
    "ragdoll_pretrained": {"query": "env=Ragdoll&checkpoint=ppo", "seconds": 10, "center": 0.6},
    # (training from scratch: every episode the newest, still random-ish policy: it
    # stumbles, flails and falls in ever new ways)
    "ragdoll_neat_fresh": {"query": "env=Ragdoll&algo=neat", "lead": 2.0, "seconds": 15, "center": 0.6},
    "walker_pretrained": {"query": "env=Walker&checkpoint=ppo", "seconds": 8, "center": 0.6},
    "crawler_pretrained": {"query": "env=Crawler&checkpoint=ppo", "seconds": 8, "center": 0.6},
    "lander_pretrained": {"query": "env=LunarLander&checkpoint=reinforce", "seconds": 3,  # (down in ~2 s)
                          "poster": 2.0},  # (at 1 s still above the view)
    "drone_pretrained": {"query": "env=Drone&checkpoint=ppo", "seconds": 10},
    "watercannon_pretrained": {"query": "env=WaterCannon&balls=2&checkpoint=ppo", "seconds": 9},
    # the network view: NEAT growing its network from nothing (new neurons and
    # connections glow green), the lander's 8 inputs keep it readable
    "neat_growing": {"query": "env=LunarLander&algo=neat", "lead": 3.0, "seconds": 20, "target": "#network",
                     "tab": "policy", "fit": True,
                     "style": "#side { width: 700px !important; }"},  # (a wide panel: the network drawn big)
    # the walking walker and its network, live (the whole page: the Policy tab and the world)
    "walker_network": {"query": "env=Walker&checkpoint=ppo", "seconds": 10, "target": "page", "tab": "policy"},
    # the whole page with a pretrained ragdoll walking
    "ragdoll_ui": {"query": "env=Ragdoll&checkpoint=ppo", "seconds": 10, "target": "page"},
}


def record(browser, base, scene, tmp):
    """one scene -> (the list of its frames for ffmpeg, the target's box in px). The frames:
    chrome's screencast (the page as it is painted, high quality jpegs with timestamps;
    playwright's own recording is too compressed)"""
    import base64

    from test_page import READY

    view = {"width": 1280, "height": 720} if scene["target"] == "page" else VIEW  # (the page: 16:9)
    page = browser.new_page(viewport=view)
    with page.expect_console_message(lambda m: READY in m.text, timeout=120_000):
        page.goto(f"{base}?tour=0&{scene['query']}")
    page.add_style_tag(content="#score { display: none !important; }" + scene["style"])  # (no score card)
    page.wait_for_timeout(2000)  # (the checkpoint restored)
    page.click(f"#tabs button[data-tab={scene['tab']}]")
    page.wait_for_timeout(1000 * scene["lead"])
    target = scene["target"]
    box = ({"x": 0, "y": 0, **view} if target == "page" else page.locator(target).bounding_box())
    cdp = page.context.new_cdp_session(page)
    frames = []  # (file, timestamp)

    def on_frame(event):
        if not recording:  # (a late frame, the page about to close)
            return
        path = tmp / f"{len(frames):05d}.jpg"
        path.write_bytes(base64.b64decode(event["data"]))
        frames.append((path, event["metadata"]["timestamp"]))
        cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    recording = True
    cdp.on("Page.screencastFrame", on_frame)
    page.keyboard.press("r")  # (a new episode: the clip starts with it)
    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "maxWidth": view["width"],
                                      "maxHeight": view["height"], "everyNthFrame": 1})
    page.wait_for_timeout(1000 * scene["seconds"])
    recording = False
    cdp.send("Page.stopScreencast")
    page.close()
    # (ffmpeg's concat list: each frame as long as until the next one)
    lines = []
    for (path, t), (_, t_next) in zip(frames, frames[1:] + [(None, frames[-1][1] + 1 / 30)]):
        lines += [f"file '{path}'", f"duration {max(t_next - t, 0.001):.4f}"]
    listing = tmp / "frames.txt"
    listing.write_text("\n".join([*lines, f"file '{frames[-1][0]}'"]) + "\n")
    return listing, box


def convert(listing, box, scene, name):
    """a 16:9 cut of the target -> mp4 (SIZE), its poster (jpg), gif. "fit" (or a target
    flatter than 16:9): all of it, the rest filled with the panel's color"""
    w = int(box["width"]) // 2 * 2
    h = int(w * 9 / 16) // 2 * 2
    out_w, out_h = SIZE
    center = scene["center"]
    if not scene["fit"] and h <= box["height"]:  # (the cut: 16:9 at `center`, inside the target)
        y = min(max(center * box["height"] - h / 2, 0), box["height"] - h)
        fit = f"crop={w}:{h}:{int(box['x'])}:{int(box['y'] + y)},scale={out_w}:{out_h}:flags=lanczos"
    else:  # (all of it, scaled to fit, padded)
        fit = (f"crop={w}:{int(box['height']) // 2 * 2}:{int(box['x'])}:{int(box['y'])},"
               f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
               f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color={PANEL}")
    frames = ["-f", "concat", "-safe", "0", "-i", str(listing)]
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error"]
    subprocess.run([*ffmpeg, *frames, "-vf", f"{fit},fps=30", "-c:v", "libx264",
                    "-preset", "slow", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", "-an",
                    str(OUT / f"{name}.mp4")], check=True)
    subprocess.run([*ffmpeg, "-ss", str(scene["poster"]), "-i", str(OUT / f"{name}.mp4"), "-frames:v", "1", "-q:v", "5",
                    str(OUT / f"{name}.jpg")], check=True)
    gif = f"{fit},fps=10,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse"
    subprocess.run([*ffmpeg, *frames, "-vf", gif, str(OUT / f"{name}.gif")], check=True)


def make(names):
    from playwright.sync_api import sync_playwright

    from test_page import DIST, serve

    OUT.mkdir(parents=True, exist_ok=True)
    base = serve(DIST)
    tmp = Path(tempfile.mkdtemp())
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            for name in names:
                scene = {**DEFAULTS, **SCENES[name]}
                frames = tmp / name
                frames.mkdir()
                listing, box = record(browser, base, scene, frames)
                convert(listing, box, scene, name)
                sizes = ", ".join(f"{ext} {(OUT / f'{name}.{ext}').stat().st_size // 1024} kB"
                                  for ext in ("mp4", "jpg", "gif"))
                print(f"{name}: {sizes}", flush=True)
            browser.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def upload():
    """public/help/videos/*.mp4, *.jpg and the README's gifs -> the release's assets (made if
    missing; replaced)"""
    files = sorted(str(f) for f in OUT.glob("*")
                   if f.suffix in (".mp4", ".jpg") or (f.suffix == ".gif" and f.stem in README_GIFS))
    if subprocess.run(["gh", "release", "view", RELEASE], capture_output=True, check=False).returncode != 0:
        subprocess.run(["gh", "release", "create", RELEASE, "--title", "Videos of the page",
                        "--notes", "Clips for the help page and the README, made with web/help_videos.py."],
                       check=True)
    subprocess.run(["gh", "release", "upload", RELEASE, "--clobber", *files], check=True)
    print(f"{len(files)} files -> release {RELEASE}")


def download():
    """the release's mp4 + jpg -> public/help/videos/ (the gifs: only for the README)"""
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", RELEASE, "--dir", str(OUT), "--pattern", "*.mp4",
                    "--pattern", "*.jpg", "--clobber"], check=True)
    print(f"{len(list(OUT.glob('*.mp4')))} videos in {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"scenes to record (default: all): {', '.join(SCENES)}")
    parser.add_argument("--upload", action="store_true", help="upload public/help/videos/ to the release")
    parser.add_argument("--download", action="store_true", help="download the release's mp4 + jpg")
    args = parser.parse_args()
    if args.upload or args.download:
        upload() if args.upload else download()
        return
    make(args.names or list(SCENES))


if __name__ == "__main__":
    main()
