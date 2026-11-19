"""Prepare the python side of the page in web/public:

* the packages, pinned per architecture (wasm32, wasm64) with pyjs' lock CLI (solved
  here, at build time: no solver in the browser): "core" (numpy, pyopengl, pyb2d3:
  installed when python starts) and "pytorch" (loaded in the background; only the
  packages that are not in the core). pyjs caches them in the browser after the first
  visit (installed layout + bytecode).
* `webgym-<hash>.tar.gz`: the webgym sources, unpacked at /app.
* config.json: {arch: {"core": [{url, sha256}], "pytorch": [...], "archives": [...],
  "release": pyproject.toml's version}}.

    python web/build_page.py              # then: cd web && npx vite
    python web/build_page.py --sources    # only the sources (the dev server, on a change)
"""

import contextlib
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PUBLIC = HERE / "public"
LOCK_CLI = HERE / "node_modules" / "@derthorsten" / "pyjs" / "bin" / "pyjs.mjs"
CORE = ["numpy", "pyopengl", "pyb2d3>=1.2.0"]
GROUPS = {"pytorch": ["pytorch"]}  # loaded later, in the background


@contextlib.contextmanager
def tar_gz(path):
    """a .tar.gz writer whose bytes depend on the content only (stable hash)"""
    with gzip.GzipFile(path, "wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        yield tar


def _normalized(tarinfo):
    """tar filter: no time stamps / owners, no bytecode caches"""
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc"):
        return None
    tarinfo.mtime, tarinfo.uid, tarinfo.gid, tarinfo.uname, tarinfo.gname = 0, 0, 0, "", ""
    return tarinfo


def lock(specs):
    """{arch: [{url, sha256}]} for `specs`, solved with pyjs' lock CLI"""
    out = REPO / "build" / "lock.json"
    out.parent.mkdir(exist_ok=True)
    subprocess.run(["node", str(LOCK_CLI), "lock", *specs, "-o", str(out)], check=True)
    return json.loads(out.read_text())


def pack_sources():
    """public/webgym-<hash>.tar.gz (the webgym sources) -> its name"""
    PUBLIC.mkdir(exist_ok=True)
    tmp = PUBLIC / "webgym.tar.gz.tmp"
    with tar_gz(tmp) as tar:
        tar.add(REPO / "webgym", arcname="webgym", filter=_normalized)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:16]
    archive = f"webgym-{digest}.tar.gz"
    tmp.replace(PUBLIC / archive)
    return archive


def pretrained_index():
    """public/pretrained/index.json: the pretrained runs there (checkpoints written by
    `python -m webgym.cli ENV --save public/pretrained/NAME.safetensors`), from their own
    metadata: [{file, env (its key, with the settings), algo, iteration, mean_return}]"""
    from webgym import checkpoints

    folder = PUBLIC / "pretrained"
    if not folder.exists():
        return
    index = []
    for path in sorted(folder.glob("*.safetensors")):
        _, meta = checkpoints.loads(path.read_bytes())
        index.append({"file": path.name, "env": meta.get("env_key", meta["env"]), "algo": meta["algo"],
                      "iteration": meta["iteration"], "mean_return": meta["mean_return"]})
    (folder / "index.json").write_text(json.dumps(index, indent=2))
    print(f"pretrained: {', '.join(f'{r['env']} / {r['algo']}' for r in index) or 'none'}")


def build(sources_only=False):
    """write public/webgym-<hash>.tar.gz and public/config.json (sources_only: keep the
    pinned packages of the config.json there is). A page loading meanwhile (dev server)
    always finds the archive its config.json names: new archive first, then config.json
    (atomic rename), old archives removed last"""
    archive = pack_sources()
    pretrained_index()
    if sources_only and (PUBLIC / "config.json").exists():
        config = json.loads((PUBLIC / "config.json").read_text())
        for groups in config.values():
            groups["archives"] = [{"url": archive, "path": "/app"}]
    else:
        config = locked_config(archive)
    # the release (pyproject.toml's version): a new one asks browsers to delete their old
    # saved data (web/main.js)
    release = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    for groups in config.values():
        groups["release"] = release
    (PUBLIC / "config.json.tmp").write_text(json.dumps(config, indent=2))
    (PUBLIC / "config.json.tmp").replace(PUBLIC / "config.json")
    for old in [*PUBLIC.glob("webgym-*.tar.gz"), *PUBLIC.glob("core-*.tar.gz")]:
        if old.name != archive:
            old.unlink()
    for arch, groups in config.items():
        print(arch, ", ".join(f"{g}: {len(p)} packages" for g, p in groups.items() if g not in ("archives", "release")))
    print(f"{archive}: {(PUBLIC / archive).stat().st_size / 1e3:.0f} kB")


def locked_config(archive):
    """config.json with the packages pinned per architecture (pyjs' lock CLI)"""
    core = lock(CORE)
    config = {arch: {"core": pkgs, "archives": [{"url": archive, "path": "/app"}]} for arch, pkgs in core.items()}
    for group, specs in GROUPS.items():
        # solved together with the core (the same versions of the shared packages);
        # installed later: only what the core does not have
        full = lock(CORE + specs)
        for arch, pkgs in full.items():
            core_urls = {p["url"] for p in core[arch]}
            if not core_urls <= {p["url"] for p in pkgs}:
                raise SystemExit(f"{group} ({arch}) needs other versions of the core packages")
            config[arch][group] = [p for p in pkgs if p["url"] not in core_urls]
    return config


if __name__ == "__main__":
    build(sources_only="--sources" in sys.argv)
