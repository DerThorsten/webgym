"""Pack the python side of the page into web/public: per architecture (wasm32, wasm64)
one `core-<arch>-<hash>.tar.gz` (numpy, openblas, pyopengl, pyb2d3 from emscripten-forge
+ the webgym sources at /app) and config.json naming them ({arch: {"archives": [...]}}).
No package install (mamba) in the browser for the core.

pytorch / jax are NOT packed (too big for the deployment): the worker fetches them
from the channel at runtime.

Adapted from pyb2d3/examples/web/build_page.py.

    python web/build_page.py      # then: cd web && npx vite
"""

import contextlib
import gzip
import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PUBLIC = HERE / "public"
CHANNEL = "https://repo.prefix.dev/emscripten-forge/6x/emscripten-{arch}"
ARCHS = ("wasm32", "wasm64")
PACKAGES = ["numpy", "openblas", "pyopengl", "pyb2d3"]
PACKAGE_CACHE = REPO / "build" / "conda-packages"
# must match the python of the pyjs npm package (web/package.json)
PYTHON_VERSION = "3.14"
# not needed to run anything: tests, headers, docs, metadata, ...
DROP_DIRS = {"tests", "testing", "f2py", "distutils", "include", "__pycache__", "doc"}
DROP_SUFFIXES = (".pyi", ".pxd", ".h", ".a", ".pc", ".cmake", ".pyc", ".c", ".cpp")


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


def _download(url):
    """the bytes at `url` (the channel refuses urllib's default user agent)"""
    request = urllib.request.Request(url, headers={"User-Agent": "webgym-build-page"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _version_key(info):
    """sort key: newest version, then build number, then upload time"""
    version = tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-_]", info["version"]))
    return version, info.get("build_number", 0), info.get("timestamp", 0)


def fetch_packages(names, arch):
    """the newest emscripten-<arch> builds of `names` (for PYTHON_VERSION), downloaded
    into PACKAGE_CACHE/<arch> once. Returns their paths."""
    channel = CHANNEL.format(arch=arch)
    packages = json.loads(_download(f"{channel}/repodata.json"))["packages"]
    python_build = f"py{PYTHON_VERSION.replace('.', '')}"
    cache = PACKAGE_CACHE / arch
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        builds = {f: i for f, i in packages.items() if i["name"] == name}
        if any(python_build in i["build"] for i in builds.values()):
            builds = {f: i for f, i in builds.items() if python_build in i["build"]}
        if not builds:
            raise SystemExit(f"no {name} for python {PYTHON_VERSION} in {channel}")
        filename = max(builds, key=lambda f: _version_key(builds[f]))
        path = cache / filename
        if not path.exists():
            print(f"downloading {arch}/{filename}")
            path.write_bytes(_download(f"{channel}/{filename}"))
        paths.append(path)
    return paths


def _keep(name):
    """files of a conda package needed at runtime (not tests, headers, metadata, ...)"""
    parts = name.split("/")
    return not (
        parts[0] in ("info", "etc", "share", "bin")
        or any(p in DROP_DIRS or p.endswith(".dist-info") for p in parts)
        or name.endswith(DROP_SUFFIXES)
    )


def pack_core(path, conda_pkgs):
    """the core archive (unpacked at /): runtime files of the conda packages + the
    webgym sources at app/webgym"""
    with tar_gz(path) as dst:
        for conda_pkg in conda_pkgs:
            with tarfile.open(conda_pkg, "r:bz2") as src:
                for member in src.getmembers():
                    if member.isfile() and _keep(member.name):
                        dst.addfile(_normalized(member), src.extractfile(member))
        dst.add(REPO / "webgym", arcname="app/webgym", filter=_normalized)


def build():
    """write public/core-<arch>-<hash>.tar.gz and public/config.json. A page loading
    meanwhile (dev server) always finds the archives its config.json names: new
    archives first, then config.json (atomic rename), old archives removed last"""
    PUBLIC.mkdir(exist_ok=True)
    config, urls = {}, set()
    for arch in ARCHS:
        tmp = PUBLIC / f"core-{arch}.tar.gz.tmp"
        pack_core(tmp, fetch_packages(PACKAGES, arch))
        data = tmp.read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:16]
        url = f"core-{arch}-{digest}.tar.gz"
        tmp.replace(PUBLIC / url)
        urls.add(url)
        config[arch] = {"archives": [{"name": "core", "url": url, "path": "/", "hash": digest, "size": len(data)}]}
        print(f"{url}: {len(data) / 1e6:.1f} MB")
    (PUBLIC / "config.json.tmp").write_text(json.dumps(config, indent=2))
    (PUBLIC / "config.json.tmp").replace(PUBLIC / "config.json")
    for old in PUBLIC.glob("core-*.tar.gz"):
        if old.name not in urls:
            old.unlink()


if __name__ == "__main__":
    build()
