"""Assemble the static site that GitHub Pages publishes.

The repo has one thing a browser can usefully look at, and it is not a plain
file in the tree: `web/` is the live simulation UI. It is written against
`fetch('/api/...')` and its four asset tags are root-absolute (`/style.css`,
`/app.js`), which is right for `scripts/serve.py` and wrong for a project Pages
site: that is served from `/<repo>/`, so `/app.js` would resolve to the *user*
root and 404. The build rewrites those four tags to relative paths, which work
at any depth.

This script builds `_site/`, which is what CI uploads:

    _site/index.html      web/index.html with the replay shim wired in
    _site/*.css *.js      the rest of web/ verbatim
    _site/replay.json     the recording, from scripts/make_replay.py
    _site/.nojekyll       so nothing is filtered on the way through

The edits made to `web/index.html` are two inserted lines (a marker telling
`web/replay.js` that this build has no server, and the `replay.js` tag itself)
plus a path rewrite. `web/index.html` itself is untouched, so
`python scripts/serve.py` still serves the live version.

Usage:

    python scripts/make_replay.py      # once, or when the checkpoint changes
    python scripts/build_site.py
    python -m http.server -d _site 8000
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MARKER = (
    '<!-- injected by scripts/build_site.py: no server here, so app.js is\n'
    '     answered from replay.json by replay.js. the original file is\n'
    '     web/index.html and is unchanged. -->\n'
    '<script>window.__FLYBRAIN_REPLAY__ = "replay.json";</script>\n'
    '<script src="replay.js"></script>\n'
)

# Root-absolute asset tags would point at the user root on a project Pages
# site, so they are made relative. Only tags are touched - the `fetch('/api/')`
# strings live in app.js and are intercepted by replay.js, never resolved.
ABS_ASSET = re.compile(r'((?:href|src)=")/(?!/)')

# Copied to the site root. Anything else in web/ (or added later) rides along;
# this list only exists so a renamed entry point cannot slip through silently.
REQUIRED_WEB = ["index.html", "style.css", "app.js", "brain.js", "fly3d.js",
                "replay.js"]


def build(site: str) -> int:
    web = os.path.join(ROOT, "web")
    replay = os.path.join(web, "replay.json")

    missing = [f for f in REQUIRED_WEB if not os.path.isfile(os.path.join(web, f))]
    if missing:
        print(f"  ! missing from web/: {', '.join(missing)}")
        return 1
    if not os.path.isfile(replay):
        print("  ! web/replay.json is missing - run scripts/make_replay.py first")
        return 1

    if os.path.isdir(site):
        shutil.rmtree(site)
    os.makedirs(site)

    # --- the simulation, at the site root ---------------------------------
    for name in sorted(os.listdir(web)):
        src = os.path.join(web, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(site, name))

    index = os.path.join(site, "index.html")
    with open(index, "r", encoding="utf-8") as fh:
        html = fh.read()

    # Insert immediately before the first <script>, which is the first thing
    # app.js depends on. Anchoring on the tag rather than a line number keeps
    # this working if index.html is reformatted.
    m = re.search(r"^[ \t]*<script ", html, re.MULTILINE)
    if not m:
        print("  ! web/index.html has no <script> tag to anchor the shim to")
        return 1
    html = html[:m.start()] + MARKER + html[m.start():]

    before = len(ABS_ASSET.findall(html))
    html = ABS_ASSET.sub(r"\1", html)
    print(f"  rewrote {before} root-absolute asset path(s) to relative")

    with open(index, "w", encoding="utf-8") as fh:
        fh.write(html)

    open(os.path.join(site, ".nojekyll"), "w").close()

    # --- report, so a silent short build is impossible --------------------
    total = 0
    files = 0
    for dirpath, _dirnames, filenames in os.walk(site):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
            files += 1
    print(f"  wrote {os.path.relpath(site, ROOT)}/"
          f" ({files} files, {total / 1024 / 1024:.2f} MiB)")

    for probe, why in [
        (os.path.join(site, "index.html"), "the simulation"),
        (os.path.join(site, "replay.json"), "the recording"),
    ]:
        state = "ok" if os.path.isfile(probe) else "MISSING"
        print(f"    {state:>7}  /{os.path.relpath(probe, site).replace(os.sep, '/')}"
              f"   ({why})")

    if "window.__FLYBRAIN_REPLAY__" not in open(index, encoding="utf-8").read():
        print("  ! the marker did not make it into _site/index.html")
        return 1
    if ABS_ASSET.search(open(index, encoding="utf-8").read()):
        print("  ! a root-absolute asset path survived - it will 404 under "
              "/<repo>/ on Pages")
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the static FlyBrain site.")
    ap.add_argument("--out", default=os.path.join(ROOT, "_site"))
    args = ap.parse_args()
    sys.exit(build(args.out))


if __name__ == "__main__":
    main()
