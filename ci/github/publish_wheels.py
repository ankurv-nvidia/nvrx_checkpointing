#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# ruff: noqa: T201
"""Add a main build's wheels to the rolling GitHub release and write the pip index for them.

  publish_wheels.py WHEEL_DIR SITE_DIR

The release keeps the wheels of the newest builds as assets. SITE_DIR receives a
PEP 503 simple index for GitHub Pages whose links download those assets.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path

PROJECT = "nvrx-checkpointing"
RELEASE_TAG = "wheels"
KEEP_BUILDS = 50
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
UPLOADS = "https://uploads.github.com"

Api = Callable[..., object]


def _request(token: str, method: str, url: str, body: bytes | None = None, content_type: str = "application/json") -> object:
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": content_type,
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def index_url(repo: str) -> str:
    owner, name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{name}/simple/"


def release(api: Api, repo: str, sha: str) -> dict:
    """The rolling pre-release that holds the published wheels, created on first use."""
    try:
        return api("GET", f"{API}/repos/{repo}/releases/tags/{RELEASE_TAG}")
    except urllib.error.HTTPError as error:
        if error.code != HTTPStatus.NOT_FOUND:
            raise
    body = {
        "tag_name": RELEASE_TAG,
        "target_commitish": sha,
        "name": "Wheels from main",
        "body": (
            "CI adds the wheels of every main build that passes its gates to this release.\n\n"
            f"Install with `pip install --pre --index-url {index_url(repo)} {PROJECT}`."
        ),
        "prerelease": True,
    }
    return api("POST", f"{API}/repos/{repo}/releases", json.dumps(body).encode())


def assets(api: Api, repo: str, release_id: int) -> list[dict]:
    found, page = [], 1
    while batch := api("GET", f"{API}/repos/{repo}/releases/{release_id}/assets?per_page=100&page={page}"):
        found += batch
        page += 1
    return found


def upload(api: Api, repo: str, release_id: int, wheel: Path) -> dict:
    url = f"{UPLOADS}/repos/{repo}/releases/{release_id}/assets?name={urllib.parse.quote(wheel.name)}"
    asset = api("POST", url, wheel.read_bytes(), "application/octet-stream")
    if asset["name"] != wheel.name:
        raise RuntimeError(f"GitHub stored {wheel.name} as {asset['name']}, a name pip cannot match to the wheel")
    return asset


def _version(filename: str) -> str:
    return filename.split("-")[1]


def _build_order(version: str) -> tuple[int, str]:
    dev = re.search(r"\.dev(\d+)", version)
    return int(dev.group(1)) if dev else -1, version


def stale_assets(published: list[dict], keep: int) -> list[dict]:
    """The assets of every build older than the newest ``keep`` builds."""
    builds = sorted({_version(asset["name"]) for asset in published}, key=_build_order, reverse=True)
    kept = set(builds[:keep])
    return [asset for asset in published if _version(asset["name"]) not in kept]


def _page(title: str, links: list[tuple[str, str]]) -> str:
    anchors = "".join(f'    <a href="{html.escape(href)}">{html.escape(text)}</a><br>\n' for href, text in links)
    return (
        "<!DOCTYPE html>\n<html>\n  <head>\n"
        '    <meta name="pypi:repository-version" content="1.0">\n'
        f"    <title>{html.escape(title)}</title>\n  </head>\n  <body>\n"
        f"    <h1>{html.escape(title)}</h1>\n{anchors}  </body>\n</html>\n"
    )


def write_index(site: Path, repo: str, published: list[dict], digests: dict[str, str]) -> None:
    project = site / "simple" / PROJECT
    project.mkdir(parents=True, exist_ok=True)
    (site / "simple" / "index.html").write_text(_page("Simple index", [(f"{PROJECT}/", PROJECT)]))
    links = []
    for asset in sorted(published, key=lambda asset: (_build_order(_version(asset["name"])), asset["name"]), reverse=True):
        digest = digests.get(asset["name"]) or asset.get("digest", "").removeprefix("sha256:")
        links.append((asset["browser_download_url"] + (f"#sha256={digest}" if digest else ""), asset["name"]))
    (project / "index.html").write_text(_page(f"Links for {PROJECT}", links))
    (site / "index.html").write_text(
        f"<!DOCTYPE html>\n<html>\n  <head><title>{PROJECT} wheels</title></head>\n  <body>\n"
        f"    <p>Wheels from every main build of {html.escape(repo)} that passes CI. Install with:</p>\n"
        f"    <pre>python3 -m pip install --upgrade --pre --index-url {index_url(repo)} {PROJECT}</pre>\n"
        "    <p>The aarch64 wheel targets Grace (Neoverse V2) CPUs.</p>\n"
        f'    <p><a href="simple/{PROJECT}/">All published wheels</a></p>\n  </body>\n</html>\n',
    )


def publish(api: Api, repo: str, sha: str, wheel_dir: Path, site: Path) -> list[dict]:
    wheels = sorted(wheel_dir.rglob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no wheels under {wheel_dir}")
    target = release(api, repo, sha)
    published = assets(api, repo, target["id"])
    present = {asset["name"] for asset in published}
    digests = {}
    for wheel in wheels:
        digests[wheel.name] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if wheel.name in present:
            print(f"{wheel.name} is already published")
            continue
        published.append(upload(api, repo, target["id"], wheel))
        print(f"published {wheel.name}")
    for asset in stale_assets(published, KEEP_BUILDS):
        api("DELETE", f"{API}/repos/{repo}/releases/assets/{asset['id']}")
        published.remove(asset)
        print(f"removed {asset['name']}")
    write_index(site, repo, published, digests)
    return published


def main(argv: list[str]) -> int:
    token = os.environ["GITHUB_TOKEN"]
    published = publish(
        lambda *args: _request(token, *args),
        os.environ["GITHUB_REPOSITORY"],
        os.environ.get("GITHUB_SHA", "main"),
        Path(argv[0]),
        Path(argv[1]),
    )
    print(f"{len(published)} wheels in the index at {index_url(os.environ['GITHUB_REPOSITORY'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
