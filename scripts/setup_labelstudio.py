#!/usr/bin/env python3
"""Create the "Pass 1" and "Pass 2" projects in a running Label Studio and upload the 50 images to each.

Saves you clicking through project creation twice. Needs an API token: in Label Studio open
Account & Settings -> Personal Access Token (or Legacy Token), or start Label Studio with --user-token.

Usage:  python scripts/setup_labelstudio.py --token <TOKEN> [--url http://localhost:8080]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def auth_header(url: str, token: str) -> str:
    if token.count(".") == 2:  # Personal Access Token (JWT refresh token) -> short-lived access token
        body = json.dumps({"refresh": token}).encode()
        req = urllib.request.Request(f"{url}/api/token/refresh", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return "Bearer " + json.load(r)["access"]
    return "Token " + token  # legacy token


def call(url: str, auth: str, method: str, path: str, body: bytes = None, ctype: str = "application/json"):
    req = urllib.request.Request(f"{url}{path}", body, {"Authorization": auth, "Content-Type": ctype}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} failed: {e.code} {e.read().decode(errors='replace')[:500]}")


def multipart(files: list) -> tuple:
    boundary = uuid.uuid4().hex
    parts = []
    for i, f in enumerate(files):  # each part needs its own field name or Label Studio keeps only one
        parts += [f"--{boundary}\r\n".encode(),
                  f'Content-Disposition: form-data; name="file{i}"; filename="{f.name}"\r\n'.encode(),
                  b"Content-Type: image/png\r\n\r\n", f.read_bytes(), b"\r\n"]
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--projects", nargs="+", default=["Pass 1", "Pass 2"])
    args = ap.parse_args()
    url = args.url.rstrip("/")

    names = [n.strip() for n in (ROOT / "data" / "image_list.txt").read_text().splitlines() if n.strip()]
    files = [ROOT / "data" / "images" / n for n in names]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing from data/images/ - run scripts/fetch_images.py first")

    auth = auth_header(url, args.token)
    config = (ROOT / "labelstudio" / "label_config.xml").read_text()
    existing = call(url, auth, "GET", "/api/projects?page_size=1000")
    existing = {p["title"]: p for p in (existing.get("results", existing) if isinstance(existing, dict) else existing)}
    for title in args.projects:
        if title in existing:
            print(f"'{title}' already exists (id {existing[title]['id']}) - skipped")
            continue
        proj = call(url, auth, "POST", "/api/projects", json.dumps({"title": title, "label_config": config}).encode())
        body, ctype = multipart(files)
        res = call(url, auth, "POST", f"/api/projects/{proj['id']}/import", body, ctype)
        print(f"Created '{title}' (id {proj['id']}): imported {res.get('task_count', '?')} images "
              f"-> {url}/projects/{proj['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
