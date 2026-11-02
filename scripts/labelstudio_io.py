#!/usr/bin/env python3
"""Push annotations into a Label Studio project, or export a project's annotations.

  import  python scripts/labelstudio_io.py import --project "Pass 1" --labels labels/pass1.json --token <TOKEN>
          Adds each image's result from a Label Studio-style JSON file as an annotation on the matching task
          (matched by file name). Tasks that already have an annotation are skipped.
  export  python scripts/labelstudio_io.py export --project "Pass 1" --out labels/pass1.json --token <TOKEN>
          Same as Export -> JSON in the UI.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from check_labels import image_key  # noqa: E402
from setup_labelstudio import auth_header, call  # noqa: E402


def find_project(url: str, auth: str, title: str) -> dict:
    res = call(url, auth, "GET", "/api/projects?page_size=1000")
    for p in res.get("results", res) if isinstance(res, dict) else res:
        if p["title"] == title:
            return p
    sys.exit(f"no project called '{title}'")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["import", "export"])
    ap.add_argument("--project", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--labels", type=Path, help="import: file to read")
    ap.add_argument("--out", type=Path, help="export: file to write")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    auth = auth_header(url, args.token)
    proj = find_project(url, auth, args.project)

    if args.cmd == "export":
        req = urllib.request.Request(f"{url}/api/projects/{proj['id']}/export?exportType=JSON",
                                     headers={"Authorization": auth})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=1) + "\n")
        print(f"Exported {len(data)} tasks from '{args.project}' to {args.out}")
        return 0

    source = {image_key(t["data"]["image"]): t["annotations"][0]["result"]
              for t in json.loads(args.labels.read_text()) if t.get("annotations")}
    tasks = call(url, auth, "GET", f"/api/tasks?project={proj['id']}&page_size=1000")
    tasks = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
    added = skipped = missing = 0
    for t in tasks:
        key = image_key(t["data"]["image"])
        if t.get("total_annotations", 0):
            skipped += 1
            continue
        if key not in source:
            missing += 1
            continue
        call(url, auth, "POST", f"/api/tasks/{t['id']}/annotations/", json.dumps({"result": source[key]}).encode())
        added += 1
    print(f"'{args.project}': added {added} annotations, skipped {skipped} already annotated, "
          f"{missing} images not in {args.labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
