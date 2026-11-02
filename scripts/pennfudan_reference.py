#!/usr/bin/env python3
"""Turn the Penn-Fudan dataset's own pedestrian boxes into a Label Studio-style JSON export.

Useful for two things:
  * a smoke test of tools/check_labels.py on real data before you have labelled anything;
  * comparing one of your passes against the official boxes:
        python tools/check_labels.py agree labels/pass1.json labels/reference.json
    (the reference only has `person`, so two_wheeler/vehicle boxes will show up as unmatched).

Note the official boxes follow the dataset's own rules, not docs/labelling_guideline.md - e.g. the original
release skipped small or heavily occluded people - so expect some disagreement.

Usage:  python scripts/pennfudan_reference.py [--zip data/PennFudanPed.zip] [--out labels/reference.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZE_RE = re.compile(r"Image size \(X x Y x C\) : (\d+) x (\d+)")
BOX_RE = re.compile(r"Bounding box for object (\d+) .*: \((\d+), (\d+)\) - \((\d+), (\d+)\)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=ROOT / "data" / "PennFudanPed.zip")
    ap.add_argument("--list", type=Path, default=ROOT / "data" / "image_list.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "labels" / "reference.json")
    args = ap.parse_args()

    names = [n.strip() for n in args.list.read_text().splitlines() if n.strip()]
    tasks = []
    with zipfile.ZipFile(args.zip) as zf:
        for tid, name in enumerate(names, 1):
            txt = zf.read(f"PennFudanPed/Annotation/{name[:-4]}.txt").decode("latin-1")
            W, H = map(int, SIZE_RE.search(txt).groups())
            result = []
            for obj, x0, y0, x1, y1 in (map(int, m) for m in BOX_RE.findall(txt)):
                # 1-based inclusive pixel coordinates -> 0-based edges
                x, y, w, h = x0 - 1, y0 - 1, x1 - x0 + 1, y1 - y0 + 1
                result.append({
                    "id": f"{tid}-{obj}", "type": "rectanglelabels", "from_name": "box", "to_name": "image",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "value": {"x": 100 * x / W, "y": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H,
                              "rotation": 0, "rectanglelabels": ["person"]},
                })
            tasks.append({"id": tid, "data": {"image": f"images/{name}"},
                          "annotations": [{"id": tid, "result": result, "was_cancelled": False}]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, indent=1))
    print(f"Wrote {len(tasks)} tasks, {sum(len(t['annotations'][0]['result']) for t in tasks)} boxes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
