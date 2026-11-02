#!/usr/bin/env python3
"""Download the Penn-Fudan Pedestrian dataset and copy a fixed set of 50 images into data/images/.

The selection is deterministic (evenly spaced through the sorted file list), so anyone who runs this
gets the same 50 images. The list is also written to data/image_list.txt, which is committed.

Images are NOT committed to git: the dataset's readme asks that the material not be reposted
without the authors' permission. Run this script to get them locally.

Usage:  python scripts/fetch_images.py [--count 50]
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "images"
LIST_FILE = ROOT / "data" / "image_list.txt"
ZIP_FILE = ROOT / "data" / "PennFudanPed.zip"  # kept for scripts/pennfudan_reference.py
PREFIX = "PennFudanPed/PNGImages/"


def pick_evenly(names: list[str], count: int) -> list[str]:
    if count >= len(names):
        return names
    step = len(names) / count
    return [names[int(i * step)] for i in range(count)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--zip", type=Path, help="use an already-downloaded PennFudanPed.zip instead of downloading")
    args = ap.parse_args()

    if args.zip:
        data = args.zip.read_bytes()
    elif ZIP_FILE.exists():
        data = ZIP_FILE.read_bytes()
    else:
        print(f"Downloading {URL} (~54 MB) ...")
        with urllib.request.urlopen(URL, timeout=120) as r:
            data = r.read()
        ZIP_FILE.parent.mkdir(parents=True, exist_ok=True)
        ZIP_FILE.write_bytes(data)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Only take plain file names from the expected folder; ignore anything else in the archive.
        names = sorted(
            n[len(PREFIX):]
            for n in zf.namelist()
            if n.startswith(PREFIX) and n.endswith(".png") and "/" not in n[len(PREFIX):]
        )
        if not names:
            print("No images found in archive - has the dataset layout changed?", file=sys.stderr)
            return 1
        chosen = pick_evenly(names, args.count)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in chosen:
            (OUT_DIR / name).write_bytes(zf.read(PREFIX + name))

    LIST_FILE.write_text("\n".join(chosen) + "\n")
    print(f"Wrote {len(chosen)} images to {OUT_DIR.relative_to(ROOT)}/ and the list to {LIST_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
