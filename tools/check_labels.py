#!/usr/bin/env python3
"""Check exported bounding-box / polygon labels and measure agreement between two labelling passes.

Reads Label Studio "JSON" exports (list of tasks) and COCO exports (from Label Studio or CVAT); the format is
detected automatically. Standard library only, Python 3.9+.

  check   python tools/check_labels.py check labels/pass1.json --classes labelstudio/classes.txt \
              --images-dir data/images
          Finds boxes/polygons outside the image, zero-size boxes, degenerate polygons, unknown or missing
          class names, near-duplicate boxes, and images with no labels (including images in --images-dir that
          never appear in the export). Exit code 1 if any error is found.

  agree   python tools/check_labels.py agree labels/pass1.json labels/pass2.json
          python tools/check_labels.py agree labels/both.json      # annotation 1 vs 2 of each task
          Matches shapes between the passes by IoU and reports F1, mean IoU, class agreement, per-class
          scores and the images with the worst agreement.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PLURAL = {"box": "boxes", "polygon": "polygons"}
EMPTY_FLAG = "No objects"  # Choice value in labelstudio/label_config.xml
# Label Studio prefixes uploaded files with 8 hex chars and a dash: "/data/upload/3/1a2b3c4d-FudanPed00001.png"
LS_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")


# --------------------------------------------------------------------------------------------- data model

@dataclass
class Shape:
    kind: str                 # "box" or "polygon"
    label: Optional[str]      # None if the shape has no class
    points: List[Point]       # box: 4 corners (handles rotation); polygon: vertices. Pixels.
    width: float = 0.0        # box only, pixels
    height: float = 0.0
    extra_labels: List[str] = field(default_factory=list)  # more than one class on one shape
    ref: str = ""             # id inside the export, for error messages

    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class ImageLabels:
    key: str                          # normalised file name, used to pair passes
    width: Optional[float]
    height: Optional[float]
    shapes: List[Shape] = field(default_factory=list)
    marked_empty: bool = False
    source: str = ""                  # e.g. "task 12, annotation 34"


def image_key(path: str) -> str:
    """'/data/upload/3/1a2b3c4d-FudanPed00001.png' -> 'FudanPed00001.png'."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1].split("?", 1)[0]
    return LS_UPLOAD_PREFIX.sub("", name)


# --------------------------------------------------------------------------------------------- loading

def load(path: Path, annotation_index: Optional[int] = None) -> List[ImageLabels]:
    """Return one ImageLabels per (image, annotation). annotation_index picks one annotation per task."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        return load_coco(data)
    if isinstance(data, list):
        return load_label_studio(data, annotation_index)
    raise ValueError(f"{path}: not a Label Studio JSON export (list of tasks) or a COCO file")


def _rect_corners(x: float, y: float, w: float, h: float, rotation_deg: float) -> List[Point]:
    """Label Studio rotates rectangles around their top-left corner."""
    if not rotation_deg:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    a = math.radians(rotation_deg)
    c, s = math.cos(a), math.sin(a)
    return [(x + dx * c - dy * s, y + dx * s + dy * c) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]


def _labels_of(value: dict, key: str) -> Tuple[Optional[str], List[str]]:
    labels = value.get(key) or value.get("labels") or []
    if not labels:
        return None, []
    return labels[0], list(labels[1:])


def load_label_studio(tasks: list, annotation_index: Optional[int]) -> List[ImageLabels]:
    out: List[ImageLabels] = []
    for task in tasks:
        data = task.get("data", {})
        img_path = data.get("image") or next((v for v in data.values() if isinstance(v, str)), "")
        key = image_key(img_path) if img_path else f"task-{task.get('id')}"
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if annotation_index is not None:
            anns = anns[annotation_index:annotation_index + 1]
        if not anns:
            # Task exported without any annotation: still an image with no labels.
            out.append(ImageLabels(key, None, None, source=f"task {task.get('id')}, no annotation"))
            continue
        for ann in anns:
            img = ImageLabels(key, None, None, source=f"task {task.get('id')}, annotation {ann.get('id')}")
            for r in ann.get("result", []):
                rtype, v = r.get("type"), r.get("value", {})
                W, H = r.get("original_width"), r.get("original_height")
                if W and H:
                    img.width, img.height = float(W), float(H)
                if rtype == "choices":
                    if EMPTY_FLAG in v.get("choices", []):
                        img.marked_empty = True
                    continue
                if rtype not in ("rectanglelabels", "rectangle", "polygonlabels", "polygon"):
                    continue
                if not (W and H):
                    raise ValueError(f"{img.source}: region {r.get('id')} has no original_width/height")
                sx, sy = W / 100.0, H / 100.0  # Label Studio stores coordinates as % of image size
                if rtype.startswith("rectangle"):
                    label, extra = _labels_of(v, "rectanglelabels")
                    w, h = v.get("width", 0) * sx, v.get("height", 0) * sy
                    pts = _rect_corners(v.get("x", 0) * sx, v.get("y", 0) * sy, w, h, v.get("rotation", 0) or 0)
                    img.shapes.append(Shape("box", label, pts, w, h, extra, str(r.get("id", ""))))
                else:
                    label, extra = _labels_of(v, "polygonlabels")
                    pts = [(px * sx, py * sy) for px, py in v.get("points", [])]
                    img.shapes.append(Shape("polygon", label, pts, extra_labels=extra, ref=str(r.get("id", ""))))
            out.append(img)
    return out


def load_coco(coco: dict) -> List[ImageLabels]:
    cats = {c["id"]: c.get("name") for c in coco.get("categories", [])}
    images: Dict[object, ImageLabels] = {}
    for im in coco["images"]:
        images[im["id"]] = ImageLabels(image_key(im.get("file_name", str(im["id"]))),
                                       im.get("width"), im.get("height"), source=f"image id {im['id']}")
    orphans = []
    for a in coco["annotations"]:
        img = images.get(a.get("image_id"))
        if img is None:
            orphans.append(a.get("id"))
            continue
        label = cats.get(a.get("category_id"))
        if label is None:
            label = f"<category_id {a.get('category_id')}>"
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list) and len(seg[0]) >= 6 and not _seg_is_bbox(seg, a):
            for ring in seg:
                pts = list(zip(ring[0::2], ring[1::2]))
                img.shapes.append(Shape("polygon", label, pts, ref=str(a.get("id"))))
        elif a.get("bbox"):
            x, y, w, h = a["bbox"]
            img.shapes.append(Shape("box", label, _rect_corners(x, y, w, h, 0), w, h, ref=str(a.get("id"))))
    if orphans:
        print(f"warning: {len(orphans)} COCO annotations point at missing image ids: {orphans[:10]}", file=sys.stderr)
    return list(images.values())


def _seg_is_bbox(seg: list, ann: dict) -> bool:
    """Some exporters write the box again as a 4-point 'segmentation'. Treat that as a box, not a polygon."""
    if len(seg) != 1 or len(seg[0]) != 8 or not ann.get("bbox"):
        return False
    x, y, w, h = ann["bbox"]
    xs, ys = seg[0][0::2], seg[0][1::2]
    return (abs(min(xs) - x) < 1 and abs(min(ys) - y) < 1 and abs(max(xs) - (x + w)) < 1
            and abs(max(ys) - (y + h)) < 1 and len(set(xs)) <= 2 and len(set(ys)) <= 2)


# --------------------------------------------------------------------------------------------- geometry

def polygon_area(pts: Sequence[Point]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, list(pts[1:]) + [pts[0]]))) / 2.0


def box_iou(a: Shape, b: Shape) -> float:
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _row_masks(pts: Sequence[Point], x0: float, y0: float, cell: float, nx: int, ny: int) -> List[int]:
    """Scanline-fill a polygon onto an nx*ny grid; each row is an int bitmask (even-odd rule)."""
    edges = list(zip(pts, list(pts[1:]) + [pts[0]]))
    rows = []
    for j in range(ny):
        yc = y0 + (j + 0.5) * cell
        xs = sorted(x1 + (yc - y1) * (x2 - x1) / (y2 - y1)
                    for (x1, y1), (x2, y2) in edges if (y1 <= yc < y2) or (y2 <= yc < y1))
        mask = 0
        for left, right in zip(xs[0::2], xs[1::2]):
            i0 = max(0, math.ceil((left - x0) / cell - 0.5))
            i1 = min(nx - 1, math.floor((right - x0) / cell - 0.5))
            if i1 >= i0:
                mask |= ((1 << (i1 - i0 + 1)) - 1) << i0
        rows.append(mask)
    return rows


def polygon_iou(a: Shape, b: Shape, grid: int = 400) -> float:
    if box_iou(a, b) == 0.0:
        return 0.0
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    cell = max(x1 - x0, y1 - y0) / grid or 1.0
    nx, ny = int(math.ceil((x1 - x0) / cell)) + 1, int(math.ceil((y1 - y0) / cell)) + 1
    ma, mb = _row_masks(a.points, x0, y0, cell, nx, ny), _row_masks(b.points, x0, y0, cell, nx, ny)
    inter = sum(bin(p & q).count("1") for p, q in zip(ma, mb))
    union = sum(bin(p | q).count("1") for p, q in zip(ma, mb))
    return inter / union if union else 0.0


# --------------------------------------------------------------------------------------------- check

@dataclass
class Issue:
    severity: str   # "error" or "warning"
    code: str
    image: str
    message: str


def check(images: List[ImageLabels], classes: Optional[set], images_dir: Optional[Path] = None,
          tol: float = 0.5, min_size: float = 1.0, dup_iou: float = 0.95) -> List[Issue]:
    issues: List[Issue] = []

    def add(sev: str, code: str, img: ImageLabels, msg: str) -> None:
        issues.append(Issue(sev, code, img.key, f"{msg}  [{img.source}]"))

    for img in images:
        if not img.shapes:
            if img.marked_empty:
                add("warning", "MARKED_EMPTY", img, "no shapes, ticked 'No objects' - confirm it really is empty")
            else:
                add("error", "NO_LABELS", img, "image has no boxes or polygons and is not marked 'No objects'")
            continue
        if img.marked_empty:
            add("error", "EMPTY_FLAG_CONFLICT", img, f"ticked 'No objects' but has {len(img.shapes)} shapes")

        has_size = bool(img.width and img.height)
        if not has_size:
            add("warning", "NO_IMAGE_SIZE", img, "image width/height unknown - bounds not checked")

        for s in img.shapes:
            what = f"{s.kind} {s.ref or '?'} ({s.label or 'no class'})"
            # class name
            if s.label is None:
                add("error", "MISSING_CLASS", img, f"{what} has no class")
            elif classes is not None and s.label not in classes:
                add("error", "UNKNOWN_CLASS", img, f"{what}: '{s.label}' is not in the class list")
            if s.extra_labels:
                add("error", "MULTIPLE_CLASSES", img, f"{what} also labelled {s.extra_labels}")
            # size
            if s.kind == "box":
                if s.width < min_size or s.height < min_size:
                    add("error", "ZERO_SIZE", img, f"{what} is {s.width:.1f} x {s.height:.1f} px")
            else:
                if len(s.points) < 3:
                    add("error", "DEGENERATE_POLYGON", img, f"{what} has only {len(s.points)} points")
                elif polygon_area(s.points) < min_size:
                    add("error", "DEGENERATE_POLYGON", img, f"{what} has zero area")
            # bounds
            if has_size and s.points:
                x0, y0, x1, y1 = s.bbox()
                if x0 < -tol or y0 < -tol or x1 > img.width + tol or y1 > img.height + tol:
                    fully = x1 <= 0 or y1 <= 0 or x0 >= img.width or y0 >= img.height
                    add("error", "OUT_OF_BOUNDS", img,
                        f"{what} spans x {x0:.1f}..{x1:.1f}, y {y0:.1f}..{y1:.1f} but image is "
                        f"{img.width:.0f} x {img.height:.0f}" + (" (entirely outside)" if fully else ""))

        boxes = [s for s in img.shapes if s.kind == "box"]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if a.label == b.label and box_iou(a, b) >= dup_iou:
                    add("warning", "DUPLICATE", img, f"boxes {a.ref} and {b.ref} ({a.label}) overlap "
                                                     f"IoU {box_iou(a, b):.2f} - probably drawn twice")

    if images_dir is not None:
        seen = {img.key for img in images}
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS and p.name not in seen:
                issues.append(Issue("error", "NO_LABELS", p.name, "image is in --images-dir but not in the export"))
    return issues


def print_check_report(path: Path, images: List[ImageLabels], issues: List[Issue]) -> None:
    n_shapes = Counter(s.kind for img in images for s in img.shapes)
    n_cls = Counter(s.label for img in images for s in img.shapes)
    print(f"== {path}")
    print(f"   {len({i.key for i in images})} images, {n_shapes['box']} boxes, {n_shapes['polygon']} polygons")
    print("   per class (boxes + polygons): " + ", ".join(f"{k}={v}" for k, v in sorted(n_cls.items(), key=lambda kv: str(kv[0]))))
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    for i in sorted(issues, key=lambda i: (i.severity != "error", i.code, i.image)):
        print(f"   {i.severity.upper():7} {i.code:20} {i.image}: {i.message}")
    summary = Counter(i.code for i in issues)
    print(f"   -> {len(errors)} errors, {len(warnings)} warnings"
          + (f"  ({', '.join(f'{k}: {v}' for k, v in sorted(summary.items()))})" if summary else ""))


# --------------------------------------------------------------------------------------------- agree

def greedy_match(a: List[Shape], b: List[Shape], iou_fn, thr: float) -> List[Tuple[int, int, float]]:
    """Class-agnostic greedy matching, highest IoU first (as in COCO evaluation)."""
    pairs = sorted(((iou_fn(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((i, j, iou))
    return matches


def _f1(m: int, na: int, nb: int) -> float:
    return 2 * m / (na + nb) if na + nb else 1.0


def agreement(pass_a: Dict[str, ImageLabels], pass_b: Dict[str, ImageLabels], kind: str,
              thr: float, sweep: Iterable[float] = (0.5, 0.75, 0.9)) -> dict:
    iou_fn = box_iou if kind == "box" else polygon_iou
    common = sorted(set(pass_a) & set(pass_b))
    totals = Counter()
    ious: List[float] = []
    confusion = Counter()
    per_class = defaultdict(Counter)
    per_image = []
    sweep_m = Counter()
    for key in common:
        sa = [s for s in pass_a[key].shapes if s.kind == kind]
        sb = [s for s in pass_b[key].shapes if s.kind == kind]
        if not sa and not sb:
            continue
        # match once at the lowest threshold, then count how many survive each stricter one
        all_m = greedy_match(sa, sb, iou_fn, min([thr, *sweep]))
        for t in sweep:
            sweep_m[t] += sum(1 for _, _, iou in all_m if iou >= t)
        m = [x for x in all_m if x[2] >= thr]
        strict = sum(1 for i, j, _ in m if sa[i].label == sb[j].label)
        totals.update(a=len(sa), b=len(sb), matched=len(m), strict=strict)
        ious += [iou for _, _, iou in m]
        for i, j, _ in m:
            if sa[i].label != sb[j].label:
                confusion[(sa[i].label, sb[j].label)] += 1
        for s in sa:
            per_class[s.label]["a"] += 1
        for s in sb:
            per_class[s.label]["b"] += 1
        for i, j, _ in m:
            if sa[i].label == sb[j].label:
                per_class[sa[i].label]["strict"] += 1
        per_image.append((_f1(strict, len(sa), len(sb)), key, len(sa), len(sb), strict))
    ious.sort()
    return {
        "kind": kind, "thr": thr, "images": len(per_image), "totals": totals, "ious": ious,
        "confusion": confusion, "per_class": per_class, "per_image": sorted(per_image),
        "sweep": {t: _f1(sweep_m[t], totals["a"], totals["b"]) for t in sweep},
    }


def print_agreement(r: dict, worst: int) -> None:
    t = r["totals"]
    if not r["images"]:
        print(f"\n-- {PLURAL[r['kind']]}: none in either pass")
        return
    ious = r["ious"]
    print(f"\n-- {PLURAL[r['kind']]} ({r['images']} images; pass A {t['a']}, pass B {t['b']}; match at IoU >= {r['thr']})")
    print(f"   F1 (location + class)      {_f1(t['strict'], t['a'], t['b']):.3f}")
    print(f"   F1 (location only)         {_f1(t['matched'], t['a'], t['b']):.3f}")
    if ious:
        med = ious[len(ious) // 2] if len(ious) % 2 else (ious[len(ious) // 2 - 1] + ious[len(ious) // 2]) / 2
        print(f"   IoU of matched pairs       mean {sum(ious) / len(ious):.3f}, median {med:.3f}, min {ious[0]:.3f}")
        print(f"   class agreement (matched)  {t['strict'] / t['matched']:.1%}  ({t['strict']}/{t['matched']})")
    print("   F1 (location only) at IoU  " + ", ".join(f">={k}: {v:.3f}" for k, v in r["sweep"].items()))
    print("   per class F1               " + ", ".join(
        f"{c}: {_f1(v['strict'], v['a'], v['b']):.3f} (A {v['a']}, B {v['b']})"
        for c, v in sorted(r["per_class"].items(), key=lambda kv: str(kv[0]))))
    if r["confusion"]:
        print("   class swaps (A -> B)       " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in r["confusion"].most_common()))
    bad = [x for x in r["per_image"] if x[0] < 1.0][:worst]
    if bad:
        print(f"   lowest-agreement images:")
        for f1, key, na, nb, m in bad:
            print(f"     {key:24} F1 {f1:.2f}  (A {na}, B {nb}, agreed {m})")


def _by_image(images: List[ImageLabels], name: str) -> Dict[str, ImageLabels]:
    out: Dict[str, ImageLabels] = {}
    for img in images:
        if img.key in out:
            print(f"warning: {name} has more than one annotation for {img.key}; using the first", file=sys.stderr)
            continue
        out[img.key] = img
    return out


# --------------------------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate one or more exports")
    c.add_argument("exports", nargs="+", type=Path)
    c.add_argument("--classes", type=Path, help="file with one allowed class name per line")
    c.add_argument("--images-dir", type=Path, help="flag images in this folder that are missing from the export")
    c.add_argument("--tolerance", type=float, default=0.5, help="px a shape may stick out before it counts (0.5)")
    c.add_argument("--min-size", type=float, default=1.0, help="boxes narrower/shorter than this many px are zero-size")

    g = sub.add_parser("agree", help="compare two labelling passes")
    g.add_argument("pass_a", type=Path)
    g.add_argument("pass_b", type=Path, nargs="?", help="omit to compare annotation 1 vs 2 inside pass_a")
    g.add_argument("--iou", type=float, default=0.5, help="IoU needed for two shapes to count as the same object")
    g.add_argument("--worst", type=int, default=5, help="how many low-agreement images to list")
    g.add_argument("--json", type=Path, help="also write the headline numbers to this file")

    args = ap.parse_args(argv)

    if args.cmd == "check":
        classes = None
        if args.classes:
            classes = {ln.strip() for ln in args.classes.read_text().splitlines() if ln.strip()}
        failed = False
        for path in args.exports:
            images = load(path)
            issues = check(images, classes, args.images_dir, args.tolerance, args.min_size)
            print_check_report(path, images, issues)
            failed |= any(i.severity == "error" for i in issues)
        return 1 if failed else 0

    if args.pass_b is None:
        a, b = load(args.pass_a, 0), load(args.pass_a, 1)
        names = (f"{args.pass_a} [annotation 1]", f"{args.pass_a} [annotation 2]")
    else:
        a, b = load(args.pass_a, 0), load(args.pass_b, 0)
        names = (str(args.pass_a), str(args.pass_b))
    pa, pb = _by_image(a, names[0]), _by_image(b, names[1])
    print(f"Pass A: {names[0]}  ({len(pa)} images)\nPass B: {names[1]}  ({len(pb)} images)")
    only_a, only_b = sorted(set(pa) - set(pb)), sorted(set(pb) - set(pa))
    if only_a:
        print(f"   only in pass A ({len(only_a)}): {', '.join(only_a[:10])}{' ...' if len(only_a) > 10 else ''}")
    if only_b:
        print(f"   only in pass B ({len(only_b)}): {', '.join(only_b[:10])}{' ...' if len(only_b) > 10 else ''}")
    results = [agreement(pa, pb, "box", args.iou), agreement(pa, pb, "polygon", args.iou)]
    for r in results:
        print_agreement(r, args.worst)
    if args.json:
        summary = {}
        for r in results:
            t = r["totals"]
            summary[r["kind"]] = {
                "images": r["images"], "pass_a": t["a"], "pass_b": t["b"], "iou_threshold": r["thr"],
                "f1": round(_f1(t["strict"], t["a"], t["b"]), 4),
                "f1_location_only": round(_f1(t["matched"], t["a"], t["b"]), 4),
                "mean_iou": round(sum(r["ious"]) / len(r["ious"]), 4) if r["ious"] else None,
                "f1_at_iou": {str(k): round(v, 4) for k, v in r["sweep"].items()},
            }
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
