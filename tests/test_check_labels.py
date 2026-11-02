"""Tests for tools/check_labels.py. Run with:  python -m pytest -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import check_labels as cl  # noqa: E402

W, H = 400, 200  # image size used by all synthetic tasks
CLASSES = {"person", "two_wheeler", "vehicle"}


def rect(x, y, w, h, label="person", rid="r", rotation=0):
    """A Label Studio rectanglelabels result, given in pixels (converted to % like the real export)."""
    return {"id": rid, "type": "rectanglelabels", "from_name": "box", "to_name": "image",
            "original_width": W, "original_height": H,
            "value": {"x": x / W * 100, "y": y / H * 100, "width": w / W * 100, "height": h / H * 100,
                      "rotation": rotation, "rectanglelabels": [label] if label else []}}


def poly(points, label="person", rid="p"):
    return {"id": rid, "type": "polygonlabels", "from_name": "poly", "to_name": "image",
            "original_width": W, "original_height": H,
            "value": {"points": [[x / W * 100, y / H * 100] for x, y in points], "polygonlabels": [label]}}


def task(tid, name, *annotations):
    return {"id": tid, "data": {"image": f"/data/upload/1/{tid:08x}-{name}"},
            "annotations": [{"id": tid * 10 + i, "result": list(r), "was_cancelled": False}
                            for i, r in enumerate(annotations)]}


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def codes(issues):
    return sorted(i.code for i in issues)


# ------------------------------------------------------------------ check

def test_clean_export_has_no_issues(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(10, 10, 50, 100), poly([(10, 10), (60, 10), (35, 110)])])])
    assert cl.check(cl.load(p), CLASSES) == []


def test_upload_prefix_is_stripped():
    assert cl.image_key("/data/upload/3/1a2b3c4d-FudanPed00001.png") == "FudanPed00001.png"
    assert cl.image_key("/data/local-files/?d=images/PennPed00012.png") == "PennPed00012.png"
    assert cl.image_key("images/PennPed00012.png") == "PennPed00012.png"
    assert cl.image_key("images\\x.jpg") == "x.jpg"


@pytest.mark.parametrize("box,expect_fully", [
    ((380, 10, 40, 40), False),   # right edge at 420 > 400
    ((-5, 10, 40, 40), False),    # left edge negative
    ((10, 190, 40, 40), False),   # bottom past 200
    ((500, 10, 40, 40), True),    # entirely outside
])
def test_out_of_bounds(tmp_path, box, expect_fully):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(*box)])])
    issues = cl.check(cl.load(p), CLASSES)
    assert codes(issues) == ["OUT_OF_BOUNDS"]
    assert ("entirely outside" in issues[0].message) == expect_fully


def test_box_on_the_border_is_fine(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(0, 0, W, H), rect(W - 20.3, 0, 20.3 + 0.4, 10, rid="b")])])
    assert cl.check(cl.load(p), CLASSES) == []


def test_rotated_box_corner_outside(tmp_path):
    # 45 degree rotation around the top-left corner swings the box past the bottom edge
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(200, 100, 100, 100, rotation=45)])])
    assert "OUT_OF_BOUNDS" in codes(cl.check(cl.load(p), CLASSES))


def test_out_of_bounds_polygon(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(1, 1, 5, 5), poly([(10, 10), (450, 10), (30, 50)])])])
    assert codes(cl.check(cl.load(p), CLASSES)) == ["OUT_OF_BOUNDS"]


@pytest.mark.parametrize("w,h", [(0, 50), (50, 0), (0.4, 30), (0, 0)])
def test_zero_size_box(tmp_path, w, h):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(10, 10, w, h)])])
    assert codes(cl.check(cl.load(p), CLASSES)) == ["ZERO_SIZE"]


def test_degenerate_polygons(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [
        poly([(10, 10), (20, 20)], rid="two-points"),
        poly([(10, 10), (20, 20), (30, 30)], rid="collinear"),
    ])])
    assert codes(cl.check(cl.load(p), CLASSES)) == ["DEGENERATE_POLYGON", "DEGENERATE_POLYGON"]


def test_unknown_and_missing_class(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [
        rect(10, 10, 20, 20, label="Person"),       # wrong case
        rect(40, 10, 20, 20, label="pedestrian"),
        rect(70, 10, 20, 20, label=None, rid="nolabel"),
    ])])
    assert codes(cl.check(cl.load(p), CLASSES)) == ["MISSING_CLASS", "UNKNOWN_CLASS", "UNKNOWN_CLASS"]


def test_class_check_skipped_without_class_list(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(10, 10, 20, 20, label="anything")])])
    assert cl.check(cl.load(p), None) == []


def test_images_with_no_labels(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for n in ("a.png", "b.png", "c.png", "never_labelled.png", "notes.txt"):
        (images_dir / n).write_bytes(b"")
    empty_flag = {"id": "f", "type": "choices", "from_name": "flags", "to_name": "image",
                  "original_width": W, "original_height": H, "value": {"choices": ["No objects"]}}
    p = write(tmp_path, "a.json", [
        task(1, "a.png", [rect(10, 10, 20, 20)]),
        task(2, "b.png", []),               # submitted with nothing drawn
        task(3, "c.png", [empty_flag]),     # legitimately empty
        {"id": 4, "data": {"image": "/data/upload/1/00000004-d.png"}, "annotations": []},  # exported, never done
    ])
    issues = cl.check(cl.load(p), CLASSES, images_dir)
    by_image = {(i.image, i.code) for i in issues}
    assert by_image == {("b.png", "NO_LABELS"), ("d.png", "NO_LABELS"), ("c.png", "MARKED_EMPTY"),
                        ("never_labelled.png", "NO_LABELS")}


def test_empty_flag_with_shapes_is_an_error(tmp_path):
    flag = {"id": "f", "type": "choices", "value": {"choices": ["No objects"]}}
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(10, 10, 20, 20), flag])])
    assert codes(cl.check(cl.load(p), CLASSES)) == ["EMPTY_FLAG_CONFLICT"]


def test_duplicate_box_warning(tmp_path):
    p = write(tmp_path, "a.json", [task(1, "a.png", [rect(10, 10, 50, 50, rid="x"), rect(10.5, 10, 50, 50, rid="y")])])
    issues = cl.check(cl.load(p), CLASSES)
    assert [(i.code, i.severity) for i in issues] == [("DUPLICATE", "warning")]


def test_coco_export(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": W, "height": H},
                   {"id": 2, "file_name": "images/b.png", "width": W, "height": H}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [390, 10, 20, 20]},   # out of bounds
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},    # unknown class
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 50, 0, 20]},     # zero size
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20],     # polygon
             "segmentation": [[50, 50, 70, 50, 60, 70]]},
            {"id": 5, "image_id": 1, "category_id": 1, "bbox": [100, 50, 20, 20],    # box repeated as seg
             "segmentation": [[100, 50, 120, 50, 120, 70, 100, 70]]},
        ],
    }
    images = cl.load(write(tmp_path, "coco.json", coco))
    kinds = sorted(s.kind for s in images[0].shapes)
    assert kinds == ["box", "box", "box", "box", "polygon"]
    issues = cl.check(images, CLASSES)
    assert codes(issues) == ["NO_LABELS", "OUT_OF_BOUNDS", "UNKNOWN_CLASS", "ZERO_SIZE"]


def test_cli_exit_codes(tmp_path, capsys):
    classes = tmp_path / "classes.txt"
    classes.write_text("person\n")
    good = write(tmp_path, "good.json", [task(1, "a.png", [rect(10, 10, 20, 20)])])
    bad = write(tmp_path, "bad.json", [task(1, "a.png", [rect(10, 10, 0, 20)])])
    assert cl.main(["check", str(good), "--classes", str(classes)]) == 0
    assert cl.main(["check", str(bad), "--classes", str(classes)]) == 1
    assert "ZERO_SIZE" in capsys.readouterr().out


# ------------------------------------------------------------------ geometry / agreement

def test_box_iou_values():
    a = cl.Shape("box", "person", cl._rect_corners(0, 0, 10, 10, 0), 10, 10)
    b = cl.Shape("box", "person", cl._rect_corners(5, 0, 10, 10, 0), 10, 10)
    c = cl.Shape("box", "person", cl._rect_corners(20, 20, 10, 10, 0), 10, 10)
    assert cl.box_iou(a, a) == pytest.approx(1.0)
    assert cl.box_iou(a, b) == pytest.approx(50 / 150)
    assert cl.box_iou(a, c) == 0.0


def test_polygon_iou_matches_analytic_value():
    sq = cl.Shape("polygon", "person", [(0, 0), (100, 0), (100, 100), (0, 100)])
    shifted = cl.Shape("polygon", "person", [(50, 0), (150, 0), (150, 100), (50, 100)])
    tri = cl.Shape("polygon", "person", [(0, 0), (100, 0), (0, 100)])  # half the square
    assert cl.polygon_iou(sq, sq) == pytest.approx(1.0)
    assert cl.polygon_iou(sq, shifted) == pytest.approx(1 / 3, abs=0.01)
    assert cl.polygon_iou(sq, tri) == pytest.approx(0.5, abs=0.01)


def test_identical_passes_agree_perfectly(tmp_path):
    shapes = [rect(10, 10, 50, 100), rect(200, 50, 80, 60, "vehicle", "v"), poly([(10, 10), (60, 10), (35, 110)])]
    a = cl._by_image(cl.load(write(tmp_path, "a.json", [task(1, "a.png", shapes)])), "a")
    r = cl.agreement(a, a, "box", 0.5)
    assert r["totals"]["strict"] == 2 and r["ious"] == [pytest.approx(1.0)] * 2
    assert cl.agreement(a, a, "polygon", 0.5)["totals"]["strict"] == 1


def test_agreement_counts_misses_swaps_and_loose_boxes(tmp_path):
    pass1 = [task(1, "a.png", [rect(10, 10, 50, 100, rid="p1"), rect(200, 50, 80, 60, "vehicle", "v1"),
                               rect(300, 10, 20, 40, rid="small")]),
             task(2, "b.png", [rect(10, 10, 40, 40)])]
    pass2 = [task(5, "a.png", [rect(12, 10, 50, 104, rid="p1"),                  # slightly different box
                               rect(200, 50, 80, 60, "two_wheeler", "v1")]),     # same place, other class
             task(6, "b.png", [rect(10, 10, 40, 40)])]                            # (small person missed)
    a = cl._by_image(cl.load(write(tmp_path, "a.json", pass1)), "a")
    b = cl._by_image(cl.load(write(tmp_path, "b.json", pass2)), "b")
    r = cl.agreement(a, b, "box", 0.5)
    t = r["totals"]
    assert (t["a"], t["b"], t["matched"], t["strict"]) == (4, 3, 3, 2)
    assert r["confusion"] == {("vehicle", "two_wheeler"): 1}
    assert cl._f1(t["strict"], t["a"], t["b"]) == pytest.approx(4 / 7)
    assert r["per_image"][0][1] == "a.png"            # worst image first
    assert r["sweep"][0.9] < r["sweep"][0.5]           # the loose person box fails at IoU 0.9


def test_agree_cli_single_file_two_annotations(tmp_path, capsys):
    both = write(tmp_path, "both.json", [task(1, "a.png", [rect(10, 10, 50, 100)], [rect(11, 10, 50, 100)])])
    out_json = tmp_path / "summary.json"
    assert cl.main(["agree", str(both), "--json", str(out_json)]) == 0
    summary = json.loads(out_json.read_text())
    assert summary["box"]["f1"] == 1.0 and summary["box"]["mean_iou"] > 0.95
    assert "F1 (location + class)" in capsys.readouterr().out
