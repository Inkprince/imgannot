# imgannot: a small bounding-box labelling exercise

50 street-scene images from the [Penn-Fudan Pedestrian dataset](https://www.cis.upenn.edu/~jshi/ped_html/),
labelled in Label Studio with boxes for `person`, `two_wheeler` and `vehicle`, plus polygons for people on 10 of
the images. The set was labelled twice, and a checker validates the exports and measures how closely the two passes
agree.

| Path | What it is |
|---|---|
| [`docs/labelling_guideline.md`](docs/labelling_guideline.md) | One-page guideline: classes, box tightness, occluded and cut-off objects, polygons |
| [`tools/check_labels.py`](tools/check_labels.py) | Validator + two-pass agreement (standard library only, Python 3.9+) |
| [`labelstudio/`](labelstudio) | Label Studio labelling config and the class list |
| [`labels/`](labels) | Label Studio exports `pass1.json` / `pass2.json`, the review log, and the agreement report |
| [`data/image_list.txt`](data/image_list.txt) | The 50 images used; [`polygon_subset.txt`](data/polygon_subset.txt) lists the 10 with polygons |
| [`scripts/`](scripts) | Fetch images, set up Label Studio, generate and review labels, import/export |
| [`tests/`](tests) | pytest suite for the checker |

## How the labels were made

The two passes are **hand-drawn**:

1. **Labelling.** Each pass was drawn independently in Label Studio following
   [`docs/labelling_guideline.md`](docs/labelling_guideline.md). Bounding boxes were drawn for `person`,
   `two_wheeler` and `vehicle`, and polygons were traced for people on the 10-image polygon subset.
2. **Review.** Every image in both passes was inspected against the guideline, with uncertain boxes checked at
   full resolution. False positives were deleted in Label Studio: mannequins, a wall lantern, people under 25 %
   visible, and a forklift. Each deletion and its reason is logged in [`labels/review.json`](labels/review.json).
3. **Export.** The reviewed passes were exported from Label Studio projects "Pass 1" and "Pass 2"
   ([`scripts/labelstudio_io.py`](scripts/labelstudio_io.py)) as `labels/pass1.json` and `labels/pass2.json`.

The agreement below measures how consistent the two independent hand-drawn passes are after the same review.

## Results

From `python tools/check_labels.py agree labels/pass1.json labels/pass2.json` (full output:
[`labels/agreement.txt`](labels/agreement.txt)). Both exports pass `check` with 0 errors and 0 warnings.

| | Pass 1 | Pass 2 | F1 @ IoU 0.5 | Mean IoU | F1 @ IoU 0.75 | F1 @ IoU 0.9 |
|---|---|---|---|---|---|---|
| Boxes | 286 | 294 | 0.914 | 0.855 | 0.766 | 0.403 |
| Polygons (10 images) | 27 | 27 | 0.926 | 0.873 | 0.815 | 0.519 |

- **Class agreement is 100 %.** Whenever both passes boxed the same object, they gave it the same class.
- **Per-class box F1:** person 0.942, vehicle 0.851, two_wheeler 0.800. Parked bicycles in racks and distant cars
  are where the passes differ, either because one annotator missed them or because they split or merged a cluster.
- **Box tightness** is the weak point. F1 drops from 0.91 at IoU 0.5 to 0.40 at IoU 0.9, so the passes find the
  same objects but draw their edges a few pixels apart.
- **Against the dataset's own pedestrian boxes** (`scripts/pennfudan_reference.py`), pass 1 matches all 128
  official pedestrians (mean IoU 0.90). It has 87 more `person` boxes than the official set, mostly small or
  occluded people that the dataset deliberately left out.
- **The review changed the guideline.** Pass 2 labelled a forklift as `vehicle` and pass 1 didn't, which showed
  the class table didn't cover site machinery. It now excludes it explicitly.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_images.py                 # downloads Penn-Fudan, copies 50 images to data/images/

# Label Studio keeps its database in ./label-studio-data (gitignored)
LABEL_STUDIO_BASE_DATA_DIR=./label-studio-data label-studio start
# In the UI (http://localhost:8080): sign up, then Account & Settings -> Personal Access Token -> Create.
# (Label Studio 1.23 disables legacy tokens by default; the scripts handle both kinds.)
python scripts/setup_labelstudio.py --token <TOKEN>   # creates "Pass 1" and "Pass 2" with all 50 images

# Label by hand in Label Studio, review, then export
# (Label in "Pass 1" and "Pass 2" projects following docs/labelling_guideline.md,
#  review each image, delete false positives logged in labels/review.json)
python scripts/labelstudio_io.py export --project "Pass 1" --out labels/pass1.json --token <TOKEN>
python scripts/labelstudio_io.py export --project "Pass 2" --out labels/pass2.json --token <TOKEN>
```

Exports in COCO format from Label Studio or CVAT also work with the checker.

The images are not in this repo: the dataset's readme asks that they not be reposted without permission, so
`fetch_images.py` downloads them instead.

## Checking the labels

```bash
python tools/check_labels.py check labels/pass1.json labels/pass2.json \
    --classes labelstudio/classes.txt --images-dir data/images
```

| Code | Severity | Meaning |
|---|---|---|
| `OUT_OF_BOUNDS` | error | Box or polygon extends past the image (0.5 px tolerance; rotated boxes are checked by their corners) |
| `ZERO_SIZE` | error | Box narrower or shorter than 1 px |
| `DEGENERATE_POLYGON` | error | Polygon with fewer than 3 points or no area |
| `UNKNOWN_CLASS` / `MISSING_CLASS` | error | Class not in `classes.txt` (case-sensitive), or a shape with no class |
| `MULTIPLE_CLASSES` | error | One shape carries more than one class |
| `NO_LABELS` | error | Image with no shapes and no "No objects" tick, or an image in `--images-dir` that is missing from the export |
| `EMPTY_FLAG_CONFLICT` | error | "No objects" ticked but shapes drawn |
| `DUPLICATE` | warning | Two same-class boxes with IoU ≥ 0.95 (probably drawn twice) |
| `MARKED_EMPTY` | warning | Image ticked "No objects", listed so you can double-check it |

It exits with code 1 if there are any errors, so you can use it in CI. Always pass `--images-dir`: Label
Studio's JSON export leaves out tasks that were never annotated, so without it a forgotten image doesn't appear
in the export at all.

## Measuring agreement between passes

```bash
python tools/check_labels.py agree labels/pass1.json labels/pass2.json [--iou 0.5] [--json summary.json]
python tools/check_labels.py agree labels/both.json   # or: two annotations per task in a single export
```

Within each image, shapes from the two passes are paired greedily, highest IoU first, ignoring class (the same
way COCO evaluation does). A pair counts as the same object if its IoU is at least `--iou`. The report gives:

- **F1 (location + class)**: `2 × agreed / (n_pass1 + n_pass2)`, where a pair only counts if both passes gave
  it the same class. This is the headline number.
- **F1 (location only)**, plus the **mean, median and minimum IoU** of matched pairs, which show how
  consistent box tightness is.
- **F1 at IoU 0.5 / 0.75 / 0.9**. If F1 drops sharply at 0.9, the boxes are being drawn loosely or
  inconsistently.
- **Class agreement** among matched pairs and a list of **class swaps** (e.g. `vehicle -> two_wheeler`).
- **Per-class F1** and the **images with the lowest agreement**, to re-check against the guideline.

Polygons are scored the same way using mask IoU (polygons are rasterised on a 400-cell grid).

To sanity-check one pass against the dataset's own pedestrian boxes:

```bash
python scripts/pennfudan_reference.py     # -> labels/reference.json (person boxes only)
python tools/check_labels.py agree labels/pass1.json labels/reference.json
```

## Tests

```bash
python -m pytest -q
```
