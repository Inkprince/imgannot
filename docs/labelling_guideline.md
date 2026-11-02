# Labelling guideline — Penn-Fudan street scenes (50 images)

**Scope:** the 50 images in `data/image_list.txt`. **Tool:** Label Studio with `labelstudio/label_config.xml`.
**Goal:** a detector training set for people and road users. If a rule below doesn't settle a case, follow the
spirit of "would a detector be rewarded for finding this?" and note the image name for review.

## 1. Classes

| Class | Includes | Excludes |
|---|---|---|
| `person` | Any real human: walking, standing, sitting, cycling (the rider is a `person`), children, people seen through glass | Reflections, posters, statues, mannequins, people on screens |
| `two_wheeler` | Bicycles, motorbikes, scooters — parked or ridden | Wheelchairs, prams/strollers (not labelled) |
| `vehicle` | Cars, vans, trucks, buses, taxis | Trains, trailers on their own, forklifts and other site machinery, toy cars |

No other class names. Spelling and case must match exactly (`person`, not `Person` or `pedestrian`).

## 2. What to label

- Label **every** instance that is at least **15 px tall** (person) or **15 px on its shorter side** (two_wheeler,
  vehicle) **and** at least **25 % visible**. Smaller or more hidden objects are skipped, never half-labelled.
- One box per object. A ridden bicycle gets two boxes: one `person`, one `two_wheeler`.
- If an image contains nothing to label, tick **No objects**. Never submit an image with no boxes and no tick.

## 3. How tight boxes must be

- The box edges touch the **outermost visible pixels** of the object: hair, hats, clothing, limbs and shoes are
  included. Carried bags, umbrellas, prams and shadows are **excluded**.
- Allowed slack: **≤ 2 px on each side**, or 2 % of the object's size for objects over 100 px. No edge may cut
  into the object.
- Zoom in (scroll) for any object under ~60 px; check all four edges before moving on.
- A good box overlaps a second careful attempt by **IoU ≥ 0.85**. The two-pass check in `tools/check_labels.py`
  measures this.

## 4. Occluded objects (something in front)

- Box the **visible extent only** (modal boxes). Don't guess where hidden legs or a hidden car bonnet would be.
- If a thin occluder (lamp post, sign pole, another person's arm) splits the object into two visible parts,
  draw **one** box spanning both parts.
- People in a group: box each person you can separate. If you can't tell where one person ends and the next
  begins, and they are under 25 % visible each, skip them.

## 5. Cut-off objects (touching the image edge)

- Label them if they meet the size and 25 % visibility rule. The box goes **exactly to the image border** and
  never past it. Label Studio stops you dragging outside, but check after resizing.
- Don't extend the box to where the object "would be" outside the frame.

## 6. Polygons

- Only on the 10 images in `data/polygon_subset.txt`, and only for `person`. Each person in those images gets a
  polygon **in addition to** the box.
- Put a vertex at every change of direction, roughly every 10–20 px along the outline. Points within 2 px of the
  true edge. Close the polygon (click the first point).
- Follow the visible outline. Trace across thin occluders (narrower than ~10 px); for wider occluders, outline
  only the largest visible part.

## 7. Two-pass protocol (for the agreement check)

1. Pass 1 in project **"Pass 1"**. Export as **JSON** to `labels/pass1.json`.
2. Wait at least a day. Pass 2 in a separate project **"Pass 2"**, *without* looking at pass 1. Export to
   `labels/pass2.json`.
3. Run `python tools/check_labels.py check …` on both, fix errors, then run `agree`. Investigate any image with
   F1 < 0.8 and update this guideline if the disagreement came from an unclear rule.
