"""
Build the v2 robot dataset: repair missing robot boxes in the Roboflow data and add mergeRM,
keeping a new robot box only when two independent signals agree:
    1. the current model detects a robot there, and
    2. a human-labelled armor plate sits inside that detection.

Roboflow (data/robots, built by prepare_data.py):
    ~30% of train images have robots whose armor was labelled but whose robot box is missing.
    A detection that contains such an "orphan" armor plate is added, with the colour taken from
    the human armor label (armor_blue -> robot_blue, armor_red -> robot_red, armor_grey -> robot_unknown).
    val is repaired too (it picks best.pt); test keeps its original labels, and test_repaired is added.

mergeRM (~/Datasets/mergeRM, armor-plate labels only):
    Whole matches and a block of robot-camera footage are held out for honest evaluation and never
    trained on (data/heldout). For the rest, an image is kept only if every labelled plate lies inside
    an accepted robot box and the model's red/blue never contradicts a lit plate's colour (HSV check,
    99.5% correct on Roboflow armor). Plate-free images are kept as backgrounds if the model sees nothing.

Usage (on the server, from robot-detection/):
    python pseudo_label.py --model runs/robots_yolo11s/weights/best.pt \
        --zip ~/jithendra/datasets/robomaster.v1i.yolov8.zip --mergerm ~/Datasets/mergeRM

Outputs:
    data/robots_v2/{images,labels}/{train,val,test,test_repaired}/, data/robots_v2/data.yaml
    data/heldout/{images,labels}/   (held-out mergeRM images + original plate labels)
    data/robots_v2/qa/*.jpg         (sample images with new boxes, to eyeball)
"""

import argparse
import os
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from record import folder_fingerprint, md5, write_record

HERE = Path(__file__).resolve().parent
CLASSES = ["robot_blue", "robot_red", "robot_unknown"]
ARMOR_TO_ROBOT = {0: 0, 1: 2, 2: 1}  # Roboflow armor_blue / armor_grey / armor_red -> our class
HELDOUT_MATCHES = {"ARUWvsTAMU", "NYUvsPUTR"}
HELDOUT_DS2_FRACTION = 0.15  # highest ds2 frame ids are held out
DS2_BUFFER = 30  # ids just below the held-out block are dropped (neighbouring frames)
SZ = 640
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_files(folder):
    """Image files only (mergeRM also contains Jupyter's .ipynb_checkpoints folders)."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def poly_box(vals):
    """Normalised polygon 'x1 y1 x2 y2 ...' -> pixel xyxy box and centre."""
    p = np.clip(np.array(vals, float).reshape(-1, 2), 0, 1) * SZ
    box = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
    return box, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def to_yolo(cls, box):
    x0, y0, x1, y1 = [min(max(v / SZ, 0.0), 1.0) for v in box]
    return f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def plate_colour(img, box):
    """0 = blue, 1 = red, None = unlit or unclear. Validated: 99.5% correct on lit Roboflow armor."""
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = img[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][(hsv[..., 1] > 100) & (hsv[..., 2] > 120)]
    red, blue = int(((h <= 10) | (h >= 170)).sum()), int(((h >= 95) & (h <= 135)).sum())
    if max(red, blue) < 6:
        return None
    return 1 if red >= 3 * blue else 0 if blue >= 3 * red else None


def write_labels(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(to_yolo(c, b) + "\n" for c, b in rows))


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def detect(model, paths, batch=64):
    """Yield (path, [(cls, conf, xyxy)]) for each image, in fixed-size batches."""
    for i in range(0, len(paths), batch):
        for p, r in zip(paths[i:i + batch], model.predict([str(p) for p in paths[i:i + batch]],
                                                          conf=0.25, imgsz=SZ, verbose=False)):
            yield p, [(int(c), float(s), b.tolist()) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


def repair_roboflow(model, zf, src, out, stats, qa):
    """Add robot boxes around orphan armor plates in the Roboflow splits."""
    for split, zsplit in [("train", "train"), ("val", "valid"), ("test", "test")]:
        paths = image_files(src / "images" / split)
        for path, dets in detect(model, paths):
            gt = []
            for line in (src / "labels" / split / f"{path.stem}.txt").read_text().splitlines():
                c, x, y, w, h = line.split()
                x, y, w, h = (float(v) * SZ for v in (x, y, w, h))
                gt.append((int(c), [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
            armors = []
            for line in set(zf.read(f"{zsplit}/labels/{path.stem}.txt").decode().splitlines()):
                v = line.split()
                if v and int(v[0]) <= 2:
                    box, centre = poly_box(v[1:])
                    if not any(inside(centre, g[1]) for g in gt):
                        armors.append((ARMOR_TO_ROBOT[int(v[0])], centre))
            added = []
            for cls, conf, box in sorted(dets, key=lambda d: -d[1]):
                if conf < 0.5 or any(iou(box, g[1]) > 0.3 for g in gt + added):
                    continue
                colours = {c for c, centre in armors if inside(centre, box)}
                if len(colours) == 1:
                    added.append((colours.pop(), box))
            stats[f"roboflow/{split}"]["robots added"] += len(added)
            stats[f"roboflow/{split}"]["images changed"] += bool(added)
            splits = [split] if split != "test" else ["test", "test_repaired"]
            for s in splits:
                rows = gt + (added if s != "test" else [])
                link(path, out / "images" / s / path.name)
                write_labels(out / "labels" / s / f"{path.stem}.txt", rows)
            if added and split == "train" and len(qa["roboflow"]) < 400:
                qa["roboflow"].append((out / "images" / "train" / path.name, gt, added))


def mergerm_sources(root):
    """Map every mergeRM image to 'train' or 'heldout' (whole matches / a block of ds2 ids held out)."""
    images = sorted(p for s in ("train", "val", "test") for p in image_files(root / "images" / s))
    ds2_ids = sorted(int(m.group(1)) for p in images if (m := re.match(r"ds2_(\d+)", p.name)))
    cut = ds2_ids[int(len(ds2_ids) * (1 - HELDOUT_DS2_FRACTION))]
    plan = {}
    for p in images:
        if (m := re.match(r"ds1_(.+?)frame", p.name)):
            plan[p] = "heldout" if m.group(1) in HELDOUT_MATCHES else "train"
        elif (m := re.match(r"ds2_(\d+)", p.name)):
            i = int(m.group(1))
            plan[p] = "heldout" if i >= cut else "skip" if i >= cut - DS2_BUFFER else "train"
        else:
            plan[p] = "train"
    return plan


def label_mergerm(model, root, out, heldout, stats, qa):
    """Pseudo-label mergeRM training images; copy held-out images with their plate labels."""
    plan = mergerm_sources(root)
    for p, dest in plan.items():
        if dest == "heldout":
            link(p, heldout / "images" / p.name)
            lab = root / "labels" / p.parent.name / f"{p.stem}.txt"
            if lab.exists():
                link(lab, heldout / "labels" / lab.name)
        stats["mergeRM"][f"images -> {dest}"] += 1
    train = [p for p, d in plan.items() if d == "train"]
    for path, dets in detect(model, train):
        lab = root / "labels" / path.parent.name / f"{path.stem}.txt"
        if not lab.exists() or not lab.read_text().strip():
            if dets:
                stats["mergeRM"]["rejected: background with detections"] += 1
                continue
            rows = []
        else:
            img = cv2.imread(str(path))
            plates = []
            for line in lab.read_text().splitlines():
                if line.split():
                    box, centre = poly_box(line.split()[1:])
                    plates.append((centre, plate_colour(img, box)))
            accepted, reason = [], None
            for cls, conf, box in sorted(dets, key=lambda d: -d[1]):
                lit = {c for centre, c in plates if inside(centre, box) and c is not None}
                has_plate = any(inside(centre, box) for centre, _ in plates)
                if not ((conf >= 0.5 and has_plate) or conf >= 0.8):
                    continue
                if len(lit) > 1 or (lit and cls == 2) or (lit and cls != lit.copy().pop()):
                    reason = "rejected: model colour vs lit plate"
                    break
                if not any(iou(box, a[1]) > 0.6 for a in accepted):
                    accepted.append((cls, box))
            if reason is None and not all(any(inside(centre, a[1]) for a in accepted) for centre, _ in plates):
                reason = "rejected: plate outside every robot box"
            if reason:
                stats["mergeRM"][reason] += 1
                continue
            rows = accepted
        stats["mergeRM"]["kept for training"] += 1
        stats["mergeRM"]["robots labelled"] += len(rows)
        name = f"mergerm_{path.name}"
        link(path, out / "images" / "train" / name)
        write_labels(out / "labels" / "train" / f"mergerm_{path.stem}.txt", rows)
        if rows and len(qa["mergerm"]) < 400:
            qa["mergerm"].append((out / "images" / "train" / name, [], rows))


def save_grid(items, path, n=16):
    colours = {0: (255, 120, 0), 1: (0, 0, 255), 2: (200, 200, 200)}
    random.Random(0).shuffle(items)
    tiles = []
    for img_path, old, new in items[:n]:
        img = cv2.imread(str(img_path))
        for c, b in old:  # existing labels: thin green
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
        for c, b in new:  # new labels: thick, coloured by class (blue / red / grey)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colours[c], 3)
        tiles.append(cv2.resize(img, (320, 320)))
    while len(tiles) % 4:
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    p = argparse.ArgumentParser(description="Build the v2 dataset (repaired Roboflow + pseudo-labelled mergeRM)")
    p.add_argument("--model", required=True, help="current best robot model (.pt)")
    p.add_argument("--zip", required=True, help="Roboflow export zip (for the armor labels)")
    p.add_argument("--mergerm", required=True, help="mergeRM dataset root")
    p.add_argument("--src", default=str(HERE / "data" / "robots"), help="dataset built by prepare_data.py")
    p.add_argument("--out", default=str(HERE / "data" / "robots_v2"))
    p.add_argument("--heldout", default=str(HERE / "data" / "heldout"))
    args = p.parse_args()
    src, out, heldout = (Path(x).expanduser().resolve() for x in (args.src, args.out, args.heldout))
    for d in (out, heldout):
        if d.exists():
            print(f"Removing old {d}")
            shutil.rmtree(d)
    (out / "qa").mkdir(parents=True)

    model = YOLO(args.model)
    stats = {"roboflow/train": Counter(), "roboflow/val": Counter(), "roboflow/test": Counter(), "mergeRM": Counter()}
    qa = {"roboflow": [], "mergerm": []}
    with zipfile.ZipFile(Path(args.zip).expanduser()) as zf:
        repair_roboflow(model, zf, src, out, stats, qa)
    label_mergerm(model, Path(args.mergerm).expanduser(), out, heldout, stats, qa)

    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
         "names": dict(enumerate(CLASSES))}, sort_keys=False))
    for name, items in qa.items():
        save_grid(items, out / "qa" / f"{name}.jpg")
    zip_path, mergerm = Path(args.zip).expanduser().resolve(), Path(args.mergerm).expanduser().resolve()
    info = {
        "labelled_with": {"model": str(Path(args.model).resolve()), "model_md5": md5(args.model)},
        "sources": {"base_dataset": str(src), "zip": str(zip_path), "zip_md5": md5(zip_path),
                    "mergerm": str(mergerm), "mergerm_fingerprint": folder_fingerprint(mergerm)},
        "heldout_rule": {"matches": sorted(HELDOUT_MATCHES), "ds2_top_fraction": HELDOUT_DS2_FRACTION,
                         "ds2_buffer_ids": DS2_BUFFER},
        "stats": {name: dict(s) for name, s in stats.items()},
    }
    for folder in (out, heldout):
        write_record(folder, "dataset_info.yaml", {"dataset": folder.name, **info}, scripts=[__file__, HERE / "record.py"])
    for name, s in stats.items():
        print(f"{name:15s}", dict(s))
    print(f"\nDataset ready: {out / 'data.yaml'}\nHeld-out set: {heldout}\nQA grids: {out / 'qa'}")


if __name__ == "__main__":
    main()
