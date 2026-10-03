"""
Compare robot models on data no model was trained on, plus the Roboflow test split.

Held-out mergeRM (whole North American matches + a block of robot-camera footage, built by
pseudo_label.py) has armor-plate labels only, so it is scored with plate-based proxies at the
Orin pipeline's confidence threshold (0.65):
    plate coverage   % of labelled plates whose centre is inside a predicted robot box (recall proxy)
    colour agree     of covered plates that are clearly lit, % covered by a box of the same colour
    plate-less dets  detections with no plate inside, per image (false-alarm proxy; robots
                     facing away from the camera also count, so compare models, not absolutes)
    bg dets          detections per plate-free image
Roboflow test is scored with normal metrics, on its original labels and on the repaired labels.

Usage (on the server, from robot-detection/):
    python eval_heldout.py runs/robots_yolo11s/weights/best.pt runs/robots_v2_robust/weights/best.pt
"""

import argparse
import re
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from pseudo_label import image_files, inside, plate_colour, poly_box

HERE = Path(__file__).resolve().parent
CONF = 0.65


def heldout_proxies(model, root):
    """Plate-based proxies, per domain (North American matches vs robot-camera footage)."""
    paths = image_files(root / "images")
    acc = {}
    for i in range(0, len(paths), 64):
        batch = paths[i:i + 64]
        for path, r in zip(batch, model.predict([str(p) for p in batch], conf=CONF, verbose=False)):
            dets = [(int(c), b.tolist()) for c, b in zip(r.boxes.cls, r.boxes.xyxy)]
            domain = "NA matches" if path.name.startswith("ds1_") else "robot camera"
            a = acc.setdefault(domain, dict(images=0, plates=0, covered=0, lit=0, agree=0, plateless=0, bg_images=0, bg_dets=0))
            a["images"] += 1
            lab = root / "labels" / f"{path.stem}.txt"
            lines = [l.split() for l in lab.read_text().splitlines() if l.split()] if lab.exists() else []
            if not lines:
                a["bg_images"] += 1
                a["bg_dets"] += len(dets)
                continue
            img = cv2.imread(str(path))
            plates = [poly_box(v[1:]) for v in lines]
            for box, centre in plates:
                a["plates"] += 1
                covering = [c for c, b in dets if inside(centre, b)]
                if covering:
                    a["covered"] += 1
                    colour = plate_colour(img, box)
                    if colour is not None:
                        a["lit"] += 1
                        a["agree"] += colour in covering
            a["plateless"] += sum(not any(inside(centre, b) for _, centre in plates) for _, b in dets)
    rows = {}
    for domain, a in sorted(acc.items()):
        rows[domain] = (f"{100 * a['covered'] / max(a['plates'], 1):.1f}%",
                        f"{100 * a['agree'] / max(a['lit'], 1):.1f}%",
                        f"{a['plateless'] / max(a['images'] - a['bg_images'], 1):.2f}",
                        f"{a['bg_dets'] / max(a['bg_images'], 1):.2f}")
    return rows


SWEEP = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def heldout_sweep(model, root):
    """Plate coverage and plate-less detections per image at several confidence thresholds.

    Lets models be compared at the same false-alarm level, and shows which threshold to run them at.
    One prediction pass at the lowest threshold; higher thresholds keep a subset of its boxes.
    """
    acc = {}
    paths = image_files(root / "images")
    for i in range(0, len(paths), 64):
        batch = paths[i:i + 64]
        for path, r in zip(batch, model.predict([str(p) for p in batch], conf=min(SWEEP), verbose=False)):
            lab = root / "labels" / f"{path.stem}.txt"
            lines = [l.split() for l in lab.read_text().splitlines() if l.split()] if lab.exists() else []
            if not lines:
                continue
            centres = [poly_box(v[1:])[1] for v in lines]
            dets = [(float(s), b.tolist()) for s, b in zip(r.boxes.conf, r.boxes.xyxy)]
            domain = "NA matches" if path.name.startswith("ds1_") else "robot camera"
            for t in SWEEP:
                kept = [b for s, b in dets if s >= t]
                a = acc.setdefault((domain, t), [0, 0, 0, 0])  # plates, covered, plate-less dets, images
                a[0] += len(centres)
                a[1] += sum(any(inside(c, b) for b in kept) for c in centres)
                a[2] += sum(not any(inside(c, b) for c in centres) for b in kept)
                a[3] += 1
    for domain in ("NA matches", "robot camera"):
        cells = [f"{t:.2f}: {100 * acc[(domain, t)][1] / acc[(domain, t)][0]:4.1f}% / {acc[(domain, t)][2] / acc[(domain, t)][3]:.2f}"
                 for t in SWEEP if (domain, t) in acc]
        print(f"  {domain:12s} conf: coverage / plate-less per img  |  " + "  |  ".join(cells))


def roboflow_test(model, data_yaml):
    """mAP on the Roboflow test split, original and repaired labels."""
    out = {}
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    for name, split in [("original", "images/test"), ("repaired", "images/test_repaired")]:
        tmp = HERE / "runs" / f".eval_{name}.yaml"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_text(yaml.safe_dump({**cfg, "test": split}, sort_keys=False))
        m = model.val(data=str(tmp), split="test", conf=0.001, plots=False, verbose=False,
                      project=str(HERE / "runs" / ".eval"), name=name, exist_ok=True)
        out[name] = f"P {m.box.mp:.3f} R {m.box.mr:.3f} mAP50 {m.box.map50:.3f} mAP50-95 {m.box.map:.3f}"
        tmp.unlink()
    return out


def main():
    p = argparse.ArgumentParser(description="Compare robot models on held-out data")
    p.add_argument("models", nargs="+", help="model .pt files")
    p.add_argument("--heldout", default=str(HERE / "data" / "heldout"))
    p.add_argument("--data", default=str(HERE / "data" / "robots_v2" / "data.yaml"))
    p.add_argument("--sweep", action="store_true",
                   help="only print held-out coverage vs plate-less detections over confidence thresholds")
    args = p.parse_args()

    if args.sweep:
        for path in args.models:
            print(f"## {re.sub(r'.*/runs/', '', str(Path(path).resolve()))}")
            heldout_sweep(YOLO(path), Path(args.heldout))
        return
    print(f"Held-out mergeRM proxies at conf {CONF}: plate coverage | colour agree | plate-less dets/img | dets per background img\n")
    for path in args.models:
        model = YOLO(path)
        name = re.sub(r".*/runs/", "", str(Path(path).resolve()))
        print(f"## {name}")
        for domain, row in heldout_proxies(model, Path(args.heldout)).items():
            print(f"  held-out {domain:12s}  " + " | ".join(row))
        for split, row in roboflow_test(model, args.data).items():
            print(f"  roboflow test ({split:8s})  {row}")
        print()


if __name__ == "__main__":
    main()
