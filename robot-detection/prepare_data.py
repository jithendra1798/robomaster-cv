"""
Build the robot-detection dataset from the Roboflow export (robomaster.v1i.yolov8.zip).

What it does:
    - Keeps whole-robot boxes and merges them into 3 classes:
          car_blue,   watcher_blue   -> 0 robot_blue
          car_red,    watcher_red    -> 1 robot_red
          car_unknow, watcher_unknow -> 2 robot_unknown  (dead / lights off / colour not visible)
      ("watcher" is the old rail-mounted sentry.)
    - Drops the armor_* boxes. Images left with no robot (people holding plates in a lab)
      stay in as background images.
    - Converts Roboflow's 5-point polygons to normal YOLO boxes and removes the
      duplicated rows in the export.
    - Writes data/robots/{images,labels}/{train,val,test}/ and data/robots/data.yaml,
      rebuilding the folder from the zip on every run.

Usage (from robot-detection/):
    python prepare_data.py --zip ~/jithendra/robomaster.v1i.yolov8.zip
"""

import argparse
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CLASSES = ["robot_blue", "robot_red", "robot_unknown"]
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # Roboflow folder -> YOLO split


def robot_class(name):
    """Roboflow class name -> our class id, or None to drop it (armor)."""
    kind, colour = name.split("_", 1)
    if kind not in ("car", "watcher"):
        return None
    return {"blue": 0, "red": 1}.get(colour, 2)


def convert(text, mapping):
    """Roboflow label rows -> sorted, de-duplicated 'cls xc yc w h' rows for robots only."""
    rows = set()
    for line in text.splitlines():
        v = line.split()
        if not v or mapping[int(v[0])] is None:
            continue
        vals = [float(x) for x in v[1:]]
        if len(vals) == 4:  # already a YOLO box
            xc, yc, w, h = vals
        else:  # polygon x1 y1 x2 y2 ... -> enclosing box
            xs = [min(max(x, 0.0), 1.0) for x in vals[0::2]]
            ys = [min(max(y, 0.0), 1.0) for y in vals[1::2]]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > 0 and h > 0:
            rows.add(f"{mapping[int(v[0])]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return sorted(rows)


def main():
    p = argparse.ArgumentParser(description="Roboflow export -> robot-only YOLO dataset")
    p.add_argument("--zip", required=True, help="path to robomaster.v1i.yolov8.zip")
    p.add_argument("--out", default=str(HERE / "data" / "robots"), help="output dataset folder")
    args = p.parse_args()
    out = Path(args.out).expanduser().resolve()

    if out.exists():
        print(f"Removing old {out}")
        shutil.rmtree(out)

    stats = {split: Counter() for split in SPLITS.values()}
    print(f"Reading {args.zip} ...")
    with zipfile.ZipFile(Path(args.zip).expanduser()) as zf:
        names = yaml.safe_load(zf.read("data.yaml"))["names"]
        mapping = {i: robot_class(n) for i, n in enumerate(names)}
        print("Class mapping:", {n: CLASSES[c] if c is not None else "dropped" for n, c in zip(names, mapping.values())})

        for member in zf.namelist():
            parts = member.split("/")  # e.g. ["train", "images", "x.jpg"]
            if len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in ("images", "labels") or not parts[2]:
                continue
            split, kind, fname = SPLITS[parts[0]], parts[1], parts[2]
            dst = out / kind / split / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if kind == "images":
                dst.write_bytes(zf.read(member))
                stats[split]["images"] += 1
            else:
                rows = convert(zf.read(member).decode(), mapping)
                dst.write_text("".join(r + "\n" for r in rows))
                stats[split]["background"] += not rows
                stats[split].update(CLASSES[int(r.split()[0])] for r in rows)

    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
         "names": dict(enumerate(CLASSES))},
        sort_keys=False,
    ))

    cols = ["images", "background", *CLASSES]
    print(f"\n{'split':6s}" + "".join(f"{c:>15s}" for c in cols))
    for split, s in stats.items():
        print(f"{split:6s}" + "".join(f"{s[c]:>15d}" for c in cols))
    print(f"\nDataset ready: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
