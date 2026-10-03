"""
Train the robot detector (robot_blue / robot_red / robot_unknown) for sentry decision-making.

Run prepare_data.py once first, then on the GPU server (from robot-detection/):
    python train.py                                   # yolo11s, 100 epochs, batch 32
    python train.py --model yolo11n.pt --name robots_nano
    PYTHONPATH=~/jithendra/pylibs python train.py --data data/robots_v2/data.yaml --robust \
        --epochs 150 --patience 40 --save-period 25 --name robots_v2_robust

Outputs (gitignored):
    runs/<name>/weights/best.pt   -> copy this to the Orin (see GUIDE.md)
    runs/<name>/                  -> training curves, confusion matrix, val predictions
    runs/<name>/test/             -> final test-split evaluation
"""

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from record import write_record

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train the robot detector")
    p.add_argument("--model", default="yolo11s.pt", help="pretrained weights to start from")
    p.add_argument("--data", default=str(HERE / "data" / "robots" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="GPU index, or 'mps' / 'cpu'")
    p.add_argument("--name", default=None, help="run folder under runs/ (default: robots_<model>)")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of train images, for quick tests")
    p.add_argument("--patience", type=int, default=20, help="stop if val mAP hasn't improved for this many epochs")
    p.add_argument("--save-period", type=int, default=-1, help="also keep a checkpoint every N epochs")
    p.add_argument("--robust", action="store_true",
                   help="extra camera augmentations (needs albumentations on PYTHONPATH, see GUIDE.md)")
    p.add_argument("--force", action="store_true", help="start even if the shared GPU is busy")
    return p.parse_args()


def robust_augmentations():
    """Camera effects the training images lack. No hue shift or grayscale: colour is the class."""
    import albumentations as A

    return [
        A.MotionBlur(blur_limit=(3, 15), p=0.25),  # the sentry spins and its gimbal moves
        A.GaussNoise(std_range=(0.02, 0.08), p=0.15),  # sensor noise in dark arenas
        # Training images were squashed to 640x640 from wider frames; the Orin letterboxes instead.
        A.Affine(scale={"x": (1.0, 1.5), "y": (1.0, 1.0)}, p=0.3),
    ]


def main():
    args = parse_args()
    name = args.name or f"robots_{Path(args.model).stem}"

    # The 4090 is shared: if someone else already uses >10 GB, wait instead of crashing both jobs.
    if args.device not in ("cpu", "mps") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used_gb = (total - free) / 1e9
        if used_gb > 10 and not args.force:
            raise SystemExit(f"GPU already has {used_gb:.1f} GB in use. Check nvidia-smi and wait, or pass --force.")

    # Ultralytics auto-logs to MLflow when it's installed; keep those logs inside this folder.
    os.environ.setdefault("MLFLOW_TRACKING_URI", str(HERE / "runs" / "mlflow"))
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "robot-detection")

    # [] (not None) switches off Ultralytics' default albumentations set (incl. ToGray), so a plain run
    # trains the same whether or not albumentations happens to be installed.
    augmentations = robust_augmentations() if args.robust else []
    start = time.time()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        fraction=args.fraction,
        project=str(HERE / "runs"),
        name=name,
        patience=args.patience,
        save_period=args.save_period,
        augmentations=augmentations,
        # Set explicitly: optimizer="auto" silently overrides lr0 and momentum.
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        # Other augmentation stays at Ultralytics defaults; its hue jitter (hsv_h=0.015) is far too small to turn red into blue.
    )

    # Final score on the held-out test split with the best checkpoint (also prints per-class results).
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    metrics = YOLO(best).val(
        data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(save_dir), name="test",
    )
    print(f"\nTest: P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}  "
          f"mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    print(f"Best weights: {best}")

    # Reproducibility record next to the run (all hyperparameters are in args.yaml beside it).
    data_info = Path(args.data).resolve().parent / "dataset_info.yaml"
    dataset = yaml.safe_load(data_info.read_text()) if data_info.exists() else {}
    write_record(save_dir, "run_info.yaml", {
        "run": save_dir.name,
        "data": {"yaml": str(Path(args.data).resolve()),
                 "dataset": dataset.get("dataset", Path(args.data).resolve().parent.name),
                 "dataset_created": dataset.get("created"),
                 "dataset_code": dataset.get("code")},
        "hyperparameters": "args.yaml in this folder",
        "augmentations": [str(a) for a in augmentations] or "none beyond Ultralytics built-ins (see args.yaml)",
        "train_hours": round((time.time() - start) / 3600, 2),
        "test": {"precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4),
                 "mAP50": round(float(metrics.box.map50), 4), "mAP50-95": round(float(metrics.box.map), 4)},
    }, scripts=[__file__, HERE / "record.py"])


if __name__ == "__main__":
    main()
