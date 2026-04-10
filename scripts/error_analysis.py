"""
RoboMaster OBB Error Analysis Pipeline

Runs model on the test set, computes geometric OBB IoU (Shapely polygon
intersection), and categorizes every prediction as TP/FP/FN. Saves annotated
images and logs summary to MLflow.

Features:
    - Progress bar with ETA
    - Timing and throughput logging
    - Graceful interruption (saves partial results)
    - FP confidence distribution stats

Usage:
    python scripts/error_analysis.py --weights runs/obb/best.pt
    python scripts/error_analysis.py --weights best.pt --log_mlflow
    python scripts/error_analysis.py --weights best.pt --conf 0.25 --iou 0.5
"""

import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO

try:
    from shapely.geometry import Polygon
except ImportError:
    print("Error: shapely required. Install with: pip install shapely")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = str(PROJECT_ROOT / "configs" / "data.yaml")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "error_analysis")
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging(name="error_analysis"):
    """Configure logging to console and file."""
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{name}_{timestamp}.log"

    logger = logging.getLogger("error_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s"))
    logger.addHandler(fh)

    return logger, log_file


def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


# ============================================================
# Geometry helpers
# ============================================================

def obb_iou(box1, box2):
    """Compute IoU between two OBB boxes using Shapely polygon intersection."""
    try:
        p1 = Polygon(box1)
        p2 = Polygon(box2)
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def get_gt_boxes(label_path, img_w, img_h):
    """Load OBB ground truth. Returns list of (4,2) pixel-coord arrays."""
    boxes = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                coords = np.array([float(x) for x in parts[1:9]]).reshape(4, 2)
                coords[:, 0] *= img_w
                coords[:, 1] *= img_h
                boxes.append(coords)
    return boxes


def draw_obb_boxes(img, boxes, color, label_prefix, confidences=None):
    """Draw oriented bounding boxes on image."""
    for i, box in enumerate(boxes):
        pts = box.astype(np.int32)
        cv2.polylines(img, [pts], True, color, 2)
        label = label_prefix
        if confidences is not None and i < len(confidences):
            label = f"{label_prefix}:{confidences[i]:.2f}"
        cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


# ============================================================
# Main analysis
# ============================================================

def run_error_analysis(weights_path, data_yaml, output_dir, conf_thresh=0.25,
                       iou_thresh=0.5, device=None):
    """Run full error analysis pipeline. Returns summary dict."""
    import yaml

    log, log_file = setup_logging()

    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    # Parse data yaml
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_cfg["path"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    img_dir = data_root / data_cfg["test"] if "test" in data_cfg else data_root / "images/test"
    lbl_dir = str(img_dir).replace("images", "labels")

    log.info(f"Weights: {weights_path}")
    log.info(f"Images:  {img_dir}")
    log.info(f"Labels:  {lbl_dir}")
    log.info(f"Conf threshold: {conf_thresh}, IoU threshold: {iou_thresh}")

    for folder in ["false_positives", "false_negatives", "true_positives"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    model = YOLO(weights_path)

    # Counters
    tp_count = 0
    fp_count = 0
    fn_count = 0
    fp_images = []
    fn_images = []
    total_images = 0
    fp_confidences = []

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    total_files = len(image_files)
    log.info(f"Processing {total_files} test images...")
    start_time = time.time()

    try:
        for idx, img_name in enumerate(image_files):
            # Progress every 50 images or at key milestones
            if (idx + 1) % 50 == 0 or (idx + 1) == total_files:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_files - idx - 1) / rate if rate > 0 else 0
                log.info(f"  [{idx+1}/{total_files}] {rate:.1f} img/s, "
                         f"ETA: {format_duration(eta)}, "
                         f"TP={tp_count} FP={fp_count} FN={fn_count}")

            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(lbl_dir, Path(img_name).stem + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                log.warning(f"  Could not read: {img_name}")
                continue
            total_images += 1
            h, w = img.shape[:2]

            gt_boxes = get_gt_boxes(label_path, w, h)
            results = model(img, conf=conf_thresh, device=device, verbose=False)

            # Extract OBB predictions
            pred_boxes = []
            pred_confs = []
            if results and len(results) > 0:
                res = results[0]
                if hasattr(res, "obb") and res.obb is not None and res.obb.xyxyxyxy is not None:
                    for i in range(len(res.obb)):
                        box = res.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                        conf = float(res.obb.conf[i].cpu())
                        pred_boxes.append(box)
                        pred_confs.append(conf)

            # Greedy IoU matching
            gt_matched = [False] * len(gt_boxes)
            pred_matched = [False] * len(pred_boxes)

            for pi, pbox in enumerate(pred_boxes):
                best_iou = 0.0
                best_gi = -1
                for gi, gbox in enumerate(gt_boxes):
                    if gt_matched[gi]:
                        continue
                    iou = obb_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= iou_thresh and best_gi >= 0:
                    gt_matched[best_gi] = True
                    pred_matched[pi] = True
                    tp_count += 1

            img_fps = [pred_boxes[i] for i in range(len(pred_boxes)) if not pred_matched[i]]
            fp_confs = [pred_confs[i] for i in range(len(pred_confs)) if not pred_matched[i]]
            fp_count += len(img_fps)
            fp_confidences.extend(fp_confs)

            img_fns = [gt_boxes[i] for i in range(len(gt_boxes)) if not gt_matched[i]]
            fn_count += len(img_fns)

            # Save annotated FP images
            if len(img_fps) > 0:
                vis = img.copy()
                vis = draw_obb_boxes(vis, gt_boxes, (0, 255, 0), "GT")
                vis = draw_obb_boxes(vis, img_fps, (0, 0, 255), "FP", fp_confs)
                cv2.imwrite(os.path.join(output_dir, "false_positives", img_name), vis)
                fp_images.append(img_name)

            # Save annotated FN images
            if len(img_fns) > 0:
                vis = img.copy()
                matched_gt = [gt_boxes[i] for i in range(len(gt_boxes)) if gt_matched[i]]
                vis = draw_obb_boxes(vis, matched_gt, (0, 255, 0), "TP")
                vis = draw_obb_boxes(vis, img_fns, (0, 165, 255), "MISSED")
                for i, pbox in enumerate(pred_boxes):
                    color = (255, 0, 0) if pred_matched[i] else (0, 0, 255)
                    label = f"P:{pred_confs[i]:.2f}" if pred_matched[i] else f"FP:{pred_confs[i]:.2f}"
                    pts = pbox.astype(np.int32)
                    cv2.polylines(vis, [pts], True, color, 2)
                    cv2.putText(vis, label, (pts[0][0], pts[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imwrite(os.path.join(output_dir, "false_negatives", img_name), vis)
                fn_images.append(img_name)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        log.warning(f"INTERRUPTED after {idx+1}/{total_files} images ({format_duration(elapsed)})")
        log.warning("Partial results saved below.")

    total_elapsed = time.time() - start_time
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

    summary = {
        "total_images": total_images,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "fp_images": len(fp_images),
        "fn_images": len(fn_images),
        "precision_geometric": precision,
        "recall_geometric": recall,
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "mean_fp_confidence": float(np.mean(fp_confidences)) if fp_confidences else 0,
        "median_fp_confidence": float(np.median(fp_confidences)) if fp_confidences else 0,
        "elapsed_seconds": total_elapsed,
        "images_per_second": total_images / total_elapsed if total_elapsed > 0 else 0,
    }

    log.info(f"{'='*50}")
    log.info(f"ERROR ANALYSIS SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"Images processed: {total_images}/{total_files}")
    log.info(f"Time: {format_duration(total_elapsed)} ({summary['images_per_second']:.1f} img/s)")
    log.info(f"True Positives:   {tp_count}")
    log.info(f"False Positives:  {fp_count} ({len(fp_images)} images)")
    log.info(f"False Negatives:  {fn_count} ({len(fn_images)} images)")
    log.info(f"Precision (geom): {precision:.4f}")
    log.info(f"Recall (geom):    {recall:.4f}")
    if fp_confidences:
        log.info(f"Mean FP conf:     {summary['mean_fp_confidence']:.3f}")
        log.info(f"Median FP conf:   {summary['median_fp_confidence']:.3f}")
    log.info(f"Output: {output_dir}")
    log.info(f"Log: {log_file}")

    return summary, log_file


def log_to_mlflow(summary, weights_path, log_file):
    """Log error analysis results to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("robomaster-plate-detection")

    model_name = Path(weights_path).parent.parent.name

    with mlflow.start_run(run_name=f"error_analysis_{model_name}"):
        mlflow.set_tags({
            "stage": "error_analysis",
            "model_variant": model_name,
            "iou_method": "shapely_geometric",
            "status": "completed",
        })
        mlflow.log_param("weights_path", str(weights_path))
        mlflow.log_param("conf_thresh", summary["conf_thresh"])
        mlflow.log_param("iou_thresh", summary["iou_thresh"])

        mlflow.log_metrics({
            "ea/true_positives": summary["true_positives"],
            "ea/false_positives": summary["false_positives"],
            "ea/false_negatives": summary["false_negatives"],
            "ea/fp_images": summary["fp_images"],
            "ea/fn_images": summary["fn_images"],
            "ea/precision_geometric": summary["precision_geometric"],
            "ea/recall_geometric": summary["recall_geometric"],
            "ea/mean_fp_confidence": summary["mean_fp_confidence"],
            "ea/median_fp_confidence": summary["median_fp_confidence"],
            "timing/total_seconds": summary["elapsed_seconds"],
            "timing/images_per_second": summary["images_per_second"],
        })

        if log_file and Path(log_file).exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")

    print(f"\nLogged to MLflow as: error_analysis_{model_name}")


def parse_args():
    p = argparse.ArgumentParser(description="OBB Error Analysis")
    p.add_argument("--weights", type=str, required=True, help="Path to model weights")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Data YAML")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output dir")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU matching threshold")
    p.add_argument("--log_mlflow", action="store_true", help="Log results to MLflow")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    summary, log_file = run_error_analysis(
        weights_path=args.weights,
        data_yaml=args.data,
        output_dir=args.output,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    if args.log_mlflow:
        log_to_mlflow(summary, args.weights, log_file)