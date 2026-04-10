"""
RoboMaster YOLO Training Script with MLflow Tracking

Supports both standard detection and OBB (oriented bounding box) tasks.
Logs comprehensive experiment metadata for tracking model improvements.

Features:
    - Structured logging with timestamps to console and log file
    - Total training time logged to MLflow
    - Graceful interruption handling (Ctrl+C logs partial results)
    - Generalization gap tracking (val-to-test metric drop)

Usage:
    python scripts/train.py --model yolo11s-obb.pt --epochs 200
    python scripts/train.py --config configs/hyperparams/yolo11s_obb.yaml
    python scripts/train.py --model yolo11s.pt --task detect --epochs 100
"""

import argparse
import logging
import os
import platform
import sys
import time
import yaml
import mlflow
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "robomaster-plate-detection"
DEFAULT_DATA_YAML = str(PROJECT_ROOT / "configs" / "data.yaml")
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging(run_name):
    """Configure logging to both console and file with timestamps."""
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger, log_file


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info():
    """Get hardware info for logging."""
    import torch
    info = {
        "machine": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
    elif torch.backends.mps.is_available():
        info["device"] = "Apple MPS"
        info["apple_silicon"] = platform.processor()
    else:
        info["device"] = "CPU"
    return info


def load_config(config_path):
    """Load hyperparameters from YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_duration(seconds):
    """Format seconds into human readable string."""
    return str(timedelta(seconds=int(seconds)))


def compute_generalization_gap(val_metrics, test_metrics):
    """Compute val-to-test metric drops — key signal for deployment quality."""
    gap = {}
    for key in ["precision", "recall", "mAP50", "mAP50-95"]:
        vk = f"val/{key}"
        tk = f"test/{key}"
        if vk in val_metrics and tk in test_metrics:
            gap[f"gap/{key}"] = val_metrics[vk] - test_metrics[tk]
    return gap


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO with MLflow tracking")

    # Model & task
    p.add_argument("--model", type=str, default="yolo11s-obb.pt", help="Model weights")
    p.add_argument("--task", type=str, default="obb", choices=["detect", "obb"])
    p.add_argument("--config", type=str, default=None, help="Hyperparameter config YAML")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Data YAML path")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=640)

    # Optimizer — set explicitly to avoid auto-override!
    p.add_argument("--optimizer", type=str, default="AdamW",
                   help="Optimizer (AdamW/SGD/Adam). Set explicitly to avoid auto-override")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=0.0005)

    # Loss
    p.add_argument("--box", type=float, default=7.5)

    # Augmentation
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--degrees", type=float, default=15.0, help="Rotation augmentation")
    p.add_argument("--shear", type=float, default=2.0)
    p.add_argument("--mixup", type=float, default=0.1)
    p.add_argument("--perspective", type=float, default=0.0001)

    # Run metadata
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--dataset_version", type=str, default="mergeRM")
    p.add_argument("--notes", type=str, default="", help="Free-text notes for this run")

    return p.parse_args()


def train(args):
    os.chdir(PROJECT_ROOT)

    # Load config file if provided (overrides CLI args)
    if args.config:
        config = load_config(args.config)
        for key, val in config.items():
            if hasattr(args, key):
                setattr(args, key, val)

    # Auto-generate run name
    if args.run_name is None:
        model_name = Path(args.model).stem
        args.run_name = f"{model_name}_{args.epochs}ep"

    log, log_file = setup_logging(args.run_name)
    device = get_device()
    device_info = get_device_info()
    metric_suffix = "(B)"

    log.info(f"{'='*60}")
    log.info(f"  Task: {args.task}")
    log.info(f"  Model: {args.model}")
    log.info(f"  Run: {args.run_name}")
    log.info(f"  Device: {device} ({device_info.get('device', 'unknown')})")
    log.info(f"  Optimizer: {args.optimizer} (lr0={args.lr0}, momentum={args.momentum})")
    log.info(f"  Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    log.info(f"  Data: {args.data}")
    log.info(f"{'='*60}")

    # ============================================================
    # MLflow Run
    # ============================================================
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_start = time.time()
    interrupted = False

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow_run_id = run.info.run_id
        log.info(f"MLflow run ID: {mlflow_run_id}")

        # --- Tags ---
        mlflow.set_tags({
            "task": args.task,
            "stage": "training",
            "model_variant": Path(args.model).stem,
            "machine": device_info.get("machine", "unknown"),
            "dataset": args.dataset_version,
            "status": "running",
        })
        if args.notes:
            mlflow.set_tag("notes", args.notes)

        # --- Parameters ---
        mlflow.log_params({
            "model": args.model, "task": args.task,
            "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
            "optimizer": args.optimizer, "lr0": args.lr0,
            "momentum": args.momentum, "weight_decay": args.weight_decay,
            "box": args.box, "mosaic": args.mosaic, "degrees": args.degrees,
            "shear": args.shear, "mixup": args.mixup,
            "perspective": args.perspective,
            "dataset_version": args.dataset_version, "data_yaml": args.data,
        })
        mlflow.log_params({f"hw_{k}": v for k, v in device_info.items()})

        if args.config:
            mlflow.log_artifact(args.config, artifact_path="configs")

        try:
            # ========================================================
            # Train
            # ========================================================
            log.info("Starting training...")
            train_start = time.time()

            model = YOLO(args.model)
            results = model.train(
                data=args.data, task=args.task, device=device,
                epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                optimizer=args.optimizer, lr0=args.lr0,
                momentum=args.momentum, weight_decay=args.weight_decay,
                box=args.box, mosaic=args.mosaic, degrees=args.degrees,
                shear=args.shear, mixup=args.mixup, perspective=args.perspective,
                name=args.run_name,
                project=str(PROJECT_ROOT / "runs" / args.task),
            )

            train_elapsed = time.time() - train_start
            log.info(f"Training complete in {format_duration(train_elapsed)}")
            mlflow.log_metric("timing/train_seconds", train_elapsed)

            # ========================================================
            # Validation Metrics
            # ========================================================
            val_metrics = {
                "val/precision": results.results_dict.get(f"metrics/precision{metric_suffix}", 0),
                "val/recall": results.results_dict.get(f"metrics/recall{metric_suffix}", 0),
                "val/mAP50": results.results_dict.get(f"metrics/mAP50{metric_suffix}", 0),
                "val/mAP50-95": results.results_dict.get(f"metrics/mAP50-95{metric_suffix}", 0),
            }
            mlflow.log_metrics(val_metrics)
            log.info(f"Val: P={val_metrics['val/precision']:.3f} R={val_metrics['val/recall']:.3f} "
                     f"mAP50={val_metrics['val/mAP50']:.3f} mAP50-95={val_metrics['val/mAP50-95']:.3f}")

            # ========================================================
            # Test Set Evaluation
            # ========================================================
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"

            if not args.skip_test:
                log.info("Running test set evaluation...")
                test_start = time.time()

                best_model = YOLO(str(best_model_path))
                test_results = best_model.val(data=args.data, split="test", device=device)

                test_elapsed = time.time() - test_start
                log.info(f"Test evaluation complete in {format_duration(test_elapsed)}")
                mlflow.log_metric("timing/test_seconds", test_elapsed)

                test_metrics = {
                    "test/precision": test_results.results_dict.get(f"metrics/precision{metric_suffix}", 0),
                    "test/recall": test_results.results_dict.get(f"metrics/recall{metric_suffix}", 0),
                    "test/mAP50": test_results.results_dict.get(f"metrics/mAP50{metric_suffix}", 0),
                    "test/mAP50-95": test_results.results_dict.get(f"metrics/mAP50-95{metric_suffix}", 0),
                    "test/inference_ms": test_results.speed.get("inference", 0),
                    "test/preprocess_ms": test_results.speed.get("preprocess", 0),
                    "test/postprocess_ms": test_results.speed.get("postprocess", 0),
                }
                mlflow.log_metrics(test_metrics)

                gap = compute_generalization_gap(val_metrics, test_metrics)
                mlflow.log_metrics(gap)

                log.info(f"Test: P={test_metrics['test/precision']:.3f} R={test_metrics['test/recall']:.3f} "
                         f"mAP50={test_metrics['test/mAP50']:.3f} mAP50-95={test_metrics['test/mAP50-95']:.3f} "
                         f"Inference={test_metrics['test/inference_ms']:.1f}ms")
                if gap:
                    log.info(f"Gap: P={gap.get('gap/precision',0):+.3f} R={gap.get('gap/recall',0):+.3f}")

            # Model info
            try:
                n_params = sum(p.numel() for p in model.model.parameters())
                size_mb = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1e6
                mlflow.log_metrics({"model/parameters": n_params, "model/size_mb": size_mb})
            except Exception:
                pass

            mlflow.set_tag("best_weights", str(best_model_path))
            mlflow.set_tag("status", "completed")

        except KeyboardInterrupt:
            interrupted = True
            elapsed = time.time() - run_start
            log.warning(f"INTERRUPTED by user after {format_duration(elapsed)}")
            mlflow.set_tag("status", "interrupted")
            mlflow.log_metric("timing/interrupted_after_seconds", elapsed)

        except Exception as e:
            elapsed = time.time() - run_start
            log.error(f"FAILED after {format_duration(elapsed)}: {e}", exc_info=True)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e)[:250])
            mlflow.log_metric("timing/failed_after_seconds", elapsed)
            raise

        finally:
            total_elapsed = time.time() - run_start
            mlflow.log_metric("timing/total_seconds", total_elapsed)

            # Log the log file as artifact
            if log_file.exists():
                mlflow.log_artifact(str(log_file), artifact_path="logs")

            log.info(f"{'='*60}")
            log.info(f"  MLflow Run ID: {mlflow_run_id}")
            log.info(f"  Total time: {format_duration(total_elapsed)}")
            log.info(f"  Status: {'INTERRUPTED' if interrupted else 'completed'}")
            log.info(f"  Log file: {log_file}")
            log.info(f"  View: mlflow ui --port 5000")
            log.info(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)