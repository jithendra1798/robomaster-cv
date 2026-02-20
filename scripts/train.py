"""
RoboMaster YOLO Training Script with MLflow Tracking

Usage:
    python scripts/train.py --model yolo11s.pt --config configs/hyperparams/yolo11s_tuned.yaml
    python scripts/train.py --model yolo11l.pt --config configs/hyperparams/yolo11l_tuned.yaml
    python scripts/train.py --model yolo11s.pt --epochs 100 --weight_decay 0.001 --box 6.22111

MLflow UI:
    mlflow ui --port 5000
    Then open http://localhost:5000
"""

import argparse
import os
import platform
import yaml
import mlflow
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_NAME = "robomaster-plate-detection"
DEFAULT_DATA_YAML = str(PROJECT_ROOT / "configs" / "data_mergeRM_v1.yaml")


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
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO with MLflow tracking")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Model variant (e.g., yolo11s.pt, yolo11l.pt)")
    parser.add_argument("--config", type=str, default=None, help="Path to hyperparameter config YAML")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name (auto-generated if not set)")
    parser.add_argument("--skip_test", action="store_true", help="Skip test set evaluation after training")
    parser.add_argument("--dataset_version", type=str, default="mergeRM_v1", help="Dataset version for tracking")
    return parser.parse_args()


def train(args):
    # ============================================================
    # Setup MLflow
    # ============================================================
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load config file if provided (overrides CLI args)
    if args.config:
        config = load_config(args.config)
        args.model = config.get("model", args.model)
        args.epochs = config.get("epochs", args.epochs)
        args.batch = config.get("batch", args.batch)
        args.imgsz = config.get("imgsz", args.imgsz)
        args.weight_decay = config.get("weight_decay", args.weight_decay)
        args.box = config.get("box", args.box)
        args.mosaic = config.get("mosaic", args.mosaic)

    # Auto-generate run name if not set
    if args.run_name is None:
        model_name = Path(args.model).stem  # e.g., "yolo11s"
        args.run_name = f"{model_name}_{args.epochs}ep"

    print(f"\n{'='*60}")
    print(f"  Training: {args.model}")
    print(f"  Run name: {args.run_name}")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    print(f"  weight_decay={args.weight_decay}, box={args.box}, mosaic={args.mosaic}")
    print(f"{'='*60}\n")

    # ============================================================
    # Start MLflow Run
    # ============================================================
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log parameters
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "weight_decay": args.weight_decay,
            "box": args.box,
            "mosaic": args.mosaic,
            "dataset_version": args.dataset_version,
            "data_yaml": args.data,
        })

        # Log hardware info
        device_info = get_device_info()
        mlflow.log_params({f"hw_{k}": v for k, v in device_info.items()})

        # Log config file as artifact if used
        if args.config:
            mlflow.log_artifact(args.config, artifact_path="configs")

        # ============================================================
        # Train
        # ============================================================
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            weight_decay=args.weight_decay,
            box=args.box,
            mosaic=args.mosaic,
            name=args.run_name,
            project=str(PROJECT_ROOT / "runs" / "detect"),
        )

        # ============================================================
        # Log Validation Metrics (from training)
        # ============================================================
        val_metrics = {
            "val/precision": results.results_dict.get("metrics/precision(B)", 0),
            "val/recall": results.results_dict.get("metrics/recall(B)", 0),
            "val/mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
            "val/mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        }
        mlflow.log_metrics(val_metrics)
        print(f"\nValidation: P={val_metrics['val/precision']:.3f}, "
              f"R={val_metrics['val/recall']:.3f}, "
              f"mAP50={val_metrics['val/mAP50']:.3f}, "
              f"mAP50-95={val_metrics['val/mAP50-95']:.3f}")

        # ============================================================
        # Test Set Evaluation
        # ============================================================
        if not args.skip_test:
            print("\nRunning test set evaluation...")
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            best_model = YOLO(str(best_model_path))
            test_results = best_model.val(split="test")

            test_metrics = {
                "test/precision": test_results.results_dict.get("metrics/precision(B)", 0),
                "test/recall": test_results.results_dict.get("metrics/recall(B)", 0),
                "test/mAP50": test_results.results_dict.get("metrics/mAP50(B)", 0),
                "test/mAP50-95": test_results.results_dict.get("metrics/mAP50-95(B)", 0),
            }

            # Log speed metrics
            speed = test_results.speed
            test_metrics["test/inference_ms"] = speed.get("inference", 0)
            test_metrics["test/preprocess_ms"] = speed.get("preprocess", 0)
            test_metrics["test/postprocess_ms"] = speed.get("postprocess", 0)

            mlflow.log_metrics(test_metrics)
            print(f"\nTest: P={test_metrics['test/precision']:.3f}, "
                  f"R={test_metrics['test/recall']:.3f}, "
                  f"mAP50={test_metrics['test/mAP50']:.3f}, "
                  f"mAP50-95={test_metrics['test/mAP50-95']:.3f}, "
                  f"Inference={test_metrics['test/inference_ms']:.1f}ms")

        # Log model info
        model_info = model.info(verbose=False)
        if model_info:
            mlflow.log_params({
                "model_parameters": model.model.yaml.get("parameters", "unknown") if hasattr(model.model, 'yaml') else "unknown",
            })

        # Log best weights path (not the file itself - too heavy)
        mlflow.set_tag("best_weights", str(best_model_path) if not args.skip_test else "N/A")
        mlflow.set_tag("training_machine", device_info.get("machine", "unknown"))

        print(f"\n{'='*60}")
        print(f"  MLflow Run ID: {run.info.run_id}")
        print(f"  View results: mlflow ui --port 5000")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
