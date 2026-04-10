"""
RoboMaster YOLO Evaluation Script

Evaluate trained models on test/val set, log to MLflow, or compare models.

Usage:
    # Evaluate a single model (logs to MLflow)
    python scripts/evaluate.py --weights runs/obb/yolo11s-obb_200ep/weights/best.pt

    # Compare two models side by side
    python scripts/evaluate.py --compare \
        runs/obb/yolo11s-obb_200ep/weights/best.pt \
        /path/to/benchmark/best.pt \
        --labels "ours" "benchmark"

    # Evaluate on validation split instead
    python scripts/evaluate.py --weights best.pt --split val
"""

import argparse
import os
import mlflow
from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = str(PROJECT_ROOT / "configs" / "data.yaml")
EXPERIMENT_NAME = "robomaster-plate-detection"


def evaluate_model(weights_path, data_yaml, split="test", device=None):
    """Evaluate a model and return metrics dict."""
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = YOLO(weights_path)
    results = model.val(data=data_yaml, split=split, device=device)

    # Works for both detect (B) and OBB (B) metrics
    metrics = {
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "inference_ms": results.speed.get("inference", 0),
        "preprocess_ms": results.speed.get("preprocess", 0),
        "postprocess_ms": results.speed.get("postprocess", 0),
    }

    # Try to get model size
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        metrics["parameters"] = n_params
        metrics["size_mb"] = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1e6
    except Exception:
        pass

    return metrics


def log_evaluation_to_mlflow(name, weights_path, metrics, split, data_yaml):
    """Log an evaluation-only run to MLflow."""
    os.chdir(PROJECT_ROOT)
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"eval_{name}_{split}"):
        mlflow.set_tags({
            "stage": "evaluation",
            "model_variant": name,
            "eval_split": split,
        })
        mlflow.log_param("weights_path", str(weights_path))
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_param("split", split)

        prefixed = {f"{split}/{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed)

    print(f"  Logged to MLflow: eval_{name}_{split}")


def print_metrics(name, metrics):
    """Pretty print metrics for a model."""
    print(f"\n  {name}:")
    print(f"    Precision:   {metrics['precision']:.1%}")
    print(f"    Recall:      {metrics['recall']:.1%}")
    print(f"    mAP50:       {metrics['mAP50']:.1%}")
    print(f"    mAP50-95:    {metrics['mAP50-95']:.1%}")
    print(f"    Inference:   {metrics['inference_ms']:.1f}ms")
    if "parameters" in metrics:
        print(f"    Parameters:  {metrics['parameters']:,}")
        print(f"    Size:        {metrics['size_mb']:.1f} MB")


def compare_models(weights_list, labels, data_yaml, split="test"):
    """Evaluate multiple models and print comparison."""
    all_metrics = {}
    for w, label in zip(weights_list, labels):
        print(f"\nEvaluating: {label} ({w})")
        all_metrics[label] = evaluate_model(w, data_yaml, split)

    names = list(all_metrics.keys())
    metrics_keys = ["precision", "recall", "mAP50", "mAP50-95", "inference_ms"]
    row_labels = ["Precision", "Recall", "mAP50", "mAP50-95", "Inference (ms)"]

    # Add model size if available
    if "parameters" in all_metrics[names[0]]:
        metrics_keys.append("parameters")
        row_labels.append("Parameters")

    print(f"\n{'='*70}")
    print(f"  Model Comparison ({split} set)")
    print(f"{'='*70}")

    header = f"  {'Metric':<18}"
    for name in names:
        header += f" {name:>18}"
    if len(names) == 2:
        header += f" {'Delta':>10}"
    print(header)
    print(f"  {'-'*18}" + f" {'-'*18}" * len(names) + (" " + "-" * 10 if len(names) == 2 else ""))

    for key, label in zip(metrics_keys, row_labels):
        row = f"  {label:<18}"
        values = []
        for name in names:
            val = all_metrics[name].get(key, 0)
            values.append(val)
            if key == "inference_ms":
                row += f" {val:>17.1f}ms"
            elif key == "parameters":
                row += f" {val:>17,}"
            else:
                row += f" {val:>17.1%}"
        if len(names) == 2:
            delta = values[1] - values[0]
            if key == "inference_ms":
                row += f" {delta:>+9.1f}ms"
            elif key == "parameters":
                row += f" {delta:>+10,}"
            else:
                row += f" {delta:>+9.1%}"
        print(row)

    print(f"{'='*70}\n")
    return all_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO models")
    p.add_argument("--weights", type=str, help="Path to model weights")
    p.add_argument("--compare", nargs="+", help="Compare multiple model weights")
    p.add_argument("--labels", nargs="+", help="Labels for compared models (same order as --compare)")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_YAML)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--log_mlflow", action="store_true", help="Log results to MLflow")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.compare:
        labels = args.labels if args.labels else [
            Path(w).parent.parent.name for w in args.compare
        ]
        all_metrics = compare_models(args.compare, labels, args.data, args.split)

        if args.log_mlflow:
            for label, metrics in all_metrics.items():
                log_evaluation_to_mlflow(label, label, metrics, args.split, args.data)

    elif args.weights:
        name = Path(args.weights).parent.parent.name
        metrics = evaluate_model(args.weights, args.data, args.split)
        print_metrics(name, metrics)

        if args.log_mlflow:
            log_evaluation_to_mlflow(name, args.weights, metrics, args.split, args.data)
    else:
        print("Provide --weights or --compare. See --help.")