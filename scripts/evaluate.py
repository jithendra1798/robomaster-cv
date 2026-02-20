"""
RoboMaster YOLO Evaluation Script

Evaluate a trained model on test/val set and log results to an existing MLflow run,
or create a new evaluation-only run.

Usage:
    # Evaluate and log to existing run
    python scripts/evaluate.py --weights runs/detect/yolo11s_100ep/weights/best.pt --split test

    # Compare two models side by side (prints comparison table)
    python scripts/evaluate.py --compare runs/detect/yolo11s_100ep/weights/best.pt runs/detect/yolo11l_100ep/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_YAML = str(PROJECT_ROOT / "configs" / "data_mergeRM_v1.yaml")


def evaluate_model(weights_path, data_yaml, split="test"):
    """Evaluate a model and return metrics dict."""
    model = YOLO(weights_path)
    results = model.val(data=data_yaml, split=split)

    metrics = {
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "inference_ms": results.speed.get("inference", 0),
        "preprocess_ms": results.speed.get("preprocess", 0),
        "postprocess_ms": results.speed.get("postprocess", 0),
    }
    return metrics


def print_metrics(name, metrics):
    """Pretty print metrics for a model."""
    print(f"\n  {name}:")
    print(f"    Precision:  {metrics['precision']:.1%}")
    print(f"    Recall:     {metrics['recall']:.1%}")
    print(f"    mAP50:      {metrics['mAP50']:.1%}")
    print(f"    mAP50-95:   {metrics['mAP50-95']:.1%}")
    print(f"    Inference:  {metrics['inference_ms']:.1f}ms")


def compare_models(weights_list, data_yaml, split="test"):
    """Evaluate multiple models and print comparison."""
    all_metrics = {}
    for w in weights_list:
        name = Path(w).parent.parent.name  # gets the run name from .../run_name/weights/best.pt
        print(f"\nEvaluating: {name} ({w})")
        all_metrics[name] = evaluate_model(w, data_yaml, split)

    # Print comparison table
    names = list(all_metrics.keys())
    metrics_keys = ["precision", "recall", "mAP50", "mAP50-95", "inference_ms"]
    labels = ["Precision", "Recall", "mAP50", "mAP50-95", "Inference (ms)"]

    print(f"\n{'='*70}")
    print(f"  Model Comparison ({split} set)")
    print(f"{'='*70}")

    # Header
    header = f"  {'Metric':<18}"
    for name in names:
        header += f" {name:>18}"
    if len(names) == 2:
        header += f" {'Delta':>10}"
    print(header)
    print(f"  {'-'*18}" + f" {'-'*18}" * len(names) + (" " + "-"*10 if len(names) == 2 else ""))

    # Rows
    for key, label in zip(metrics_keys, labels):
        row = f"  {label:<18}"
        values = []
        for name in names:
            val = all_metrics[name][key]
            values.append(val)
            if key == "inference_ms":
                row += f" {val:>17.1f}ms"
            else:
                row += f" {val:>17.1%}"
        if len(names) == 2:
            delta = values[1] - values[0]
            if key == "inference_ms":
                row += f" {delta:>+9.1f}ms"
            else:
                row += f" {delta:>+9.1%}"
        print(row)

    print(f"{'='*70}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--compare", nargs="+", help="Compare multiple model weights")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        compare_models(args.compare, args.data, args.split)
    elif args.weights:
        metrics = evaluate_model(args.weights, args.data, args.split)
        print_metrics(Path(args.weights).parent.parent.name, metrics)
    else:
        print("Provide --weights or --compare. See --help.")
