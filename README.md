# RoboMaster CV — Armor Plate Detection

YOLO OBB (Oriented Bounding Box) training, evaluation, and error analysis for the NYU RoboMaster VIP team. Experiments tracked with MLflow for reproducible comparisons across models, hyperparameters, and machines.

## Quick Start

```bash
git clone https://github.com/jithendra1798/robomaster-cv.git
cd robomaster-cv
pip install -r requirements.txt
```

Add your dataset (see [Dataset Management](#dataset-management)), then:

```bash
# Train
python scripts/train.py --config configs/hyperparams/yolo11s_obb.yaml

# Evaluate
python scripts/evaluate.py --weights runs/obb/yolo11s-obb_200ep/weights/best.pt

# Error analysis
python scripts/error_analysis.py --weights runs/obb/yolo11s-obb_200ep/weights/best.pt --log_mlflow

# View all experiments
mlflow ui --port 5000
```

## Project Structure

```
robomaster-cv/
├── configs/
│   ├── data.yaml                  # Dataset config (relative paths)
│   ├── datasets.yaml              # Dataset version registry (documentation)
│   └── hyperparams/               # Saved hyperparameter configs
├── scripts/
│   ├── train.py                   # Training with MLflow (detect + OBB)
│   ├── evaluate.py                # Evaluation and model comparison
│   └── error_analysis.py          # OBB error analysis pipeline
├── notebooks/                     # Jupyter notebooks (exploration / history)
├── mlruns/                        # MLflow metadata (Git tracked)
├── logs/                          # Script log files (gitignored)
├── data/                          # Datasets (gitignored)
├── runs/                          # Training outputs (gitignored)
├── error_analysis/                # Annotated failure images (gitignored)
└── requirements.txt
```

**What's in Git:** scripts, configs, notebooks, mlruns/, README  
**What stays local:** data/, runs/, logs/, error_analysis/, model weights (.pt)

---

## Dataset Management

### Current dataset

The default dataset is `mergeRM` — configured in `configs/data.yaml` with relative paths that work on any machine.

**Setup on a new machine:**
```bash
# Option 1: Symlink (preferred — saves disk space)
ln -s /path/to/Datasets/mergeRM data/mergeRM

# Option 2: Copy
cp -r /path/to/Datasets/mergeRM data/mergeRM

# Required structure:
# data/mergeRM/images/{train,val,test}/
# data/mergeRM/labels/{train,val,test}/
```

### Adding a new dataset version

When the dataset changes (new images, re-annotations, different splits):

**Step 1.** Place the new dataset in `data/<new_name>/` with the same `images/` and `labels/` structure.

**Step 2.** Create a new data config:
```bash
cp configs/data.yaml configs/data_<new_name>.yaml
```
Edit the new file to point to `data/<new_name>`.

**Step 3.** Document it in `configs/datasets.yaml`:
```yaml
mergeRM_v2:
  description: "Added 500 close-range plate images"
  date_added: "2025-04-15"
  source: "NYU server /Datasets/mergeRM_v2"
  classes: ["plate"]
  nc: 1
  splits:
    train: { images: 6305, with_labels: 5336, backgrounds: 968 }
    val: { images: 909, with_labels: 742, backgrounds: 166 }
    test: { images: 706, with_labels: 573, backgrounds: 132 }
  data_yaml: "configs/data_mergeRM_v2.yaml"
  notes: "Augmented train set with close-range arena footage"
```

**Step 4.** Train with the new dataset:
```bash
python scripts/train.py --data configs/data_mergeRM_v2.yaml --dataset_version mergeRM_v2
```

The `--dataset_version` tag is logged to MLflow so you can filter and compare experiments across dataset versions.

**Nothing auto-updates** — dataset changes are intentional and manual so every experiment is reproducible.

---

## Scripts

### `scripts/train.py` — Training

```bash
# OBB training with config
python scripts/train.py --config configs/hyperparams/yolo11s_obb.yaml

# OBB training with CLI args
python scripts/train.py --model yolo11s-obb.pt --epochs 200 --optimizer AdamW

# Standard detection
python scripts/train.py --model yolo11s.pt --task detect --epochs 100

# Add notes for tracking
python scripts/train.py --model yolo11s-obb.pt --notes "Testing higher shear=5.0" --shear 5.0

# Skip test evaluation (faster iteration)
python scripts/train.py --model yolo11s-obb.pt --epochs 50 --skip_test
```

### `scripts/evaluate.py` — Evaluation & Comparison

```bash
# Evaluate single model
python scripts/evaluate.py --weights runs/obb/yolo11s-obb_200ep/weights/best.pt

# Compare two models
python scripts/evaluate.py --compare \
    runs/obb/yolo11s-obb_200ep/weights/best.pt \
    /path/to/benchmark/best.pt \
    --labels "ours" "benchmark"

# Log to MLflow
python scripts/evaluate.py --weights best.pt --log_mlflow
```

### `scripts/error_analysis.py` — Error Analysis

```bash
# Run analysis
python scripts/error_analysis.py --weights runs/obb/yolo11s-obb_200ep/weights/best.pt

# Custom thresholds
python scripts/error_analysis.py --weights best.pt --conf 0.3 --iou 0.4

# Log to MLflow
python scripts/error_analysis.py --weights best.pt --log_mlflow

# Review results
ls error_analysis/false_positives/   # Red boxes = FP, Green = GT
ls error_analysis/false_negatives/   # Orange = missed, Blue = matched
```

---

## Logging, Timing, and Interruption Handling

All scripts produce structured logs with timestamps, saved to the `logs/` directory.

**What gets logged:**
- Start/end timestamps for each phase (training, evaluation, error analysis)
- Total wall-clock time and per-phase duration
- Progress updates with throughput (images/second) and ETA
- Hardware info (GPU, VRAM, machine hostname)
- All parameters and results

**If interrupted (Ctrl+C):**
- Training: MLflow run is marked `status=interrupted` with the elapsed time logged. Ultralytics saves a `last.pt` checkpoint you can resume from. The log file captures everything up to the interruption point.
- Error analysis: Partial results are saved — annotated images processed so far are in `error_analysis/`, and the summary prints with whatever was completed.

**If it crashes:**
- MLflow run is marked `status=failed` with the error message logged as a tag.
- The full traceback is saved to the log file.
- Check `logs/` for the timestamped log file, or look at the MLflow run's `error` tag.

**Where to look:**
- Console: real-time progress
- `logs/<run_name>_<timestamp>.log`: full log file (also saved as MLflow artifact)
- MLflow UI → run → artifacts → logs/: same log file, browsable in the UI

---

## MLflow Workflow

### Syncing between machines

```bash
# ON SERVER (after training):
git add mlruns/ configs/
git commit -m "yolo11s-obb 200ep: P=93% R=90%"
git push

# ON MAC (to review):
git pull
mlflow ui --port 5000    # http://localhost:5000
```

### MLflow Tags (for filtering)

| Tag | Values | Use |
|-----|--------|-----|
| `task` | `detect`, `obb` | Filter by detection type |
| `stage` | `training`, `evaluation`, `error_analysis` | Filter by pipeline step |
| `model_variant` | `yolo11s-obb`, `yolo8n-obb`, etc. | Filter by architecture |
| `machine` | hostname | Filter by training machine |
| `dataset` | `mergeRM`, `mergeRM_v2`, etc. | Filter by dataset version |
| `status` | `completed`, `interrupted`, `failed`, `running` | Filter by outcome |
| `notes` | free text | Search by experiment notes |

### Key Metrics

| Metric | Why it matters |
|--------|---------------|
| `test/precision` | FPs cause erratic gimbal PID behavior |
| `gap/precision` | Val→test drop = generalization quality |
| `test/inference_ms` | Must be real-time on Jetson Orin Nano |
| `model/size_mb` | Jetson has 8GB shared memory |
| `ea/false_positives` | Direct count from geometric IoU analysis |
| `timing/train_seconds` | Compare training efficiency across machines |

### Comparing runs in the UI

1. Open MLflow UI → select experiment "robomaster-plate-detection"
2. Use the search bar to filter: `tags.task = "obb" AND tags.stage = "training"`
3. Select runs → click "Compare"
4. Sort by `test/precision` or `gap/precision` to find best models

---

## Notebooks vs Scripts

**Notebooks** (`notebooks/`) are for exploration and prototyping — trying new ideas interactively, visualizing results, one-off experiments. They're kept as historical records of what was tried.

**Scripts** (`scripts/`) are the production pipeline — reproducible, logged, tracked, and work identically on both machines. Use scripts for all training, evaluation, and error analysis going forward.

You don't need to keep them in sync. Notebooks are snapshots; scripts are the living pipeline.

---

## Important Notes

- **Always set `optimizer` explicitly** (e.g. `--optimizer AdamW`). Ultralytics `optimizer: auto` silently overrides `lr0` and `momentum`.
- **Probiou vs geometric IoU**: Ultralytics uses probabilistic IoU for OBB metrics; `error_analysis.py` uses Shapely polygon intersection. Both are valid but produce different counts — don't compare numbers directly.
- **Deployment target**: Jetson Orin Nano 8GB. Inference speed and model size are critical alongside accuracy.
- **Run scripts from the project root** (`robomaster-cv/`). All paths resolve relative to this directory.