# RoboMaster CV - Armor Plate Detection

YOLO model training and evaluation for NYU RoboMaster VIP team. Tracks experiments with MLflow for reproducible comparisons across models and datasets.

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/jithendra1798/robomaster-cv.git
cd robomaster-cv
pip install -r requirements.txt
```

### 2. Add your dataset
Copy (or symlink) the dataset into the `data/` folder:
```bash
# Structure should be:
# data/mergeRM_v1/images/{train,val,test}/
# data/mergeRM_v1/labels/{train,val,test}/
```

### 3. Train a model
```bash
# Using config file
python scripts/train.py --model yolo11s.pt --config configs/hyperparams/yolo11s_tuned.yaml

# Or with CLI args
python scripts/train.py --model yolo11l.pt --epochs 100 --weight_decay 0.001 --box 6.22111
```

### 4. View results
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Compare models
```bash
python scripts/evaluate.py --compare \
    runs/detect/yolo11s_100ep/weights/best.pt \
    runs/detect/yolo11l_100ep/weights/best.pt
```

## Project Structure

```
robomaster-cv/
├── configs/
│   ├── datasets.yaml              # Dataset version registry
│   ├── data_mergeRM_v1.yaml       # YOLO data config (relative paths)
│   └── hyperparams/               # Saved hyperparameter configs
├── scripts/
│   ├── train.py                   # Training with MLflow logging
│   └── evaluate.py                # Evaluation and model comparison
├── notebooks/                     # Jupyter notebooks for exploration
├── mlruns/                        # MLflow metadata (Git tracked)
├── data/                          # Datasets (NOT in Git)
├── runs/                          # YOLO outputs (NOT in Git)
└── .gitignore
```

## Multi-Machine Workflow

MLflow metadata is tracked in Git. Train on any machine and sync:

```bash
# Before training: get latest results from other machines
git pull

# After training: push your new results
git add mlruns/ configs/
git commit -m "Add yolo11l training run"
git push
```

**Important:** Only mlruns metadata (params, metrics) is in Git. Model weights and datasets stay local.

## Deployment Target

Models are quantized and deployed on **Jetson Orin Nano 8GB**. Inference speed and model size are critical constraints alongside accuracy.
