# Robot detection: datasets and training runs

This is the registry of every dataset version and training run. Each entry records the exact data, code, command and results, so any model can be rebuilt or improved on later.

**Rules**
- Never change a dataset version in place. Build a new one (`robots_v3`, …) and add it here.
- Every run gets an entry: data version, code version, command, results, decision.
- Pick models by the **held-out** scores (see [Evaluation](#evaluation)), not by val/test. Roboflow val/test contain near-copies of training frames, so they flatter every model.
- Machine-written records sit next to the outputs (not in git): `data/<version>/dataset_info.yaml`, `runs/<name>/args.yaml`, `runs/<name>/run_info.yaml`.

**Adding a new dataset version or run**
1. Commit the code first. The automatic record shows `uncommitted_changes: true` otherwise.
2. Build into a new folder (`--out data/robots_v3`) or train with a new `--name`. Never overwrite an existing version.
3. The scripts write `dataset_info.yaml` / `run_info.yaml` and copy themselves into `code/` beside the output. This covers datasets built and runs started after 2026-10-02; earlier entries were recorded by hand below.
4. Score the run with `eval_heldout.py` (together with the current best model), add an entry here, and record the decision.

Environment for everything below unless noted: NYU server `hemeraTwo`, RTX 4090, Python 3.9.7, Ultralytics 8.4.66, torch 2.6.0+cu118. `--robust` runs also use albumentations 2.0.8 + albucore 0.0.24 from `~/jithendra/pylibs`.

Note: on 2026-10-02 at 15:19, someone installed albumentations 2.0.8 into the shared account's `~/.local`. Both v2 runs had finished training by then, so they're unaffected. Since that day's second commit, `train.py` passes `augmentations=[]` for non-`--robust` runs. That keeps Ultralytics' default extras (Blur, MedianBlur, ToGray, CLAHE) off, whether or not the library is installed.

---

## Sources

| Source | Where (server) | Fingerprint | Notes |
|---|---|---|---|
| Roboflow export `robomaster.v1i.yolov8.zip` | `~/jithendra/datasets/` | 979,594,706 bytes, md5 `e489be301d13af459170be79311afe18` | Roboflow Universe `horizon-53bsy/robomaster-oycif` v1, CC BY 4.0, exported 2025-08-01. 9 classes: armor / car / watcher (old rail sentry) × blue / red / grey-or-unknown. |
| mergeRM | `~/Datasets/mergeRM` (team copy; don't edit) | 13,573 jpg+txt files; md5 of the sorted "path size" list `921e84fbe21d2e0357a0febc0a7b814f` | Last year's armor dataset: plate OBB labels only, no robot boxes. `ds1_*` = North American match footage (match name in the filename). `ds2_*` = camera mounted on a robot, dark arenas. |

---

## Datasets

### `robots` (v1)
- **Built:** 2026-10-01 with `python prepare_data.py --zip ~/jithendra/datasets/robomaster.v1i.yolov8.zip`
- **Code:** commit `8752da6`
- **Classes:** `0 robot_blue` (car_blue + watcher_blue), `1 robot_red`, `2 robot_unknown` (car_unknow + watcher_unknow). Armor boxes dropped, exact duplicate rows removed, 5-point polygons → YOLO boxes.

| Split | Images | Background | robot_blue | robot_red | robot_unknown |
|---|---|---|---|---|---|
| train | 18,627 | 534 | 29,736 | 27,371 | 8,822 |
| val | 1,775 | 145 | 1,282 | 1,207 | 434 |
| test | 885 | 80 | 667 | 559 | 224 |

**Known issues:**
- Every train image is a Roboflow mosaic.
- All images were stretched to 640×640.
- In ~30% of train and ~13% of val/test images, some robots are unlabelled (their armor is labelled, the robot box is missing).
- The random split put near-duplicate video frames in different splits.

### `robots_v2`
- **Built:** 2026-10-02 with `python pseudo_label.py --model runs/robots_yolo11s/weights/best.pt --zip ~/jithendra/datasets/robomaster.v1i.yolov8.zip --mergerm ~/Datasets/mergeRM`
- **Code:** commit `4091df7` (`pseudo_label.py` md5 `4bb5d1a7421c561a9e1b75a3253a5767`)
- **Input:** `robots` (v1) + mergeRM, labelled with the v1 model `robots_yolo11s`.

What changed from v1:
1. **Roboflow repair.** A v1 detection (conf ≥ 0.5, IoU ≤ 0.3 with every existing box) became a new robot box only when a human-labelled armor plate with no robot box sits inside it. The colour comes from that human armor label (blue → robot_blue, red → robot_red, grey → robot_unknown). If the plates inside disagree, nothing is added. Result: +976 boxes in train, +133 in val. `test` is unchanged; `test_repaired` has +50.
2. **mergeRM added to train** (files prefixed `mergerm_`; held-out images excluded). The model's detections became robot boxes when conf ≥ 0.5 with a plate inside, or conf ≥ 0.8. An image was kept only if:
   - every plate lies inside an accepted box, and
   - no lit plate contradicts the box's red/blue (HSV colour check, 99.5% correct on Roboflow's human-labelled armor).

   Plate-free images were kept as backgrounds only if v1 detected nothing (conf ≥ 0.25).

   Of 5,654 candidate images, 2,207 were kept (2,281 robot boxes). Rejected: 2,366 with an unexplained plate, 670 backgrounds with detections, 411 colour conflicts.

| Split | Images | Background | robot_blue | robot_red | robot_unknown |
|---|---|---|---|---|---|
| train | 20,834 (2,207 mergeRM) | 828 | 31,387 | 28,855 | 8,944 |
| val | 1,775 | 144 | 1,350 | 1,266 | 440 |
| test (original labels) | 885 | 80 | 667 | 559 | 224 |
| test_repaired | 885 | 78 | 699 | 576 | 225 |

**Known issues:**
- The new boxes have v1's shape (tightness).
- Robots that v1 missed are still unlabelled.
- mergeRM colours come from the model, checked against lit plates.

### `heldout`
- **Built:** with `robots_v2` (same command). Never used for training or for building labels.
- **Contents:** 1,733 mergeRM images.
  - **North American matches (1,116):** every frame of `ARUWvsTAMU` (329) and `NYUvsPUTR` (787), from all mergeRM splits.
  - **Robot camera (617):** the 15% highest `ds2` frame numbers. The 30 frame numbers just below that block are dropped from training.
- **Labels:** only the original human plate labels (1,953 plates in 1,433 images; 300 images have no plates). No robot boxes, on purpose: the scores must not depend on any model's own labels.

---

## Evaluation

`python eval_heldout.py <model.pt> ...` scores each model at the Orin pipeline's confidence threshold, 0.65:

| Metric | Meaning | Better |
|---|---|---|
| Plate coverage | % of held-out plates inside a predicted robot box. Stands in for recall: a missed robot leaves its plates uncovered. Plates on bases/outposts count too, so 100% isn't reachable. | higher |
| Colour agreement | Of covered plates that are clearly lit, % covered by a box of the same colour. | higher |
| Plate-less detections / image | Detections with no plate inside. Stands in for false alarms; robots facing away also count. | lower |
| Detections / background image | On held-out images with no plates. | lower |
| Roboflow test | P / R / mAP50 / mAP50-95 on the original and repaired test labels. | sanity check only |

---

## Runs

| Run | Data | Change from previous | Status |
|---|---|---|---|
| `robots_yolo11s` | `robots` | baseline | done, 2026-10-01 |
| `robots_v2_robust` | `robots_v2` | repaired data + mergeRM + camera augmentations | done, 2026-10-02 |
| `robots_v2_plain` | `robots_v2` | same data, no extra augmentations (ablation) | done, 2026-10-02 |

### `robots_yolo11s` (v1)
- **Command:** `python train.py`
- **Code:** commit `8752da6`
- **Settings:** yolo11s.pt (COCO-pretrained), 100 epochs, patience 20, batch 32, imgsz 640, SGD lr0 0.01 momentum 0.937, seed 0, Ultralytics default augmentation (mosaic 1.0, close_mosaic 10, fliplr 0.5, scale 0.5, hsv_h 0.015), no albumentations.
- **Time:** 1.98 h. Best epoch 100: it was still improving slowly, and val loss was still falling.

**Results:**

| Data | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Roboflow val | 0.947 | 0.946 | 0.970 | 0.829 |
| Roboflow test (original) | 0.958 | 0.942 | 0.969 | 0.829 |
| Roboflow test (repaired) | 0.966 | 0.923 | 0.968 | 0.829 |

| Held-out | Plate coverage | Colour agreement | Plate-less dets/img | Dets/background img |
|---|---|---|---|---|
| North American matches | 44.2% | 97.9% | 0.19 | 0.26 |
| Robot camera | 61.1% | 86.3% | 0.01 | 0.96 |

**Notes:**
- Red ↔ blue mix-ups on test: 3 of 1,226.
- Most "false positives" on test are robots the labellers never boxed.
- It is much weaker out of domain: under half the North American match plates are covered at conf 0.65, and colour is shaky on robot-camera footage.

**Decision:** baseline; used to build `robots_v2`. Not deployed yet.

### `robots_v2_robust`
- **Command:**
  ```bash
  PYTHONPATH=~/jithendra/pylibs python train.py --data data/robots_v2/data.yaml --robust \
    --epochs 150 --patience 40 --save-period 25 --name robots_v2_robust
  ```
- **Code:** commit `4091df7` (`train.py` md5 `d6a4f8e79fb9a0b6d3a4c69b9d15f800`)
- **Settings:** as v1, plus 150 epochs, patience 40, a checkpoint every 25 epochs, and these augmentations:
  - MotionBlur (blur_limit 3–15, p 0.25)
  - GaussNoise (std 0.02–0.08, p 0.15)
  - Affine horizontal stretch ×1.0–1.5 (p 0.3)

  These replace Ultralytics' default albumentations set, so there is no ToGray.
- **Time:** 3.42 h, all 150 epochs (no early stop). `best.pt` = `last.pt` = epoch 150.

### `robots_v2_plain`
- **Command:**
  ```bash
  python train.py --data data/robots_v2/data.yaml --epochs 150 --patience 40 --save-period 25 --name robots_v2_plain
  ```
  Run without `PYTHONPATH`, so it uses the same augmentation as v1.
- **Code:** same `train.py` as above.
- **Time:** 3.33 h, all 150 epochs (no early stop). `best.pt` = `last.pt` = epoch 150.

### v2 results (`runs/eval_v2.log`)

Final models (`best.pt`), held-out at conf 0.65 (coverage / colour agreement / plate-less dets per image / dets per background image):

| Model | North American matches | Robot camera | Roboflow test P / R / mAP50 / mAP50-95 |
|---|---|---|---|
| v1 `robots_yolo11s` | 44.2% / 97.9% / 0.19 / 0.26 | 61.1% / 86.3% / 0.01 / 0.96 | 0.958 / 0.942 / 0.969 / 0.829 |
| `robots_v2_plain` | 48.5% / 98.8% / 0.17 / 0.19 | 94.7% / 85.3% / 0.03 / 1.00 | 0.956 / 0.941 / 0.968 / 0.844 |
| `robots_v2_robust` | 51.4% / 98.7% / 0.14 / 0.26 | 93.9% / 85.0% / 0.03 / 1.00 | 0.958 / 0.940 / 0.968 / 0.833 |

Held-out plate coverage over training (`epochN.pt` holds the weights after N+1 epochs):

| Epochs trained | robust: NA / robot cam | plain: NA / robot cam | Roboflow test mAP50-95 (robust / plain) |
|---|---|---|---|
| 1 | 42.3% / 60.7% | 37.4% / 8.4% | 0.584 / 0.607 |
| 26 | 54.0% / 98.2% | 54.7% / 93.2% | 0.742 / 0.753 |
| **51** | **57.9% / 98.9%** | **58.5% / 98.1%** | 0.777 / 0.787 |
| 76 | 55.9% / 98.4% | 55.2% / 98.6% | 0.792 / 0.804 |
| 101 | 54.6% / 97.5% | 53.9% / 98.2% | 0.807 / 0.817 |
| 126 | 51.9% / 96.5% | 49.9% / 96.5% | 0.822 / 0.832 |
| 150 (`best.pt`) | 51.4% / 93.9% | 48.5% / 94.7% | 0.833 / 0.844 |

**Findings:**
1. **The v2 data helps a lot out of domain.** Robot-camera plate coverage went from 61% to 94–99%, and North American matches from 44% to 49–58%. In-domain scores are unchanged (Roboflow test mAP50 ≈ 0.97).
2. **Long training overfits the training domain.** Held-out coverage peaks around 51 epochs and falls by epoch 150, while Roboflow test keeps improving. Picking `best.pt` by Roboflow val therefore picks the overfitted end.
3. **A mid-run checkpoint is not a finished model.** `epoch50.pt` was saved with the learning rate still high and before the final no-mosaic epochs. Its Roboflow test precision is 0.92, against 0.96 for `best.pt`. The fix is a complete short schedule, not deploying `epoch50.pt`.
4. **Robust vs plain:** no clear difference at the peak. At epoch 150, robust kept more North American coverage (51.4% vs 48.5%) with fewer plate-less detections (0.14 vs 0.17); plain has tighter boxes (mAP50-95 0.844 vs 0.833). The held-out set doesn't contain spin blur, so the augmentations' real test is the sentry camera.
5. **Open: robot-camera colour agreement is about 85% for every model, v1 included.** It's unclear whether that's the models or the HSV check on dark, saturated LEDs. Needs a visual check.
6. **Open: robot-camera "background" images get about 1 detection each from every model.** These images may contain robots with no labelled plates. Needs a visual check.

**Decision:** no model deployed yet. Proposed next run: the same robust setup with a ~60-epoch schedule, so the model finishes training around the held-out peak. Then pick the Tuesday (2026-10-06) model by held-out scores. Until then, the provisional choice is `robots_v2_robust/best.pt`.

---

## Deployed models

| Date | File on the Orin | From run | Engine build | Notes |
|---|---|---|---|---|
| — | — | — | — | Nothing deployed yet. First live test planned for 2026-10-06. |
