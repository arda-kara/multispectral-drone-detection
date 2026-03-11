# Command Reference

All commands are run from the repo root with the shared venv active:

```
source /Users/arda-home/Projects/homeworks/venv/bin/activate
```

---

## 1. Build LWIR dataset

```bash
python scripts/build_lwir_dataset.py
```

Extracts LWIR frames from the three scenario videos and pairs them with YOLO
label files. Outputs `data/lwir/` with `train/`, `val/`, `test/` splits and a
`data.yaml`. Only needs to run once.

**Sources** (`data_annotations_lwir/`):

| Video | Annotations | Role |
|-------|-------------|------|
| `Senaryo1_Train_edited_lwir.mp4` | `Senaryo1_lwir_annotations.zip` | train |
| `Senaryo2_edited_lwir.mp4` | `Senaryo2_lwir_annotations.zip` | train |
| `Senaryo3_Test_edited_lwir.mp4` | `Senaryo3_lwir_annotations.zip` | test |

**Hardcoded constants** (edit at the top of the script):

| Name | Default | Effect |
|------|---------|--------|
| `VAL_FRAC` | `0.15` | Fraction of train scenarios held out for val |
| `SEED` | `42` | Shuffle seed for train/val split |

---

## 2. Train

```bash
python train.py --config configs/train.yaml       # RGB
python train.py --config configs/train_lwir.yaml  # LWIR
```

Trains each architecture listed under `architectures` and saves weights to
`runs/<modality>/<arch>_<imgsz>px/weights/`. Prints a comparison table at the
end and saves `runs/<modality>/comparison.csv`.

**Config keys** (`configs/train.yaml`, `configs/train_lwir.yaml`):

| Key | Current value | Effect |
|-----|---------------|--------|
| `modality` | `rgb` / `lwir` | Label used for the run directory |
| `data` | path to `data.yaml` | Dataset definition |
| `architectures` | `[yolo26n.pt]` | List of pretrained weights to iterate over |
| `epochs` | `3` | Training epochs |
| `imgsz` | `320` | Input resolution |
| `batch` | `8` | Base batch size (halved automatically for `s` models) |
| `device` | `mps` | `mps`, `cpu`, `0` (CUDA GPU index) |
| `amp` | `false` | Mixed precision — disable on MPS for stability |
| `fraction` | `0.05` | **Smoke-test only.** Remove this line for real training |
| `workers` | `2` | Dataloader threads |
| `cache` | `false` | Cache images in RAM |
| `plots` | `true` | Save training curve plots |

To add an architecture, uncomment a line or add a new `.pt` name under
`architectures`. Supported: `yolo26n.pt`, `yolo26s.pt`, `yolov8n.pt`,
`yolo11n.pt`, `yolov12n.pt`, `yolov8n-p2`.

---

## 3. Evaluate individual models

```bash
python evaluate.py --config configs/eval.yaml
```

Runs `model.val()` on the test split for each entry in `evaluations`. Saves
plots to the YOLO run directory and a summary table to
`runs/eval_<split>.csv`.

**Config keys** (`configs/eval.yaml`):

| Key | Effect |
|-----|--------|
| `split` | Dataset split to evaluate on: `test`, `val` |
| `imgsz` | Inference resolution |
| `batch` | Batch size |
| `device` | Compute device |
| `plots` | Save confusion matrix and PR curve images |
| `evaluations` | List of `{modality, weights, data}` entries |

To evaluate a new checkpoint, add an entry to `evaluations`:
```yaml
evaluations:
  - modality: rgb
    weights: runs/rgb/yolo26s_640px/weights/best.pt
    data: data/roboflow/data.yaml
```

---

## 4. Evaluate fusion

```bash
python evaluate_fusion.py --config configs/eval_fusion.yaml
```

Iterates over the LWIR test image set, seeks the temporally corresponding RGB
frame from `rgb_video`, and computes 11-point interpolated mAP@50 for four
variants: `rgb`, `lwir`, `wbf`, `bayesian`. Saves
`runs/eval_fusion.csv`.

RGB frame alignment: `rgb_idx = lwir_frame_num × (rgb_fps / lwir_fps)`

**Config keys** (`configs/eval_fusion.yaml`):

| Key | Current value | Effect |
|-----|---------------|--------|
| `rgb_model` | path | RGB model weights |
| `lwir_model` | path | LWIR model weights |
| `lwir_images` | `data/lwir/test/images` | Labeled LWIR frames |
| `lwir_labels` | `data/lwir/test/labels` | YOLO label files |
| `rgb_video` | `data/rgb_test.mp4` | Synchronized RGB stream |
| `lwir_fps` | `30.0` | FPS of the original LWIR source video (for frame mapping) |
| `iou_thr` | `0.55` | IoU threshold for TP matching and WBF clustering |
| `skip_thr` | `0.01` | WBF: discard predictions below this confidence |
| `imgsz` | `320` | Inference resolution |
| `device` | `mps` | Compute device |
| `max_frames` | `200` | Frame cap for quick runs; set to `null` for all 4697 |

---

## 5. Fused inference (video output)

```bash
python infer.py --config configs/infer.yaml
```

Reads two synchronized video streams (or image directories), runs each frame
through its model, fuses the detections, and writes an annotated output video
to `output`. Prints progress every 100 frames.

**Config keys** (`configs/infer.yaml`):

| Key | Current value | Effect |
|-----|---------------|--------|
| `rgb_model` | path | RGB model weights |
| `lwir_model` | path | LWIR model weights |
| `rgb_source` | `data/rgb_test.mp4` | RGB input (video file or image directory) |
| `lwir_source` | `data/lwir_test.mp4` | LWIR input (video file or image directory) |
| `output` | `runs/inference/output.mp4` | Output video path |
| `render_mode` | `debug` | See render modes below |
| `max_frames` | `300` | Frame cap; `null` to run until a stream ends |
| `fusion_method` | `wbf` | `wbf` or `bayesian` |
| `iou_thr` | `0.55` | IoU threshold for fusion |
| `skip_thr` | `0.01` | WBF: discard predictions below this confidence |
| `conf_thr` | `0.05` | Minimum fused confidence to draw on the output |
| `imgsz` | `320` | Inference resolution |
| `device` | `mps` | Compute device |
| `show` | `false` | Display live preview window (requires a display) |

**Render modes:**

| Mode | Output layout | Use for |
|------|---------------|---------|
| `debug` | 3-panel: RGB-individual \| LWIR-individual \| RGB-fused | Inspecting fusion behaviour |
| `fused_on_rgb` | Single RGB frame with fused boxes | Clean demo output |
| `sbs` | Side-by-side RGB \| LWIR, both annotated with fused boxes | Full-frame comparison |

**Box colors:**

| Stream | Color |
|--------|-------|
| RGB individual | Orange |
| LWIR individual | Cyan |
| Fused | Green |

---

## Artifacts

| Path | Written by | Contents |
|------|-----------|----------|
| `data/lwir/` | `build_lwir_dataset.py` | Extracted LWIR frames + labels + `data.yaml` |
| `runs/<modality>/<arch>/weights/best.pt` | `train.py` | Best checkpoint |
| `runs/<modality>/comparison.csv` | `train.py` | Per-architecture metrics from training run |
| `runs/eval_test.csv` | `evaluate.py` | Individual model mAP on the test split |
| `runs/eval_fusion.csv` | `evaluate_fusion.py` | rgb / lwir / wbf / bayesian mAP@50 |
| `runs/inference/output.mp4` | `infer.py` | Annotated dual-stream video |

---

## Upgrading from smoke test to real training

In both `configs/train.yaml` and `configs/train_lwir.yaml`:

```yaml
# Remove this line:
fraction: 0.05

# Set these:
epochs: 30
imgsz: 640
```

Then rerun `train.py`, `evaluate.py`, `evaluate_fusion.py`, and `infer.py` in
order. The RGB model trains best on GPU — use `yolo_cntd.ipynb` in Colab.
