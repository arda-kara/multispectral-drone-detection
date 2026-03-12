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

Reads up to three synchronized video streams (RGB, LWIR, UV), runs each frame
through its model, warps LWIR/UV detections into RGB space via homography,
fuses the detections, and writes an annotated output video to `output`.
Prints progress every 100 frames.

**Config keys** (`configs/infer.yaml`):

| Key | Current value | Effect |
|-----|---------------|--------|
| `rgb_model` | path | RGB model weights |
| `lwir_model` | path | LWIR model weights |
| `uv_model` | path or `null` | UV model weights (optional) |
| `rgb_source` | path | RGB input (video file or image directory) |
| `lwir_source` | path | LWIR input (video file or image directory) |
| `uv_source` | path or `null` | UV input (optional) |
| `lwir_homography` | `null` | Path to `.npy` (3×3) homography H_lwir→rgb; `null` = no warp |
| `uv_homography` | `null` | Path to `.npy` (3×3) homography H_uv→rgb; `null` = no warp |
| `lwir_homography_offset` | `[0.0, 0.0]` | Post-warp pixel offset `[dx, dy]` applied after H_lwir |
| `uv_homography_offset` | `[0.0, 0.0]` | Post-warp pixel offset `[dx, dy]` applied after H_uv |
| `output` | `runs/inference/output.mp4` | Output video path |
| `render_mode` | `grid` | See render modes below |
| `max_frames` | `300` | Frame cap; `null` to run until a stream ends |
| `fusion_method` | `spatial` | `spatial`, `wbf`, or `bayesian` |
| `fusion_streams` | `[rgb, lwir, uv]` | Which available streams participate in fusion |
| `center_dist_px` | `60.0` | Spatial fusion: cluster detections by center distance in RGB pixels |
| `fusion_margin` | `0.10` | Expand fused boxes after clustering to absorb small alignment error |
| `iou_thr` | `0.35` | IoU threshold for `wbf` / `bayesian` |
| `skip_thr` | `0.01` | `wbf` / `spatial`: discard predictions below this confidence |
| `conf_thr` | `0.25` | Minimum fused confidence to draw on the output |
| `model_conf` | `0.25` | Per-model YOLO NMS threshold |
| `imgsz` | `960` | Inference resolution |
| `device` | `mps` | Compute device |
| `show` | `true` | Display live preview window (requires a display) |
| `display_scale` | `0.4` | Scale factor for the live window only; saved video is full resolution |

**Annotated `configs/infer.yaml`:**

```yaml
# ── Models ────────────────────────────────────────────────────────────────────
rgb_model:  runs/detect/rgb/yolov8n.pt
# Path to the trained RGB YOLO weights (.pt). Must point to a valid checkpoint.

lwir_model: runs/detect/lwir/yolov8n_960px/weights/best.pt
# Path to the trained LWIR YOLO weights. This model has a single class (drone=0).

uv_model:   runs/detect/uv/yolov8n_800px/weights/best.pt
# Optional UV model. Remove or set to null to run RGB+LWIR only.

# ── Sources ───────────────────────────────────────────────────────────────────
rgb_source:  stream_data/rgb_synced.mp4
# RGB video file or path to a sorted directory of images.

lwir_source: stream_data/lwir_synced.mp4
# LWIR video file or image directory. Read in lockstep with rgb_source.

uv_source:   stream_data/uv_synced.mp4
# UV video file or image directory. Optional — omit or set to null to disable.

# ── Geometric alignment ───────────────────────────────────────────────────────
lwir_homography: homography/H_lwir_to_rgb.npy
# Path to a .npy file containing a (3,3) float32 matrix H that maps
# LWIR pixel coordinates to RGB pixel coordinates. When null, LWIR boxes
# are fused directly in LWIR-normalised space (no geometric correction).

uv_homography: homography/H_uv_to_rgb.npy
# Same as lwir_homography but for the UV→RGB transform.

lwir_homography_offset: [0.0, 0.0]
# Extra post-homography translation in RGB pixels. Use this to fine-tune
# small residual alignment errors without editing the .npy matrix.

uv_homography_offset: [0.0, 0.0]
# Same as lwir_homography_offset. Positive x moves UV boxes to the right
# on the RGB canvas.

# ── Output ────────────────────────────────────────────────────────────────────
output: runs/inference/output.mp4
# Where to write the annotated output video. Parent directories are
# created automatically.

# ── Render mode ───────────────────────────────────────────────────────────────
render_mode: grid
# Controls the layout of the output video. Options:
#   solo         — one panel per active stream, each with its own boxes
#   debug        — 3-panel: RGB-individual | LWIR-individual | RGB-fused
#   sbs          — side-by-side RGB | LWIR, both overlaid with fused boxes
#   fused_on_rgb — single RGB frame with only the fused boxes drawn
#   grid         — 2×2 grid: RGB | UV / LWIR | Fused

max_frames: 300
# Stop after this many frames. Set to null to run until a stream ends.
# Useful for quick smoke-checks without processing the full video.

# ── Fusion ────────────────────────────────────────────────────────────────────
fusion_method: spatial
# Box fusion algorithm:
#   spatial  — center-distance clustering in the shared RGB canvas after
#              homography warping, with confidence-weighted box averaging.
#   wbf      — Weighted Boxes Fusion (Solovyev 2021): clusters overlapping boxes
#              from all streams and returns a weighted-average box per cluster.
#   bayesian — Independent-sensor fusion: matched pairs get
#              P = s1*s2 / (s1*s2 + (1-s1)*(1-s2)); unmatched boxes pass through.

fusion_streams: [rgb, lwir, uv]
# Optional subset of available streams to fuse. Examples:
#   [rgb, lwir]  -> fuse only RGB+LWIR
#   [lwir, uv]   -> fuse only LWIR+UV
#   [uv]         -> pass UV detections through the fused view
# Omit or set to null to fuse all currently available streams.

iou_thr: 0.35
# IoU threshold used by WBF and Bayesian fusion.
# WBF: minimum overlap to merge two boxes into the same cluster.
# Bayesian: minimum overlap to consider two boxes as the same detection.
# Lower → more aggressive merging; higher → keeps more separate boxes.

center_dist_px: 60.0
# Spatial fusion only. Detections whose centers fall within this many RGB
# pixels are treated as the same target, even if IoU is poor because of
# small homography errors or cross-modality box-shape differences.

fusion_margin: 0.10
# Spatial fusion display margin. Expands the final fused box from its center
# so small calibration errors do not clip the target.

skip_thr: 0.01
# WBF and spatial only. Boxes with confidence below this value are dropped
# before clustering. Has no effect in bayesian mode.

conf_thr: 0.25
# Post-fusion gate. Fused boxes below this confidence are not drawn.
# Lower this (e.g. 0.05) if the models are weak (smoke-test weights).

model_conf: 0.25
# Per-model YOLO NMS threshold applied during inference, before fusion.
# Set lower than conf_thr to pass more candidates into the fusion step,
# letting WBF/Bayesian boost confidence through agreement across streams.

# ── Inference ─────────────────────────────────────────────────────────────────
imgsz: 960
# Input resolution fed to each YOLO model. Must match the resolution the
# model was trained at for best accuracy.

device: mps
# Compute device: mps (Apple Silicon), cpu, or a CUDA index such as 0.

show: true
# Open a live preview window during inference (requires a display).
# Press q to quit early.

display_scale: 0.4
# Scales the live preview window only. The saved output.mp4 is always
# written at full resolution regardless of this value.
```

**Render modes:**

| Mode | Output layout | Use for |
|------|---------------|---------|
| `solo` | One panel per active stream, each with its own detections | Per-stream inspection |
| `debug` | 3-panel: RGB-individual \| LWIR-individual \| RGB-fused | Inspecting fusion behaviour |
| `fused_on_rgb` | Single RGB frame with fused boxes | Clean demo output |
| `sbs` | Side-by-side RGB \| LWIR, both annotated with fused boxes | Full-frame comparison |
| `grid` | 2×2 grid: RGB \| UV / LWIR \| Fused | Three-stream monitoring and demos |

**Box colors:**

| Stream | Color |
|--------|-------|
| RGB individual | Orange |
| LWIR individual | Cyan |
| UV individual | Purple |
| Fused | Green |

**Homography convention:**
`H` is a `(3,3) float32` matrix stored as `.npy`. It maps source pixels to RGB
pixels: `p_rgb = H @ [u_src, v_src, 1]^T` (perspective divide applied). Both
LWIR and UV boxes are warped into RGB-normalised space before fusion. An
optional `[dx, dy]` offset is then applied in RGB pixels, which is useful for
fine-tuning cases like UV boxes landing slightly too far left or right without
recomputing the homography itself. When `null`, boxes are fused directly in
each stream's own normalised space (original behaviour).

---

## 6. Live GUI

```bash
python gui.py --config configs/infer.yaml
```

Opens a PyQt6 window with a configurable 2×2 video grid, live detection
metrics, and controls for all inference parameters. Internally runs the same
inference loop as `infer.py` in a background thread.

**CLI flags (all override the YAML config):**

| Flag | Effect |
|------|--------|
| `-c / --config` | YAML config file (default: `configs/infer.yaml`) |
| `-r / --rgb` | RGB model weights |
| `-l / --lwir` | LWIR model weights |
| `-u / --uv` | UV model weights |
| `-rs / --rgb-src` | RGB video source |
| `-ls / --lwir-src` | LWIR video source |
| `-us / --uv-src` | UV video source |
| `--fusion-streams` | Comma-separated fusion subset, e.g. `rgb,lwir` or `lwir,uv` |
| `-d / --device` | Compute device |
| `-n / --max-frames` | Frame cap |

**Grid panel options** (each of the four grid cells is independently selectable):

| Option | Shows |
|--------|-------|
| `RGB` | RGB feed with RGB model detections |
| `LWIR` | LWIR feed with LWIR model detections |
| `UV` | UV feed with UV model detections |
| `Fused` | RGB feed with fused detections from all active streams |
| `Aligned LWIR` | LWIR frame warped to RGB canvas via H_lwir (requires homography loaded) |
| `Aligned UV` | UV frame warped to RGB canvas via H_uv (requires homography loaded) |

**Notable GUI controls:**

| Control | Effect |
|---------|--------|
| `LWIR Offset`, `UV Offset` | Live-edit `[dx, dy]` post-homography shifts |
| `Fuse Streams` | Toggle RGB / LWIR / UV participation in the fused output while the GUI is running |
| `Fusion Algo` | Switch between `spatial`, `wbf`, and `bayesian` |
| `Center Dist px` | Spatial fusion cluster distance threshold |
| `Box Margin %` | Expansion applied to fused boxes for display |

**Detection Metrics panel** (right side, updated every frame):

| Field | Meaning |
|-------|---------|
| FPS | Inference throughput |
| Detections | Number of boxes above `conf_thr` this frame |
| Max Conf | Highest confidence score this frame |
| Avg Conf | Mean confidence across all detections this frame |
| Align | `identity`, `H-loaded`, or `H+off (dx,dy)` depending on current warp setup |

---

## Artifacts

| Path | Written by | Contents |
|------|-----------|----------|
| `data/lwir/` | `build_lwir_dataset.py` | Extracted LWIR frames + labels + `data.yaml` |
| `runs/<modality>/<arch>/weights/best.pt` | `train.py` | Best checkpoint |
| `runs/<modality>/comparison.csv` | `train.py` | Per-architecture metrics from training run |
| `runs/eval_test.csv` | `evaluate.py` | Individual model mAP on the test split |
| `runs/eval_fusion.csv` | `evaluate_fusion.py` | rgb / lwir / wbf / bayesian mAP@50 |
| `runs/inference/output.mp4` | `infer.py` | Annotated multi-stream video |
| `calibration/H_lwir_to_rgb.npy` | user-provided | (3,3) float32 homography for LWIR→RGB alignment |
| `calibration/H_uv_to_rgb.npy` | user-provided | (3,3) float32 homography for UV→RGB alignment |

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
