"""
Fusion evaluation: compare RGB-only, LWIR-only, WBF, and Bayesian against
LWIR test-set ground-truth labels.

RGB frames are read from a synchronized video by mapping each LWIR frame's
embedded index through the fps ratio:  rgb_idx = lwir_frame_num * (rgb_fps / lwir_fps)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from fusion import filter_lwir, filter_rgb, fuse


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── ground truth ──────────────────────────────────────────────────────────────

def load_gt(label_path: Path) -> np.ndarray:
    """YOLO label file → (N, 4) normalised [x1, y1, x2, y2]."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = [float(v) for v in parts[:5]]
        rows.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


# ── IoU + matching ────────────────────────────────────────────────────────────

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) pairwise IoU for two sets of [x1, y1, x2, y2] boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_preds(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
) -> list[tuple[float, bool]]:
    """Return (score, is_tp) pairs for this image, sorted by descending score."""
    if len(pred_boxes) == 0:
        return []

    order = np.argsort(-pred_scores)
    pred_boxes  = pred_boxes[order]
    pred_scores = pred_scores[order]

    matched_gt: set = set()
    ious = iou_matrix(pred_boxes, gt_boxes)   # (N_pred, N_gt)
    records = []

    for i in range(len(pred_boxes)):
        is_tp = False
        if ious.shape[1] > 0:
            best_j = int(np.argmax(ious[i]))
            if ious[i, best_j] >= iou_thr and best_j not in matched_gt:
                is_tp = True
                matched_gt.add(best_j)
        records.append((float(pred_scores[i]), is_tp))

    return records


# ── AP calculation ────────────────────────────────────────────────────────────

def compute_ap(records: list[tuple[float, bool]], n_gt: int) -> float:
    """11-point interpolated AP from (score, is_tp) records across all images."""
    if n_gt == 0 or not records:
        return 0.0
    records = sorted(records, key=lambda x: -x[0])
    tp = fp = 0
    prec, rec = [], []
    for _, is_tp in records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        vals = [p for p, r in zip(prec, rec) if r >= t]
        ap += max(vals) if vals else 0.0
    return ap / 11


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    rgb_model  = YOLO(cfg["rgb_model"])
    lwir_model = YOLO(cfg["lwir_model"])

    lwir_imgs = sorted(Path(cfg["lwir_images"]).glob("*.jpg"))
    lbl_dir   = Path(cfg["lwir_labels"])

    rgb_cap  = cv2.VideoCapture(cfg["rgb_video"])
    rgb_fps  = rgb_cap.get(cv2.CAP_PROP_FPS) or 50.0
    lwir_fps = float(cfg.get("lwir_fps", 30.0))

    iou_thr  = float(cfg.get("iou_thr",  0.55))
    skip_thr = float(cfg.get("skip_thr", 0.01))
    imgsz    = int(cfg.get("imgsz",      320))
    device   = cfg.get("device", "cpu")
    max_frames = cfg.get("max_frames", None)

    infer_kw = dict(imgsz=imgsz, device=device, verbose=False, conf=0.01)

    # Accumulators: {variant: (records_list, n_gt)}
    acc: dict[str, tuple[list, int]] = {
        k: ([], 0) for k in ("rgb", "lwir", "wbf", "bayesian")
    }

    if max_frames is not None:
        lwir_imgs = lwir_imgs[:max_frames]

    print(f"Evaluating {len(lwir_imgs)} frames ...")

    for i, lwir_path in enumerate(lwir_imgs):
        # Frame number is the trailing integer in the stem: senaryo3_frame_000042 → 42
        try:
            frame_num = int(lwir_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            frame_num = i

        lwir_frame = cv2.imread(str(lwir_path))
        if lwir_frame is None:
            continue

        # Seek RGB to temporally corresponding frame
        rgb_idx = int(round(frame_num * rgb_fps / lwir_fps))
        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
        ok, rgb_frame = rgb_cap.read()
        if not ok or rgb_frame is None:
            rgb_frame = np.zeros_like(lwir_frame)   # blank fallback

        gt = load_gt(lbl_dir / f"{lwir_path.stem}.txt")

        # Inference
        rgb_res  = rgb_model(rgb_frame,   **infer_kw)[0]
        lwir_res = lwir_model(lwir_frame, **infer_kw)[0]

        boxes_r,   scores_r   = filter_rgb(rgb_res)
        boxes_l,   scores_l   = filter_lwir(lwir_res)
        boxes_wbf, scores_wbf = fuse(boxes_r, scores_r, boxes_l, scores_l,
                                     method="wbf", iou_thr=iou_thr, skip_thr=skip_thr)
        boxes_bay, scores_bay = fuse(boxes_r, scores_r, boxes_l, scores_l,
                                     method="bayesian", iou_thr=iou_thr)

        for name, (bx, sc) in [
            ("rgb",      (boxes_r,   scores_r)),
            ("lwir",     (boxes_l,   scores_l)),
            ("wbf",      (boxes_wbf, scores_wbf)),
            ("bayesian", (boxes_bay, scores_bay)),
        ]:
            recs = match_preds(bx, sc, gt, iou_thr)
            records, n_gt = acc[name]
            records.extend(recs)
            acc[name] = (records, n_gt + len(gt))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(lwir_imgs)} frames", flush=True)

    rgb_cap.release()

    rows = []
    for name, (records, n_gt) in acc.items():
        ap = compute_ap(records, n_gt)
        rows.append({
            "variant":    name,
            "mAP@50":     round(ap, 4),
            "n_gt_boxes": n_gt,
            "n_preds":    len(records),
        })

    df = pd.DataFrame(rows)
    print("\n=== Fusion Evaluation (LWIR test set) ===")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))

    out = Path(cfg.get("project", "runs")) / "eval_fusion.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fusion vs individual models.")
    parser.add_argument("--config", default="configs/eval_fusion.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
