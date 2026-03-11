"""
Dual-stream inference with bounding-box fusion.

Reads a pair of video files or image directories (RGB + LWIR), runs each
frame through its trained YOLO model, fuses the detections, and writes an
annotated output video.

Render modes
------------
sbs          : side-by-side RGB | LWIR, both panels annotated with fused boxes
fused_on_rgb : single RGB frame annotated with fused boxes only
debug        : three-panel — RGB-individual | LWIR-individual | RGB-fused
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from fusion import filter_lwir, filter_rgb, fuse


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── frame source ──────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class FrameSource:
    """Unified reader for a video file or a sorted directory of images."""

    def __init__(self, path: str):
        p = Path(path)
        if p.is_dir():
            self._files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)
            self._idx = 0
            self._cap = None
            self._fps = 25.0
        elif p.is_file():
            self._cap = cv2.VideoCapture(str(p))
            self._files = None
            self._idx = 0
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        else:
            raise FileNotFoundError(f"Source not found: {path}")

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        if self._cap is not None:
            return self._cap.read()
        if self._idx >= len(self._files):
            return False, None
        frame = cv2.imread(str(self._files[self._idx]))
        self._idx += 1
        return frame is not None, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    label: str = "drone",
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw normalised xyxy boxes with score labels onto a copy of frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    for box, score in zip(boxes, scores):
        pt1 = (int(box[0] * w), int(box[1] * h))
        pt2 = (int(box[2] * w), int(box[3] * h))
        cv2.rectangle(out, pt1, pt2, color, thickness)
        cv2.putText(
            out, f"{label} {score:.2f}", (pt1[0], max(pt1[1] - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out


def add_banner(frame: np.ndarray, text: str, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Overlay a text banner at the top-left with a dark outline for readability."""
    out = frame.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    return cv2.resize(frame, None, fx=scale, fy=scale)


# ── rendering ─────────────────────────────────────────────────────────────────

def render_frame(
    rgb_frame: np.ndarray,
    lwir_frame: np.ndarray,
    boxes_r: np.ndarray,
    scores_r: np.ndarray,
    boxes_l: np.ndarray,
    scores_l: np.ndarray,
    fused_boxes: np.ndarray,
    fused_scores: np.ndarray,
    mode: str,
    frame_idx: int,
    fusion_method: str,
) -> np.ndarray:
    rgb  = to_bgr(rgb_frame)
    lwir = to_bgr(lwir_frame)

    if mode == "fused_on_rgb":
        out = draw_boxes(rgb, fused_boxes, fused_scores, label="fused", color=(0, 255, 0))
        return add_banner(out, f"fused:{fusion_method.upper()}  f:{frame_idx}", color=(0, 255, 0))

    if mode == "sbs":
        rgb_ann  = draw_boxes(rgb,  fused_boxes, fused_scores, label="fused", color=(0, 255, 0))
        lwir_ann = draw_boxes(lwir, fused_boxes, fused_scores, label="fused", color=(0, 200, 255))
        rgb_ann  = add_banner(rgb_ann, f"fused:{fusion_method.upper()}  f:{frame_idx}", color=(0, 255, 0))
        h = rgb_ann.shape[0]
        return np.hstack([rgb_ann, resize_to_height(lwir_ann, h)])

    # debug: RGB-individual | LWIR-individual | RGB-fused
    panel_rgb  = draw_boxes(rgb,  boxes_r,     scores_r,     label="rgb",   color=(255, 128, 0))
    panel_lwir = draw_boxes(lwir, boxes_l,     scores_l,     label="lwir",  color=(0, 200, 255))
    panel_fuse = draw_boxes(rgb,  fused_boxes, fused_scores, label="fused", color=(0, 255, 0))

    panel_rgb  = add_banner(panel_rgb,  "RGB individual",                        color=(255, 128, 0))
    panel_lwir = add_banner(panel_lwir, "LWIR individual",                       color=(0, 200, 255))
    panel_fuse = add_banner(panel_fuse, f"Fused:{fusion_method.upper()}  f:{frame_idx}", color=(0, 255, 0))

    h = panel_rgb.shape[0]
    return np.hstack([panel_rgb, resize_to_height(panel_lwir, h), panel_fuse])


# ── main loop ─────────────────────────────────────────────────────────────────

def run(cfg: dict):
    rgb_model  = YOLO(cfg["rgb_model"])
    lwir_model = YOLO(cfg["lwir_model"])

    rgb_src  = FrameSource(cfg["rgb_source"])
    lwir_src = FrameSource(cfg["lwir_source"])

    method      = cfg.get("fusion_method", "wbf")
    iou_thr     = cfg.get("iou_thr",     0.55)
    skip_thr    = cfg.get("skip_thr",    0.01)
    conf_thr    = cfg.get("conf_thr",    0.05)
    imgsz       = cfg.get("imgsz",       640)
    device      = cfg.get("device",      "cpu")
    show        = cfg.get("show",        False)
    render_mode = cfg.get("render_mode", "debug")
    max_frames  = cfg.get("max_frames",  None)

    out_path = Path(cfg.get("output", "runs/inference/output.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    infer_kwargs = dict(imgsz=imgsz, device=device, verbose=False, conf=0.01)

    writer    = None
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok_r, rgb_frame  = rgb_src.read()
        ok_l, lwir_frame = lwir_src.read()

        if not ok_r or not ok_l:
            break

        # Inference
        rgb_result  = rgb_model(rgb_frame,   **infer_kwargs)[0]
        lwir_result = lwir_model(lwir_frame, **infer_kwargs)[0]

        # Individual detections
        boxes_r, scores_r = filter_rgb(rgb_result)
        boxes_l, scores_l = filter_lwir(lwir_result)

        # Fused detections
        fused_boxes, fused_scores = fuse(
            boxes_r, scores_r, boxes_l, scores_l,
            method=method, iou_thr=iou_thr, skip_thr=skip_thr,
        )

        # Confidence gate on fused output
        if len(fused_scores):
            keep = fused_scores >= conf_thr
            fused_boxes  = fused_boxes[keep]
            fused_scores = fused_scores[keep]

        combined = render_frame(
            rgb_frame, lwir_frame,
            boxes_r, scores_r,
            boxes_l, scores_l,
            fused_boxes, fused_scores,
            mode=render_mode,
            frame_idx=frame_idx,
            fusion_method=method,
        )

        if writer is None:
            h, w = combined.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), rgb_src.fps, (w, h),
            )

        writer.write(combined)

        if show:
            cv2.imshow("Inference  [q to quit]", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames processed", flush=True)

    rgb_src.release()
    lwir_src.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nDone — {frame_idx} frames → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-stream fused inference.")
    parser.add_argument("--config", default="configs/infer.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
