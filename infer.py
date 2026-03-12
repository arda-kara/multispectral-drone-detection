"""
Multi-stream inference with optional bounding-box fusion.

Reads up to three synchronized video streams (RGB, LWIR, UV), runs each frame
through its trained YOLO model, and writes an annotated output video.

Render modes
------------
solo         : one panel per active stream, each annotated with its own detections
sbs          : side-by-side RGB | LWIR, both panels annotated with fused boxes
fused_on_rgb : single RGB frame annotated with fused boxes only
debug        : RGB-individual | LWIR-individual | RGB-fused

CLI (all flags override the config file)
-----------------------------------------
  -c / --config     path to YAML config        (default: configs/infer.yaml)
  -r / --rgb        RGB model weights (.pt)
  -l / --lwir       LWIR model weights (.pt)
  -u / --uv         UV model weights (.pt)
  -rs/ --rgb-src    RGB video source
  -ls/ --lwir-src   LWIR video source
  -us/ --uv-src     UV video source
  -d / --device     compute device: mps, cpu, 0  (default: mps)
  -o / --output     output video path
  -n / --max-frames frame cap (default: run until a stream ends)
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

# Per-band colours
_COLOR = {
    "rgb":   (255, 128,   0),   # orange
    "lwir":  (  0, 200, 255),   # cyan
    "uv":    (180,   0, 255),   # purple
    "fused": (  0, 255,   0),   # green
}


def render_frame(
    frames:  dict[str, np.ndarray],   # {"rgb": ..., "lwir": ..., "uv": ...}  uv optional
    boxes:   dict[str, np.ndarray],
    scores:  dict[str, np.ndarray],
    fused_boxes:  np.ndarray,
    fused_scores: np.ndarray,
    mode: str,
    frame_idx: int,
    fusion_method: str,
    fused_streams: list[str] | None = None,
) -> np.ndarray:
    fusion_label = f"{'·'.join(s.upper() for s in (fused_streams or ['RGB', 'LWIR']))}  {fusion_method.upper()}"
    h_ref = to_bgr(frames["rgb"]).shape[0]

    if mode == "fused_on_rgb":
        out = draw_boxes(to_bgr(frames["rgb"]), fused_boxes, fused_scores,
                         label="fused", color=_COLOR["fused"])
        return add_banner(out, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])

    if mode == "sbs":
        p_rgb  = draw_boxes(to_bgr(frames["rgb"]),  fused_boxes, fused_scores,
                            label="fused", color=_COLOR["fused"])
        p_lwir = draw_boxes(to_bgr(frames["lwir"]), fused_boxes, fused_scores,
                            label="fused", color=_COLOR["fused"])
        p_rgb  = add_banner(p_rgb, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])
        return np.hstack([p_rgb, resize_to_height(p_lwir, h_ref)])

    if mode == "debug":
        p_rgb  = draw_boxes(to_bgr(frames["rgb"]),  boxes["rgb"],  scores["rgb"],
                            label="rgb",  color=_COLOR["rgb"])
        p_lwir = draw_boxes(to_bgr(frames["lwir"]), boxes["lwir"], scores["lwir"],
                            label="lwir", color=_COLOR["lwir"])
        p_fuse = draw_boxes(to_bgr(frames["rgb"]),  fused_boxes,   fused_scores,
                            label="fused", color=_COLOR["fused"])
        p_rgb  = add_banner(p_rgb,  "RGB individual",  color=_COLOR["rgb"])
        p_lwir = add_banner(p_lwir, "LWIR individual", color=_COLOR["lwir"])
        p_fuse = add_banner(p_fuse, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])
        return np.hstack([p_rgb, resize_to_height(p_lwir, h_ref), p_fuse])

    # solo: one panel per active stream, own detections only
    panels = []
    for band in ("rgb", "lwir", "uv"):
        if band not in frames:
            continue
        p = draw_boxes(to_bgr(frames[band]), boxes[band], scores[band],
                       label=band, color=_COLOR[band])
        p = add_banner(p, f"{band.upper()}  f:{frame_idx}", color=_COLOR[band])
        panels.append(resize_to_height(p, h_ref))
    return np.hstack(panels)


# ── main loop ─────────────────────────────────────────────────────────────────

def run(cfg: dict):
    # Load models — UV is optional
    rgb_model  = YOLO(cfg["rgb_model"])
    lwir_model = YOLO(cfg["lwir_model"])
    uv_model   = YOLO(cfg["uv_model"]) if cfg.get("uv_model") else None

    rgb_src  = FrameSource(cfg["rgb_source"])
    lwir_src = FrameSource(cfg["lwir_source"])
    uv_src   = FrameSource(cfg["uv_source"]) if cfg.get("uv_source") else None

    method      = cfg.get("fusion_method", "wbf")
    iou_thr     = cfg.get("iou_thr",     0.55)
    skip_thr    = cfg.get("skip_thr",    0.01)
    conf_thr    = cfg.get("conf_thr",    0.25)
    imgsz       = cfg.get("imgsz",       960)
    device      = cfg.get("device",      "cpu")
    show        = cfg.get("show",        False)
    render_mode   = cfg.get("render_mode",   "solo")
    max_frames    = cfg.get("max_frames",    None)
    display_scale = cfg.get("display_scale", 0.4)

    out_path = Path(cfg.get("output", "runs/inference/output.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # conf_thr drives both the model NMS and the post-fusion gate.
    # For fusion modes (debug/sbs/fused_on_rgb) a lower value can be set in
    # the config to let WBF collect more candidates before merging.
    model_conf   = cfg.get("model_conf", conf_thr)
    infer_kwargs = dict(imgsz=imgsz, device=device, verbose=False, conf=model_conf)

    if show:
        cv2.namedWindow("Inference  [q to quit]", cv2.WINDOW_NORMAL)

    writer    = None
    frame_idx = 0

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok_r, rgb_frame  = rgb_src.read()
        ok_l, lwir_frame = lwir_src.read()
        if not ok_r or not ok_l:
            break

        uv_frame = None
        if uv_src is not None:
            ok_u, uv_frame = uv_src.read()
            if not ok_u:
                break

        # Inference
        rgb_res  = rgb_model(rgb_frame,   **infer_kwargs)[0]
        lwir_res = lwir_model(lwir_frame, **infer_kwargs)[0]
        uv_res   = uv_model(uv_frame,     **infer_kwargs)[0] if uv_model and uv_frame is not None else None

        # Individual detections
        boxes_r, scores_r = filter_rgb(rgb_res)
        boxes_l, scores_l = filter_lwir(lwir_res)
        boxes_u, scores_u = filter_lwir(uv_res) if uv_res is not None else (
            np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        )

        # Fused detections (RGB + LWIR + UV when UV is active)
        fused_boxes, fused_scores = fuse(
            boxes_r, scores_r, boxes_l, scores_l,
            method=method, iou_thr=iou_thr, skip_thr=skip_thr,
            boxes3=boxes_u if uv_model else None,
            scores3=scores_u if uv_model else None,
        )
        if len(fused_scores):
            keep = fused_scores >= conf_thr
            fused_boxes  = fused_boxes[keep]
            fused_scores = fused_scores[keep]

        # Build per-band dicts for renderer
        frames_dict = {"rgb": rgb_frame, "lwir": lwir_frame}
        boxes_dict  = {"rgb": boxes_r,   "lwir": boxes_l}
        scores_dict = {"rgb": scores_r,  "lwir": scores_l}
        if uv_frame is not None:
            frames_dict["uv"] = uv_frame
            boxes_dict["uv"]  = boxes_u
            scores_dict["uv"] = scores_u

        fused_streams = ["rgb", "lwir"] + (["uv"] if uv_model else [])
        combined = render_frame(
            frames_dict, boxes_dict, scores_dict,
            fused_boxes, fused_scores,
            mode=render_mode,
            frame_idx=frame_idx,
            fusion_method=method,
            fused_streams=fused_streams,
        )

        if writer is None:
            h, w = combined.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), rgb_src.fps, (w, h),
            )

        writer.write(combined)

        if show:
            preview = cv2.resize(combined, None, fx=display_scale, fy=display_scale)
            cv2.imshow("Inference  [q to quit]", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames processed", flush=True)

    rgb_src.release()
    lwir_src.release()
    if uv_src:
        uv_src.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nDone — {frame_idx} frames → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-stream inference (RGB / LWIR / UV) with optional fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c",  "--config",    default="configs/infer.yaml", help="YAML config file")
    p.add_argument("-r",  "--rgb",       metavar="PT",  help="RGB model weights")
    p.add_argument("-l",  "--lwir",      metavar="PT",  help="LWIR model weights")
    p.add_argument("-u",  "--uv",        metavar="PT",  help="UV model weights")
    p.add_argument("-rs", "--rgb-src",   metavar="SRC", help="RGB video source")
    p.add_argument("-ls", "--lwir-src",  metavar="SRC", help="LWIR video source")
    p.add_argument("-us", "--uv-src",    metavar="SRC", help="UV video source")
    p.add_argument("-m",  "--mode",      metavar="MODE",
                   choices=["solo", "debug", "sbs", "fused_on_rgb"],
                   help="render mode: solo | debug | sbs | fused_on_rgb")
    p.add_argument("-d",  "--device",    metavar="DEV", help="mps | cpu | 0 (CUDA)")
    p.add_argument("-o",  "--output",    metavar="MP4", help="output video path")
    p.add_argument("-n",  "--max-frames", metavar="N", type=int, help="frame cap")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg  = load_config(args.config)

    # CLI overrides config
    if args.rgb:        cfg["rgb_model"]   = args.rgb
    if args.lwir:       cfg["lwir_model"]  = args.lwir
    if args.uv:         cfg["uv_model"]    = args.uv
    if args.rgb_src:    cfg["rgb_source"]  = args.rgb_src
    if args.lwir_src:   cfg["lwir_source"] = args.lwir_src
    if args.uv_src:     cfg["uv_source"]   = args.uv_src
    if args.mode:       cfg["render_mode"] = args.mode
    if args.device:     cfg["device"]      = args.device
    if args.output:     cfg["output"]      = args.output
    if args.max_frames: cfg["max_frames"]  = args.max_frames

    run(cfg)
