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
grid         : 2x2 grid: RGB | UV / LWIR | Fused

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

from fusion import expand_boxes, filter_lwir, filter_rgb, fuse, load_homography, warp_boxes


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


def draw_fused_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    details: list[dict[str, object]] | None = None,
    margin_percent: float = 0.0,
    thickness: int = 3,
) -> np.ndarray:
    """Draw fused boxes with sensor-count aware styling."""
    h, w = frame.shape[:2]
    out = frame.copy()
    draw_boxes_norm = expand_boxes(boxes, margin_percent=margin_percent)

    for idx, (box, score) in enumerate(zip(draw_boxes_norm, scores)):
        detail = details[idx] if details is not None and idx < len(details) else None
        sensor_count = int(detail["sensor_count"]) if detail and "sensor_count" in detail else None
        sensor_total = int(detail["sensor_total"]) if detail and "sensor_total" in detail else None

        if sensor_count is None:
            color = _COLOR["fused"]
            label = f"fused {score:.2f}"
        else:
            sensor_total = sensor_total or sensor_count
            if sensor_count >= sensor_total:
                color = (0, 255, 0)
            elif sensor_count >= 2:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            label = f"drone ({sensor_count}/{sensor_total}) {score:.2f}"

        pt1 = (int(box[0] * w), int(box[1] * h))
        pt2 = (int(box[2] * w), int(box[3] * h))
        cv2.rectangle(out, pt1, pt2, color, thickness)
        cv2.putText(
            out,
            label,
            (pt1[0], max(pt1[1] - 6, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
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


def make_blank_panel(size: tuple[int, int], text: str = "Unused") -> np.ndarray:
    """Create a labelled placeholder panel for grid layouts."""
    width, height = size
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    return add_banner(blank, text, color=(160, 160, 160))


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
    fused_details: list[dict[str, object]] | None,
    mode: str,
    frame_idx: int,
    fusion_method: str,
    fused_streams: list[str] | None = None,
    fusion_margin: float = 0.0,
) -> np.ndarray:
    fusion_label = f"{'·'.join(s.upper() for s in (fused_streams or ['RGB', 'LWIR']))}  {fusion_method.upper()}"
    h_ref = to_bgr(frames["rgb"]).shape[0]

    if mode == "fused_on_rgb":
        out = draw_fused_boxes(
            to_bgr(frames["rgb"]),
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        return add_banner(out, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])

    if mode == "sbs":
        aligned_lwir = to_bgr(frames.get("aligned_lwir", frames["lwir"]))
        p_rgb = draw_fused_boxes(
            to_bgr(frames["rgb"]),
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        p_lwir = draw_fused_boxes(
            aligned_lwir,
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        p_rgb  = add_banner(p_rgb, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])
        p_lwir = add_banner(p_lwir, "LWIR aligned", color=_COLOR["lwir"])
        return np.hstack([p_rgb, resize_to_height(p_lwir, h_ref)])

    if mode == "debug":
        p_rgb  = draw_boxes(to_bgr(frames["rgb"]),  boxes["rgb"],  scores["rgb"],
                            label="rgb",  color=_COLOR["rgb"])
        p_lwir = draw_boxes(to_bgr(frames["lwir"]), boxes["lwir"], scores["lwir"],
                            label="lwir", color=_COLOR["lwir"])
        p_fuse = draw_fused_boxes(
            to_bgr(frames["rgb"]),
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        p_rgb  = add_banner(p_rgb,  "RGB individual",  color=_COLOR["rgb"])
        p_lwir = add_banner(p_lwir, "LWIR individual", color=_COLOR["lwir"])
        p_fuse = add_banner(p_fuse, f"{fusion_label}  f:{frame_idx}", color=_COLOR["fused"])
        return np.hstack([p_rgb, resize_to_height(p_lwir, h_ref), p_fuse])

    if mode == "grid":
        panel_size = (frames["rgb"].shape[1] // 2, frames["rgb"].shape[0] // 2)
        tiles = [
            add_banner(
                cv2.resize(
                    draw_boxes(to_bgr(frames["rgb"]), boxes["rgb"], scores["rgb"], label="rgb", color=_COLOR["rgb"]),
                    panel_size,
                ),
                f"RGB  f:{frame_idx}",
                color=_COLOR["rgb"],
            )
        ]

        if "uv" in frames:
            tiles.append(
                add_banner(
                    cv2.resize(
                        draw_boxes(to_bgr(frames["uv"]), boxes["uv"], scores["uv"], label="uv", color=_COLOR["uv"]),
                        panel_size,
                    ),
                    f"UV  f:{frame_idx}",
                    color=_COLOR["uv"],
                )
            )
        else:
            tiles.append(make_blank_panel(panel_size, text="UV unavailable"))

        tiles.append(
            add_banner(
                cv2.resize(
                    draw_boxes(to_bgr(frames["lwir"]), boxes["lwir"], scores["lwir"], label="lwir", color=_COLOR["lwir"]),
                    panel_size,
                ),
                f"LWIR  f:{frame_idx}",
                color=_COLOR["lwir"],
            )
        )
        tiles.append(
            add_banner(
                cv2.resize(
                    draw_fused_boxes(
                        to_bgr(frames["rgb"]),
                        fused_boxes,
                        fused_scores,
                        details=fused_details,
                        margin_percent=fusion_margin,
                    ),
                    panel_size,
                ),
                f"FUSED ({len(fused_boxes)})  f:{frame_idx}",
                color=_COLOR["fused"],
            )
        )
        return cv2.vconcat(
            [
                cv2.hconcat(tiles[:2]),
                cv2.hconcat(tiles[2:4]),
            ]
        )

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

    H_lwir = load_homography(cfg.get("lwir_homography"), cfg.get("lwir_homography_offset"))
    H_uv   = load_homography(cfg.get("uv_homography"), cfg.get("uv_homography_offset"))

    method      = cfg.get("fusion_method", "wbf")
    iou_thr     = cfg.get("iou_thr",     0.55)
    skip_thr    = cfg.get("skip_thr",    0.01)
    conf_thr    = cfg.get("conf_thr",    0.25)
    center_dist_px = float(cfg.get("center_dist_px", 60.0))
    fusion_margin  = float(cfg.get("fusion_margin", 0.10))
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
        boxes_l_raw, scores_l = filter_lwir(lwir_res)
        boxes_u_raw, scores_u = filter_lwir(uv_res) if uv_res is not None else (
            np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        )

        boxes_l_fused = boxes_l_raw.copy()
        boxes_u_fused = boxes_u_raw.copy()

        # Warp LWIR and UV boxes into RGB pixel space before fusion
        if H_lwir is not None:
            boxes_l_fused = warp_boxes(
                boxes_l_raw,
                H_lwir,
                (lwir_frame.shape[1], lwir_frame.shape[0]),
                (rgb_frame.shape[1],  rgb_frame.shape[0]),
            )
        if H_uv is not None and uv_frame is not None:
            boxes_u_fused = warp_boxes(
                boxes_u_raw,
                H_uv,
                (uv_frame.shape[1], uv_frame.shape[0]),
                (rgb_frame.shape[1], rgb_frame.shape[0]),
            )

        # Fused detections (RGB + LWIR + UV when UV is active)
        fused_streams = ["rgb", "lwir"] + (["uv"] if uv_res is not None else [])
        if method.lower() == "spatial":
            fused_boxes, fused_scores, fused_details = fuse(
                boxes_r,
                scores_r,
                boxes_l_fused,
                scores_l,
                method=method,
                iou_thr=iou_thr,
                skip_thr=skip_thr,
                boxes3=boxes_u_fused if uv_res is not None else None,
                scores3=scores_u if uv_res is not None else None,
                image_wh=(rgb_frame.shape[1], rgb_frame.shape[0]),
                distance_thr_px=center_dist_px,
                source_names=fused_streams,
                return_details=True,
            )
        else:
            fused_boxes, fused_scores = fuse(
                boxes_r,
                scores_r,
                boxes_l_fused,
                scores_l,
                method=method,
                iou_thr=iou_thr,
                skip_thr=skip_thr,
                boxes3=boxes_u_fused if uv_res is not None else None,
                scores3=scores_u if uv_res is not None else None,
            )
            fused_details = None

        if len(fused_scores):
            keep = fused_scores >= conf_thr
            fused_boxes  = fused_boxes[keep]
            fused_scores = fused_scores[keep]
            if fused_details is not None:
                fused_details = [detail for detail, use_box in zip(fused_details, keep.tolist()) if use_box]
        elif fused_details is not None:
            fused_details = []

        # Build per-band dicts for renderer
        frames_dict = {"rgb": rgb_frame, "lwir": lwir_frame}
        boxes_dict  = {"rgb": boxes_r,   "lwir": boxes_l_raw}
        scores_dict = {"rgb": scores_r,  "lwir": scores_l}
        if uv_frame is not None:
            frames_dict["uv"] = uv_frame
            boxes_dict["uv"]  = boxes_u_raw
            scores_dict["uv"] = scores_u

        # Aligned frames: LWIR/UV warped to RGB canvas for visual verification
        if H_lwir is not None:
            dsize = (rgb_frame.shape[1], rgb_frame.shape[0])
            frames_dict["aligned_lwir"] = cv2.warpPerspective(
                to_bgr(lwir_frame), H_lwir, dsize,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        if H_uv is not None and uv_frame is not None:
            dsize = (rgb_frame.shape[1], rgb_frame.shape[0])
            frames_dict["aligned_uv"] = cv2.warpPerspective(
                to_bgr(uv_frame), H_uv, dsize,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        combined = render_frame(
            frames_dict, boxes_dict, scores_dict,
            fused_boxes, fused_scores, fused_details,
            mode=render_mode,
            frame_idx=frame_idx,
            fusion_method=method,
            fused_streams=fused_streams,
            fusion_margin=fusion_margin,
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
                   choices=["solo", "debug", "sbs", "fused_on_rgb", "grid"],
                   help="render mode: solo | debug | sbs | fused_on_rgb | grid")
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
