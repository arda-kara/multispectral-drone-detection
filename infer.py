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
from ultralytics import RTDETR, YOLO

from fusion import expand_boxes, filter_lwir, filter_rgb, fuse, load_homography, warp_boxes

_STREAM_ORDER = ("rgb", "lwir", "uv")


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(path: str):
    """Load a YOLO or RT-DETR model based on the filename."""
    if "rtdetr" in Path(path).stem.lower():
        return RTDETR(path)
    return YOLO(path)


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


def empty_detections() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)


def parse_fusion_streams(value: object | None) -> list[str] | None:
    """Parse a fusion-stream selector from YAML/CLI/UI input."""
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        raw_items = text.replace(";", ",").split(",")
    elif isinstance(value, np.ndarray):
        raw_items = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raise ValueError(
            "fusion_streams must be null, a comma-separated string, or a list/tuple/set of stream names"
        )

    selected: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        band = str(item).strip().lower()
        if not band:
            continue
        if band not in _STREAM_ORDER:
            valid = ", ".join(_STREAM_ORDER)
            raise ValueError(f"Unsupported fusion stream {band!r}. Expected one of: {valid}")
        if band not in seen:
            selected.append(band)
            seen.add(band)
    return selected


def resolve_fusion_streams(value: object | None, available_streams: list[str]) -> list[str]:
    requested = parse_fusion_streams(value)
    if requested is None:
        return [band for band in _STREAM_ORDER if band in available_streams]
    return [band for band in requested if band in available_streams]


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


def stream_title(stream_key: str) -> str:
    return {
        "rgb": "RGB",
        "lwir": "LWIR",
        "uv": "UV",
        "aligned_lwir": "LWIR aligned",
        "aligned_uv": "UV aligned",
    }.get(stream_key, stream_key.replace("_", " ").upper())


def pick_stream_key(
    frames: dict[str, np.ndarray],
    preferred: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    if preferred is None:
        preferred = _STREAM_ORDER
    for key in preferred:
        if key in frames:
            return key
    return next(iter(frames), None)


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
    fusion_canvas_key: str | None = None,
) -> np.ndarray:
    if fused_streams:
        fusion_label = f"{'·'.join(s.upper() for s in fused_streams)}  {fusion_method.upper()}"
    else:
        fusion_label = f"FUSED  {fusion_method.upper()}"
    base_key = fusion_canvas_key if fusion_canvas_key in frames else pick_stream_key(frames)
    if base_key is None:
        return make_blank_panel((640, 480), text="No frames")

    base_frame = to_bgr(frames[base_key])
    h_ref = base_frame.shape[0]

    if mode == "fused_on_rgb":
        out = draw_fused_boxes(
            base_frame,
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        return add_banner(
            out,
            f"{fusion_label} on {stream_title(base_key)}  f:{frame_idx}",
            color=_COLOR["fused"],
        )

    if mode == "sbs":
        p_rgb = draw_fused_boxes(
            base_frame,
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        p_rgb = add_banner(
            p_rgb,
            f"{fusion_label} on {stream_title(base_key)}  f:{frame_idx}",
            color=_COLOR["fused"],
        )

        compare_key = None
        for candidate in ("aligned_lwir", "lwir", "aligned_uv", "uv"):
            if candidate in frames and candidate != base_key:
                compare_key = candidate
                break

        if compare_key is None:
            return p_rgb

        compare_color = _COLOR["uv"] if "uv" in compare_key else _COLOR["lwir"]
        p_other = draw_fused_boxes(
            to_bgr(frames[compare_key]),
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        p_other = add_banner(p_other, stream_title(compare_key), color=compare_color)
        return np.hstack([p_rgb, resize_to_height(p_other, h_ref)])

    if mode == "debug":
        panels = []
        for band in _STREAM_ORDER:
            if band not in frames:
                continue
            panel = draw_boxes(
                to_bgr(frames[band]),
                boxes[band],
                scores[band],
                label=band,
                color=_COLOR[band],
            )
            panel = add_banner(panel, f"{stream_title(band)} individual", color=_COLOR[band])
            panels.append(resize_to_height(panel, h_ref))
            if len(panels) == 2:
                break

        p_fuse = draw_fused_boxes(
            base_frame,
            fused_boxes,
            fused_scores,
            details=fused_details,
            margin_percent=fusion_margin,
        )
        p_fuse = add_banner(
            p_fuse,
            f"{fusion_label} on {stream_title(base_key)}  f:{frame_idx}",
            color=_COLOR["fused"],
        )
        panels.append(p_fuse)
        return np.hstack(panels)

    if mode == "grid":
        panel_size = (base_frame.shape[1] // 2, base_frame.shape[0] // 2)
        tiles = [
            make_blank_panel(panel_size, text="RGB unavailable")
        ]
        if "rgb" in frames:
            tiles[0] = add_banner(
                cv2.resize(
                    draw_boxes(to_bgr(frames["rgb"]), boxes["rgb"], scores["rgb"], label="rgb", color=_COLOR["rgb"]),
                    panel_size,
                ),
                f"RGB  f:{frame_idx}",
                color=_COLOR["rgb"],
            )

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

        if "lwir" in frames:
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
        else:
            tiles.append(make_blank_panel(panel_size, text="LWIR unavailable"))
        tiles.append(
            add_banner(
                cv2.resize(
                    draw_fused_boxes(
                        base_frame,
                        fused_boxes,
                        fused_scores,
                        details=fused_details,
                        margin_percent=fusion_margin,
                    ),
                    panel_size,
                ),
                f"FUSED on {stream_title(base_key)} ({len(fused_boxes)})  f:{frame_idx}",
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
    for band in _STREAM_ORDER:
        if band not in frames:
            continue
        p = draw_boxes(to_bgr(frames[band]), boxes[band], scores[band],
                       label=band, color=_COLOR[band])
        p = add_banner(p, f"{band.upper()}  f:{frame_idx}", color=_COLOR[band])
        panels.append(resize_to_height(p, h_ref))
    return np.hstack(panels)


def fuse_selected_streams(
    frames: dict[str, np.ndarray],
    boxes_for_fusion: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    requested_streams: object | None,
    *,
    do_fusion: bool,
    method: str,
    iou_thr: float,
    skip_thr: float,
    conf_thr: float,
    center_dist_px: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]] | None, list[str], str | None]:
    available_streams = [band for band in _STREAM_ORDER if band in frames]
    selected_streams = resolve_fusion_streams(requested_streams, available_streams)
    fusion_canvas_key = "rgb" if "rgb" in frames else pick_stream_key(frames, available_streams)
    fused_boxes, fused_scores = empty_detections()

    if not do_fusion or not selected_streams:
        return fused_boxes, fused_scores, [], selected_streams, fusion_canvas_key

    if len(selected_streams) == 1:
        band = selected_streams[0]
        fused_boxes = boxes_for_fusion[band].copy()
        fused_scores = scores[band].copy()
        fused_details: list[dict[str, object]] | None = [
            {"sensor_count": 1, "sensor_total": 1, "sources": [band]}
            for _ in range(len(fused_boxes))
        ]
    else:
        stream_boxes = [boxes_for_fusion[band] for band in selected_streams]
        stream_scores = [scores[band] for band in selected_streams]
        fuse_kwargs: dict[str, object] = {
            "method": method,
            "iou_thr": iou_thr,
            "skip_thr": skip_thr,
        }
        if len(selected_streams) == 3:
            fuse_kwargs["boxes3"] = stream_boxes[2]
            fuse_kwargs["scores3"] = stream_scores[2]

        if method.lower() == "spatial":
            if fusion_canvas_key is None:
                return fused_boxes, fused_scores, [], selected_streams, fusion_canvas_key
            canvas = frames[fusion_canvas_key]
            fused_boxes, fused_scores, fused_details = fuse(
                stream_boxes[0],
                stream_scores[0],
                stream_boxes[1],
                stream_scores[1],
                image_wh=(canvas.shape[1], canvas.shape[0]),
                distance_thr_px=center_dist_px,
                source_names=selected_streams,
                return_details=True,
                **fuse_kwargs,
            )
        else:
            fused_boxes, fused_scores = fuse(
                stream_boxes[0],
                stream_scores[0],
                stream_boxes[1],
                stream_scores[1],
                **fuse_kwargs,
            )
            fused_details = None

    if len(fused_scores):
        keep = fused_scores >= conf_thr
        fused_boxes = fused_boxes[keep]
        fused_scores = fused_scores[keep]
        if fused_details is not None:
            fused_details = [detail for detail, keep_box in zip(fused_details, keep.tolist()) if keep_box]
    elif fused_details is not None:
        fused_details = []

    return fused_boxes, fused_scores, fused_details, selected_streams, fusion_canvas_key


# ── main loop ─────────────────────────────────────────────────────────────────

def run(cfg: dict):
    rgb_model  = load_model(cfg["rgb_model"])  if cfg.get("rgb_model")  else None
    lwir_model = load_model(cfg["lwir_model"]) if cfg.get("lwir_model") else None
    uv_model   = load_model(cfg["uv_model"])   if cfg.get("uv_model")   else None

    rgb_src  = FrameSource(cfg["rgb_source"]) if cfg.get("rgb_source") else None
    lwir_src = FrameSource(cfg["lwir_source"]) if cfg.get("lwir_source") else None
    uv_src   = FrameSource(cfg["uv_source"]) if cfg.get("uv_source") else None
    active_sources = [src for src in (rgb_src, lwir_src, uv_src) if src is not None]
    if not active_sources:
        raise ValueError("At least one video source must be configured for inference.")

    H_lwir = load_homography(cfg.get("lwir_homography"), cfg.get("lwir_homography_offset"))
    H_uv   = load_homography(cfg.get("uv_homography"), cfg.get("uv_homography_offset"))

    do_fusion   = bool(cfg.get("do_fusion", True))
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
    output_fps    = active_sources[0].fps

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

        ok_r, rgb_frame  = rgb_src.read() if rgb_src else (False, None)
        ok_l, lwir_frame = lwir_src.read() if lwir_src else (False, None)
        ok_u, uv_frame   = uv_src.read() if uv_src else (False, None)
        if not ok_r and not ok_l and not ok_u:
            break

        # Inference
        rgb_res  = rgb_model(rgb_frame,   **infer_kwargs)[0] if rgb_model and ok_r else None
        lwir_res = lwir_model(lwir_frame, **infer_kwargs)[0] if lwir_model and ok_l else None
        uv_res   = uv_model(uv_frame,     **infer_kwargs)[0] if uv_model and ok_u else None

        # Individual detections
        boxes_r, scores_r = filter_rgb(rgb_res) if rgb_res is not None else empty_detections()
        boxes_l_raw, scores_l = filter_lwir(lwir_res) if lwir_res is not None else empty_detections()
        boxes_u_raw, scores_u = filter_lwir(uv_res) if uv_res is not None else empty_detections()

        frames_dict: dict[str, np.ndarray] = {}
        boxes_dict: dict[str, np.ndarray] = {}
        scores_dict: dict[str, np.ndarray] = {}
        if ok_r:
            frames_dict["rgb"] = rgb_frame
            boxes_dict["rgb"] = boxes_r
            scores_dict["rgb"] = scores_r
        if ok_l:
            frames_dict["lwir"] = lwir_frame
            boxes_dict["lwir"] = boxes_l_raw
            scores_dict["lwir"] = scores_l
        if ok_u:
            frames_dict["uv"] = uv_frame
            boxes_dict["uv"] = boxes_u_raw
            scores_dict["uv"] = scores_u

        boxes_for_fusion = {band: boxes.copy() for band, boxes in boxes_dict.items()}

        # Warp LWIR and UV boxes into RGB pixel space before fusion
        if H_lwir is not None and ok_l and ok_r:
            boxes_for_fusion["lwir"] = warp_boxes(
                boxes_l_raw,
                H_lwir,
                (lwir_frame.shape[1], lwir_frame.shape[0]),
                (rgb_frame.shape[1],  rgb_frame.shape[0]),
            )
        if H_uv is not None and ok_u and ok_r:
            boxes_for_fusion["uv"] = warp_boxes(
                boxes_u_raw,
                H_uv,
                (uv_frame.shape[1], uv_frame.shape[0]),
                (rgb_frame.shape[1], rgb_frame.shape[0]),
            )

        fused_boxes, fused_scores, fused_details, fused_streams, fusion_canvas_key = fuse_selected_streams(
            frames_dict,
            boxes_for_fusion,
            scores_dict,
            cfg.get("fusion_streams"),
            do_fusion=do_fusion,
            method=method,
            iou_thr=iou_thr,
            skip_thr=skip_thr,
            conf_thr=conf_thr,
            center_dist_px=center_dist_px,
        )

        # Aligned frames: LWIR/UV warped to RGB canvas for visual verification
        if H_lwir is not None and ok_l and ok_r:
            dsize = (rgb_frame.shape[1], rgb_frame.shape[0])
            frames_dict["aligned_lwir"] = cv2.warpPerspective(
                to_bgr(lwir_frame), H_lwir, dsize,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        if H_uv is not None and ok_u and ok_r:
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
            fusion_canvas_key=fusion_canvas_key,
        )

        if writer is None:
            h, w = combined.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h),
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

    for src in (rgb_src, lwir_src, uv_src):
        if src is not None:
            src.release()
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
    p.add_argument(
        "--fusion-streams",
        metavar="STREAMS",
        help="comma-separated fusion subset, e.g. rgb,lwir or lwir,uv (default: all available)",
    )
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
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    if args.fusion_streams is not None:
        cfg["fusion_streams"] = args.fusion_streams

    run(cfg)
