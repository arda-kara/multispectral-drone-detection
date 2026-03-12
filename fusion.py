"""
Bounding-box fusion for RGB + LWIR drone detection.

Public API
----------
filter_rgb(results)             -> (boxes, scores)   # drone-only, normalised
fuse(boxes1, scores1,
     boxes2, scores2,
     method, **kwargs)          -> (boxes, scores)
"""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np


# ── Homography utilities ──────────────────────────────────────────────────────

def parse_offset(offset: object | None) -> tuple[float, float]:
    """Parse a homography offset as (dx, dy) pixels."""
    if offset is None:
        return 0.0, 0.0

    if isinstance(offset, str):
        text = offset.strip()
        if not text:
            return 0.0, 0.0
        parts = [part.strip() for part in text.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"Offset string must be 'dx,dy', got {offset!r}")
        return float(parts[0]), float(parts[1])

    if isinstance(offset, dict):
        return float(offset.get("x", 0.0)), float(offset.get("y", 0.0))

    if isinstance(offset, np.ndarray):
        offset = offset.tolist()

    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        return float(offset[0]), float(offset[1])

    raise ValueError(f"Unsupported offset format: {offset!r}")


def apply_homography_offset(H: np.ndarray, offset: object | None) -> np.ndarray:
    """Apply a post-homography pixel translation on the destination canvas."""
    dx, dy = parse_offset(offset)
    if dx == 0.0 and dy == 0.0:
        return H.astype(np.float32)

    translation = np.array(
        [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (translation @ H).astype(np.float32)


def load_homography(path: str | None, offset: object | None = None) -> np.ndarray | None:
    """Load a (3,3) float32 homography matrix from a .npy file.

    Returns None when path is empty/None or the file cannot be read,
    which callers treat as "no warp" (identity fallback).
    """
    if not path:
        return None
    try:
        arr = np.load(path)
        assert arr.shape == (3, 3), f"Expected shape (3, 3), got {arr.shape}"
        return apply_homography_offset(arr.astype(np.float32), offset)
    except Exception as exc:
        print(f"[homography] Could not load {path!r}: {exc}", file=sys.stderr)
        return None


def warp_boxes(
    boxes_norm: np.ndarray,
    H: np.ndarray,
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> np.ndarray:
    """Warp normalised xyxy boxes from src frame space to dst frame space.

    Parameters
    ----------
    boxes_norm : (N, 4) float32 — normalised [x1, y1, x2, y2] in src space
    H          : (3, 3) float32 — homography mapping src pixels → dst pixels
    src_wh     : (width, height) of the source frame
    dst_wh     : (width, height) of the destination (RGB) frame

    Returns
    -------
    (N, 4) float32 — normalised [x1, y1, x2, y2] in dst space, clipped to [0, 1]
    """
    if len(boxes_norm) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    sw, sh = src_wh
    dw, dh = dst_wh

    # Denormalise to src pixel coordinates
    boxes_px = boxes_norm.astype(np.float32) * np.array([sw, sh, sw, sh], dtype=np.float32)

    # Build (N*4, 1, 2) array of the four corners per box for perspectiveTransform
    n = len(boxes_px)
    corners = np.empty((n * 4, 1, 2), dtype=np.float32)
    corners[0::4, 0] = boxes_px[:, [0, 1]]  # top-left
    corners[1::4, 0] = boxes_px[:, [2, 1]]  # top-right
    corners[2::4, 0] = boxes_px[:, [2, 3]]  # bottom-right
    corners[3::4, 0] = boxes_px[:, [0, 3]]  # bottom-left

    # Apply homography
    warped = cv2.perspectiveTransform(corners, H).reshape(n, 4, 2)

    # Fit axis-aligned bounding box to the warped quadrilateral
    x1 = warped[:, :, 0].min(axis=1)
    y1 = warped[:, :, 1].min(axis=1)
    x2 = warped[:, :, 0].max(axis=1)
    y2 = warped[:, :, 1].max(axis=1)

    out = np.stack([x1 / dw, y1 / dh, x2 / dw, y2 / dh], axis=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def expand_boxes(boxes: np.ndarray, margin_percent: float = 0.10) -> np.ndarray:
    """Expand one or many normalised xyxy boxes by a percentage from their centre."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0 or margin_percent <= 0:
        return boxes.copy().astype(np.float32)

    is_single = boxes.ndim == 1
    if is_single:
        boxes = boxes[None, :]

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    centers_x = boxes[:, 0] + (widths * 0.5)
    centers_y = boxes[:, 1] + (heights * 0.5)

    scaled_w = widths * (1.0 + margin_percent)
    scaled_h = heights * (1.0 + margin_percent)

    expanded = np.stack(
        [
            centers_x - (scaled_w * 0.5),
            centers_y - (scaled_h * 0.5),
            centers_x + (scaled_w * 0.5),
            centers_y + (scaled_h * 0.5),
        ],
        axis=1,
    )
    expanded = np.clip(expanded, 0.0, 1.0).astype(np.float32)
    return expanded[0] if is_single else expanded


# ── RGB output filter ─────────────────────────────────────────────────────────

# Roboflow dataset class indices
_RGB_DRONE_CLS = 1   # 0=bird, 1=drone


def filter_rgb(results) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract drone detections from an Ultralytics RGB model result,
    normalising boxes to [0, 1] and remapping the class to 0.

    Parameters
    ----------
    results : ultralytics.engine.results.Results
        Single-image result object returned by model(frame).

    Returns
    -------
    boxes  : np.ndarray, shape (N, 4)  — normalised [x1, y1, x2, y2]
    scores : np.ndarray, shape (N,)
    """
    boxes_data = results.boxes
    if boxes_data is None or len(boxes_data) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    cls = boxes_data.cls.cpu().numpy().astype(int)
    mask = cls == _RGB_DRONE_CLS

    if mask.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    boxes  = boxes_data.xyxyn.cpu().numpy()[mask]   # normalised xyxy
    scores = boxes_data.conf.cpu().numpy()[mask]
    return boxes.astype(np.float32), scores.astype(np.float32)


def filter_lwir(results) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract drone detections from an Ultralytics LWIR model result.
    LWIR model already has a single class (drone=0), so no remapping needed.
    """
    boxes_data = results.boxes
    if boxes_data is None or len(boxes_data) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    boxes  = boxes_data.xyxyn.cpu().numpy()
    scores = boxes_data.conf.cpu().numpy()
    return boxes.astype(np.float32), scores.astype(np.float32)


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box [x1,y1,x2,y2] and an array of boxes (N,4)."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


# ── Weighted Boxes Fusion ─────────────────────────────────────────────────────

def _wbf(
    boxes_list:  list[np.ndarray],
    scores_list: list[np.ndarray],
    iou_thr: float = 0.55,
    skip_thr: float = 0.0,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted Boxes Fusion (Solovyev et al., 2021).

    Parameters
    ----------
    boxes_list  : list of (N_i, 4) arrays, normalised [x1,y1,x2,y2]
    scores_list : list of (N_i,)   confidence arrays
    iou_thr     : cluster merge threshold
    skip_thr    : discard boxes below this confidence
    weights     : per-model weight (default: equal)

    Returns
    -------
    fused_boxes  : (M, 4)
    fused_scores : (M,)
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models
    weights = np.array(weights, dtype=np.float32)

    # Flatten all boxes; track source model index so the final score can be
    # divided by the number of *contributing* models, not the total count.
    # This prevents solo-stream detections (e.g. a UV-only thermal hit) from
    # being penalised down to 1/n_models of their original confidence.
    all_boxes, all_scores, all_model_idxs = [], [], []
    for i, (bxs, scs) in enumerate(zip(boxes_list, scores_list)):
        mask = scs >= skip_thr
        all_boxes.append(bxs[mask])
        all_scores.append(scs[mask] * weights[i])
        all_model_idxs.append(np.full(mask.sum(), i, dtype=np.int32))

    if all(len(b) == 0 for b in all_boxes):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_boxes       = np.concatenate(all_boxes,       axis=0)
    all_scores      = np.concatenate(all_scores,      axis=0)
    all_model_idxs  = np.concatenate(all_model_idxs,  axis=0)

    # sort by descending weighted score
    order = np.argsort(-all_scores)
    all_boxes      = all_boxes[order]
    all_scores     = all_scores[order]
    all_model_idxs = all_model_idxs[order]

    # cluster representatives (weighted mean box per cluster)
    cluster_boxes:  list[list[np.ndarray]] = []
    cluster_scores: list[list[float]]      = []
    cluster_models: list[list[int]]        = []  # source model index per box
    rep_boxes: list[np.ndarray] = []

    for box, score, midx in zip(all_boxes, all_scores, all_model_idxs):
        if not rep_boxes:
            cluster_boxes.append([box])
            cluster_scores.append([score])
            cluster_models.append([int(midx)])
            rep_boxes.append(box.copy())
            continue

        ious = _iou(box, np.array(rep_boxes))
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr:
            cluster_boxes[best].append(box)
            cluster_scores[best].append(score)
            cluster_models[best].append(int(midx))
            # update representative as score-weighted average
            bxs = np.array(cluster_boxes[best])
            wts = np.array(cluster_scores[best])
            rep_boxes[best] = (bxs * wts[:, None]).sum(0) / wts.sum()
        else:
            cluster_boxes.append([box])
            cluster_scores.append([score])
            cluster_models.append([int(midx)])
            rep_boxes.append(box.copy())

    # Produce final boxes.
    # Score = sum(scores) / n_contributing, where n_contributing is the number
    # of *unique* source models in the cluster (not the total number of models).
    # A solo detection keeps its confidence; cross-stream agreement boosts it.
    fused_boxes, fused_scores = [], []
    for c_bxs, c_scs, c_mdls in zip(cluster_boxes, cluster_scores, cluster_models):
        bxs = np.array(c_bxs)
        scs = np.array(c_scs)
        n_contributing = len(set(c_mdls))
        fused_boxes.append((bxs * scs[:, None]).sum(0) / scs.sum())
        fused_scores.append(min(1.0, scs.sum() / n_contributing))

    return (
        np.clip(np.array(fused_boxes,  dtype=np.float32), 0, 1),
        np.clip(np.array(fused_scores, dtype=np.float32), 0, 1),
    )


def _update_spatial_cluster(cluster: dict) -> None:
    """Refresh the cluster's weighted centre after a new member is added."""
    member_scores = np.array(
        [member["weighted_score"] for member in cluster["members"]],
        dtype=np.float32,
    )
    if member_scores.sum() <= 0:
        member_scores = np.ones(len(cluster["members"]), dtype=np.float32)

    centers_x = np.array([member["cx_px"] for member in cluster["members"]], dtype=np.float32)
    centers_y = np.array([member["cy_px"] for member in cluster["members"]], dtype=np.float32)
    cluster["cx_px"] = float((centers_x * member_scores).sum() / member_scores.sum())
    cluster["cy_px"] = float((centers_y * member_scores).sum() / member_scores.sum())


def _spatial(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    image_wh: tuple[int, int],
    distance_thr_px: float = 60.0,
    skip_thr: float = 0.0,
    weights: list[float] | None = None,
    source_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Center-distance clustering in the shared RGB canvas."""
    if weights is None:
        weights = [1.0] * len(boxes_list)
    if source_names is None:
        source_names = [f"stream{i + 1}" for i in range(len(boxes_list))]
    if len(weights) != len(boxes_list):
        raise ValueError("weights must have the same length as boxes_list")
    if len(source_names) != len(boxes_list):
        raise ValueError("source_names must have the same length as boxes_list")

    frame_w, frame_h = image_wh
    detections: list[dict[str, object]] = []

    for model_idx, (boxes, scores, weight, source_name) in enumerate(
        zip(boxes_list, scores_list, weights, source_names)
    ):
        keep = scores >= skip_thr
        for box, score in zip(boxes[keep], scores[keep]):
            cx_px = float(((box[0] + box[2]) * 0.5) * frame_w)
            cy_px = float(((box[1] + box[3]) * 0.5) * frame_h)
            detections.append(
                {
                    "box": box.astype(np.float32),
                    "score": float(score),
                    "weighted_score": float(score * weight),
                    "source": source_name,
                    "model_idx": model_idx,
                    "cx_px": cx_px,
                    "cy_px": cy_px,
                }
            )

    if not detections:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )

    detections.sort(key=lambda det: float(det["weighted_score"]), reverse=True)
    clusters: list[dict[str, object]] = []

    for det in detections:
        best_cluster = None
        best_dist = float("inf")

        for idx, cluster in enumerate(clusters):
            dist = math.hypot(
                float(det["cx_px"]) - float(cluster["cx_px"]),
                float(det["cy_px"]) - float(cluster["cy_px"]),
            )
            if dist <= distance_thr_px and dist < best_dist:
                best_cluster = idx
                best_dist = dist

        if best_cluster is None:
            clusters.append(
                {
                    "cx_px": float(det["cx_px"]),
                    "cy_px": float(det["cy_px"]),
                    "members": [det],
                }
            )
            continue

        clusters[best_cluster]["members"].append(det)
        _update_spatial_cluster(clusters[best_cluster])

    fused_boxes, fused_scores, fused_details = [], [], []
    sensor_total = len(source_names)

    for cluster in clusters:
        members: list[dict[str, object]] = cluster["members"]
        member_boxes = np.array([member["box"] for member in members], dtype=np.float32)
        member_scores = np.array(
            [member["weighted_score"] for member in members],
            dtype=np.float32,
        )
        if member_scores.sum() <= 0:
            member_scores = np.ones(len(members), dtype=np.float32)

        fused_box = (member_boxes * member_scores[:, None]).sum(0) / member_scores.sum()

        per_source_score: dict[str, float] = {}
        for member in members:
            source_name = str(member["source"])
            score = float(np.clip(member["weighted_score"], 0.0, 1.0))
            per_source_score[source_name] = max(per_source_score.get(source_name, 0.0), score)

        source_scores = np.array(list(per_source_score.values()), dtype=np.float32)
        fused_boxes.append(fused_box)
        fused_scores.append(float(source_scores.mean()) if len(source_scores) else 0.0)
        fused_details.append(
            {
                "sensor_count": len(per_source_score),
                "sensor_total": sensor_total,
                "sources": sorted(per_source_score),
            }
        )

    return (
        np.clip(np.array(fused_boxes, dtype=np.float32), 0, 1),
        np.clip(np.array(fused_scores, dtype=np.float32), 0, 1),
        fused_details,
    )


# ── Bayesian fusion ───────────────────────────────────────────────────────────

def _bayesian(
    boxes1:  np.ndarray,
    scores1: np.ndarray,
    boxes2:  np.ndarray,
    scores2: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bayesian independent-sensor fusion.

    For matched boxes (IoU > iou_thr):
        P(drone | E1, E2) = s1*s2 / (s1*s2 + (1-s1)*(1-s2))
        box = weighted average by individual scores

    For unmatched boxes: pass through with original score.

    Assumes uniform prior and independent sensors.
    """
    if len(boxes1) == 0 and len(boxes2) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
    if len(boxes1) == 0:
        return boxes2.copy(), scores2.copy()
    if len(boxes2) == 0:
        return boxes1.copy(), scores1.copy()

    matched1 = set()
    matched2 = set()
    fused_boxes, fused_scores = [], []

    # match greedily by best IoU
    iou_matrix = np.stack([_iou(b, boxes2) for b in boxes1])   # (N1, N2)
    while True:
        if iou_matrix.max() < iou_thr:
            break
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        s1, s2 = float(scores1[i]), float(scores2[j])
        denom = s1 * s2 + (1 - s1) * (1 - s2)
        combined_score = (s1 * s2 / denom) if denom > 0 else 0.0
        combined_box = (boxes1[i] * s1 + boxes2[j] * s2) / (s1 + s2)
        fused_boxes.append(combined_box)
        fused_scores.append(combined_score)
        matched1.add(i)
        matched2.add(j)
        iou_matrix[i, :] = -1
        iou_matrix[:, j] = -1

    # unmatched boxes pass through
    for i in range(len(boxes1)):
        if i not in matched1:
            fused_boxes.append(boxes1[i])
            fused_scores.append(float(scores1[i]))
    for j in range(len(boxes2)):
        if j not in matched2:
            fused_boxes.append(boxes2[j])
            fused_scores.append(float(scores2[j]))

    return (
        np.clip(np.array(fused_boxes,  dtype=np.float32), 0, 1),
        np.clip(np.array(fused_scores, dtype=np.float32), 0, 1),
    )


# ── Public interface ──────────────────────────────────────────────────────────

def fuse(
    boxes1:  np.ndarray,
    scores1: np.ndarray,
    boxes2:  np.ndarray,
    scores2: np.ndarray,
    method:  str = "wbf",
    iou_thr: float = 0.55,
    skip_thr: float = 0.01,
    weights: list[float] | None = None,
    boxes3:  np.ndarray | None = None,
    scores3: np.ndarray | None = None,
    image_wh: tuple[int, int] | None = None,
    distance_thr_px: float = 60.0,
    source_names: list[str] | None = None,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict[str, object]] | None]:
    """
    Fuse bounding boxes from two or three drone-detection models.

    Parameters
    ----------
    boxes1, boxes2       : (N, 4) normalised [x1, y1, x2, y2]
    scores1, scores2     : (N,)   confidence in [0, 1]
    boxes3, scores3      : optional third stream (e.g. UV)
    method               : "wbf" | "bayesian" | "spatial"
    iou_thr              : IoU threshold for matching / clustering
    skip_thr             : discard boxes below this confidence
    weights              : per-model weights, default equal
    image_wh             : destination canvas size for spatial fusion
    distance_thr_px      : spatial cluster threshold in pixels
    source_names         : optional stream names used in returned metadata
    return_details       : return per-cluster sensor metadata when available

    Returns
    -------
    fused_boxes  : (M, 4)
    fused_scores : (M,)
    """
    has_third = boxes3 is not None and scores3 is not None

    method = method.lower()
    if method == "wbf":
        bl = [boxes1, boxes2]
        sl = [scores1, scores2]
        if has_third:
            bl.append(boxes3)
            sl.append(scores3)
        boxes, scores = _wbf(bl, sl, iou_thr=iou_thr, skip_thr=skip_thr, weights=weights)
        return (boxes, scores, None) if return_details else (boxes, scores)

    if method == "bayesian":
        b, s = _bayesian(boxes1, scores1, boxes2, scores2, iou_thr=iou_thr)
        if has_third:
            b, s = _bayesian(b, s, boxes3, scores3, iou_thr=iou_thr)
        return (b, s, None) if return_details else (b, s)

    if method == "spatial":
        if image_wh is None:
            raise ValueError("image_wh is required for spatial fusion")
        boxes_list = [boxes1, boxes2]
        scores_list = [scores1, scores2]
        if has_third:
            boxes_list.append(boxes3)
            scores_list.append(scores3)
        b, s, details = _spatial(
            boxes_list,
            scores_list,
            image_wh=image_wh,
            distance_thr_px=distance_thr_px,
            skip_thr=skip_thr,
            weights=weights,
            source_names=source_names,
        )
        return (b, s, details) if return_details else (b, s)

    raise ValueError(f"Unknown fusion method: {method!r}. Use 'wbf', 'bayesian', or 'spatial'.")
