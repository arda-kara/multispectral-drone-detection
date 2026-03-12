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

import numpy as np


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

    # flatten all boxes, tag with model index and weight
    all_boxes, all_scores, all_weights = [], [], []
    for i, (bxs, scs) in enumerate(zip(boxes_list, scores_list)):
        mask = scs >= skip_thr
        all_boxes.append(bxs[mask])
        all_scores.append(scs[mask] * weights[i])
        all_weights.append(np.full(mask.sum(), weights[i]))

    if all(len(b) == 0 for b in all_boxes):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_boxes   = np.concatenate(all_boxes,   axis=0)
    all_scores  = np.concatenate(all_scores,  axis=0)
    all_weights = np.concatenate(all_weights, axis=0)

    # sort by descending weighted score
    order = np.argsort(-all_scores)
    all_boxes   = all_boxes[order]
    all_scores  = all_scores[order]
    all_weights = all_weights[order]

    # cluster representatives (weighted mean box per cluster)
    cluster_boxes:   list[list[np.ndarray]] = []
    cluster_scores:  list[list[float]]      = []
    cluster_weights: list[list[float]]      = []
    rep_boxes: list[np.ndarray] = []   # weighted mean of each cluster

    for box, score, w in zip(all_boxes, all_scores, all_weights):
        if not rep_boxes:
            cluster_boxes.append([box])
            cluster_scores.append([score])
            cluster_weights.append([w])
            rep_boxes.append(box.copy())
            continue

        ious = _iou(box, np.array(rep_boxes))
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr:
            cluster_boxes[best].append(box)
            cluster_scores[best].append(score)
            cluster_weights[best].append(w)
            # update representative as weighted average
            bxs = np.array(cluster_boxes[best])
            wts = np.array(cluster_scores[best])   # weight by score
            rep_boxes[best] = (bxs * wts[:, None]).sum(0) / wts.sum()
        else:
            cluster_boxes.append([box])
            cluster_scores.append([score])
            cluster_weights.append([w])
            rep_boxes.append(box.copy())

    # produce final boxes
    fused_boxes, fused_scores = [], []
    for c_bxs, c_scs, c_wts in zip(cluster_boxes, cluster_scores, cluster_weights):
        bxs = np.array(c_bxs)
        scs = np.array(c_scs)
        fused_boxes.append((bxs * scs[:, None]).sum(0) / scs.sum())
        # score = sum of weighted scores / number of models contributing
        fused_scores.append(scs.sum() / n_models)

    return (
        np.clip(np.array(fused_boxes,  dtype=np.float32), 0, 1),
        np.clip(np.array(fused_scores, dtype=np.float32), 0, 1),
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse bounding boxes from two or three drone-detection models.

    Parameters
    ----------
    boxes1, boxes2       : (N, 4) normalised [x1, y1, x2, y2]
    scores1, scores2     : (N,)   confidence in [0, 1]
    boxes3, scores3      : optional third stream (e.g. UV)
    method               : "wbf" | "bayesian"
    iou_thr              : IoU threshold for matching / clustering
    skip_thr             : (WBF only) discard boxes below this confidence
    weights              : per-model weights, default equal

    Returns
    -------
    fused_boxes  : (M, 4)
    fused_scores : (M,)
    """
    has_third = boxes3 is not None and scores3 is not None and len(boxes3) >= 0

    method = method.lower()
    if method == "wbf":
        bl = [boxes1, boxes2]
        sl = [scores1, scores2]
        if has_third:
            bl.append(boxes3)
            sl.append(scores3)
        return _wbf(bl, sl, iou_thr=iou_thr, skip_thr=skip_thr, weights=weights)

    if method == "bayesian":
        b, s = _bayesian(boxes1, scores1, boxes2, scores2, iou_thr=iou_thr)
        if has_third:
            b, s = _bayesian(b, s, boxes3, scores3, iou_thr=iou_thr)
        return b, s

    raise ValueError(f"Unknown fusion method: {method!r}. Use 'wbf' or 'bayesian'.")
