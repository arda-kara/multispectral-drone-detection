"""Kalman-filter-based multi-object tracker for LWIR drone detections."""

from __future__ import annotations

import numpy as np


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of normalised xyxy boxes. Returns (M, N)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ix1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    iy1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    ix2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    iy2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _greedy_match(iou_mat: np.ndarray) -> tuple[list[int], list[int]]:
    """Greedy fallback matching when scipy is unavailable."""
    rows, cols, mat = [], [], iou_mat.copy()
    while mat.size > 0 and mat.max() > 0:
        r, c = np.unravel_index(np.argmax(mat), mat.shape)
        rows.append(int(r))
        cols.append(int(c))
        mat[r, :] = -1
        mat[:, c] = -1
    return rows, cols


# ── Box conversion helpers ────────────────────────────────────────────────────

def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    return np.array([
        (box[0] + box[2]) * 0.5,
        (box[1] + box[3]) * 0.5,
        box[2] - box[0],
        box[3] - box[1],
    ], dtype=np.float64)


def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy = float(state[0]), float(state[1])
    w,  h  = max(float(state[2]), 0.0), max(float(state[3]), 0.0)
    return np.clip(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0.0, 1.0,
    ).astype(np.float32)


# ── Single-track Kalman filter ────────────────────────────────────────────────

class _KalmanTrack:
    """
    Constant-velocity Kalman filter for one drone target.

    State       : [cx, cy, w, h, vx, vy]  (6-dim, normalised image coordinates)
    Observation : [cx, cy, w, h]           (4-dim)
    """

    def __init__(self, box: np.ndarray, score: float, track_id: int):
        self.track_id = track_id
        self.hits     = 1
        self.misses   = 0
        self.age      = 1
        self.score    = float(score)

        # State initialised from first detection; velocity = 0
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:4] = _xyxy_to_cxcywh(box)

        # Covariance: low positional uncertainty, high velocity uncertainty
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1.0, 1.0])

        # State transition: position advances by velocity
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        # Observation matrix: observe cx, cy, w, h
        self.H = np.zeros((4, 6), dtype=np.float64)
        np.fill_diagonal(self.H, 1.0)

        # Process noise: small for position/size, larger for velocity
        self.Q = np.diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4])

        # Measurement noise: centre more trusted than size
        self.R = np.diag([2e-4, 2e-4, 1e-4, 1e-4])

    def predict(self) -> None:
        self.x  = self.F @ self.x
        self.P  = self.F @ self.P @ self.F.T + self.Q
        self.age    += 1
        self.misses += 1

    def update(self, box: np.ndarray, score: float) -> None:
        z = _xyxy_to_cxcywh(box)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x  = self.x + K @ (z - self.H @ self.x)
        self.P  = (np.eye(6) - K @ self.H) @ self.P
        self.hits   += 1
        self.misses  = 0
        self.score   = float(score)

    @property
    def box(self) -> np.ndarray:
        return _cxcywh_to_xyxy(self.x)


# ── Multi-track manager ───────────────────────────────────────────────────────

class LWIRKalmanTracker:
    """
    Frame-by-frame Kalman tracker for LWIR drone detections.

    Behaviour
    ---------
    - A detection must appear in min_hits consecutive frames before its track
      is confirmed and included in the output (suppresses single-frame FPs).
    - A confirmed track that the detector misses is kept alive for max_age
      frames, outputting the Kalman-predicted box with decaying confidence
      (gap filling).
    - Confidence of a fill-in = last_detection_score × conf_decay^misses.

    Parameters
    ----------
    max_age    : delete a track after this many consecutive missed frames
    min_hits   : require this many consecutive detections before outputting
    iou_thr    : minimum IoU to associate a detection to an existing track
    conf_decay : confidence multiplier applied per missed fill-in frame
    """

    def __init__(
        self,
        max_age:    int   = 3,
        min_hits:   int   = 2,
        iou_thr:    float = 0.25,
        conf_decay: float = 0.8,
    ):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thr    = iou_thr
        self.conf_decay = conf_decay
        self.tracks: list[_KalmanTrack] = []
        self._next_id   = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        boxes:  np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process one frame of LWIR detections.

        Parameters
        ----------
        boxes  : (N, 4) normalised xyxy — may be empty
        scores : (N,)

        Returns
        -------
        out_boxes  : (M, 4) — confirmed detections + Kalman fill-ins
        out_scores : (M,)
        """
        # 1. Predict all tracks one step forward
        for trk in self.tracks:
            trk.predict()

        # 2. Associate detections to tracks (Hungarian or greedy)
        matched_trks: set[int] = set()
        matched_dets: set[int] = set()

        if self.tracks and len(boxes) > 0:
            pred_boxes = np.array([t.box for t in self.tracks])
            iou_mat = _iou_matrix(pred_boxes, boxes)
            try:
                from scipy.optimize import linear_sum_assignment
                row_idx, col_idx = linear_sum_assignment(-iou_mat)
            except ImportError:
                row_idx, col_idx = _greedy_match(iou_mat)
            for r, c in zip(row_idx, col_idx):
                if iou_mat[r, c] >= self.iou_thr:
                    self.tracks[r].update(boxes[c], float(scores[c]))
                    matched_trks.add(r)
                    matched_dets.add(c)

        # 3. Spawn new tracks for unmatched detections
        for i in range(len(boxes)):
            if i not in matched_dets:
                self.tracks.append(
                    _KalmanTrack(boxes[i], float(scores[i]), self._next_id)
                )
                self._next_id += 1

        # 4. Prune dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # 5. Emit output for confirmed tracks only
        out_boxes, out_scores = [], []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                fill_score = trk.score * (self.conf_decay ** trk.misses)
                out_boxes.append(trk.box)
                out_scores.append(float(np.clip(fill_score, 0.0, 1.0)))

        if not out_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return (
            np.array(out_boxes,  dtype=np.float32),
            np.array(out_scores, dtype=np.float32),
        )
