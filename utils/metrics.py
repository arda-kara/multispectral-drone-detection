"""Evaluation metrics for object detection."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from collections import defaultdict
import logging

logger: logging.Logger = logging.getLogger(__name__)


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: Tensor of shape (N, 4) [x1, y1, x2, y2]
        box2: Tensor of shape (M, 4) [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M)
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # (N, 1, 4)
    box2 = box2.unsqueeze(0)  # (1, M, 4)

    # Calculate intersection
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Calculate union
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def calculate_ap(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    iou_threshold: float = 0.5,
    num_classes: int = 1,
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate Average Precision (AP) for object detection.

    Args:
        predictions: List of predictions for each image [N, 6] [x1, y1, x2, y2, conf, class]
        targets: List of ground truth boxes for each image [M, 5] [x1, y1, x2, y2, class]
        iou_threshold: IoU threshold for true positives
        num_classes: Number of classes

    Returns:
        Tuple of (mAP, per_class_AP)
    """
    ap_per_class: Dict[int, float] = {}

    for class_id in range(num_classes):
        # Collect predictions and targets for this class
        class_preds: List[torch.Tensor] = []
        class_targets: List[torch.Tensor] = []

        for pred, target in zip(predictions, targets):
            # Filter predictions by class
            if len(pred) > 0:
                pred_class_mask: torch.Tensor = pred[:, 5] == class_id
                class_preds.append(pred[pred_class_mask])
            else:
                class_preds.append(torch.empty((0, 6)))

            # Filter targets by class
            if len(target) > 0:
                target_class_mask: torch.Tensor = target[:, 4] == class_id
                class_targets.append(target[target_class_mask])
            else:
                class_targets.append(torch.empty((0, 5)))

        # Flatten lists
        all_preds: torch.Tensor = torch.cat(class_preds, dim=0)
        all_targets: List[torch.Tensor] = class_targets

        if len(all_preds) == 0 and all(len(t) == 0 for t in all_targets):
            ap_per_class[class_id] = 0.0
            continue

        # Sort predictions by confidence
        if len(all_preds) > 0:
            sorted_indices: torch.Tensor = torch.argsort(
                all_preds[:, 4], descending=True
            )
            all_preds = all_preds[sorted_indices]
        else:
            all_preds = torch.empty((0, 6))

        # Calculate true positives and false positives
        tp: List[float] = []
        fp: List[float] = []

        # Track matched targets for each image
        target_matched: List[set] = [set() for _ in all_targets]

        for pred_idx, pred in enumerate(all_preds):
            # Find which image this prediction belongs to
            # For simplicity, assume we can match based on position
            best_iou: float = 0.0
            best_target_idx: int = -1
            best_image_idx: int = -1

            for img_idx, targets in enumerate(all_targets):
                if len(targets) == 0:
                    continue

                targets_tensor: torch.Tensor = targets
                ious: torch.Tensor = calculate_iou(
                    pred[:4].unsqueeze(0), targets_tensor[:, :4]
                )
                max_iou: torch.Tensor = ious.max()
                argmax_iou: int = ious.argmax().item()

                if max_iou > best_iou:
                    best_iou = max_iou.item()
                    best_target_idx = argmax_iou
                    best_image_idx = img_idx

            # Check if it's a true positive
            if best_iou >= iou_threshold and best_image_idx >= 0:
                if best_target_idx not in target_matched[best_image_idx]:
                    tp.append(1.0)
                    fp.append(0.0)
                    target_matched[best_image_idx].add(best_target_idx)
                else:
                    fp.append(1.0)
                    tp.append(0.0)
            else:
                fp.append(1.0)
                tp.append(0.0)

        # Calculate precision-recall curve
        tp = np.array(tp)
        fp = np.array(fp)

        if len(tp) == 0:
            ap_per_class[class_id] = 0.0
            continue

        tp_cumsum: np.ndarray = np.cumsum(tp)
        fp_cumsum: np.ndarray = np.cumsum(fp)

        recalls: np.ndarray = tp_cumsum / (sum(len(t) for t in all_targets) + 1e-6)
        precisions: np.ndarray = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Add sentinel values at the beginning and end
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))

        # Calculate AP using 11-point interpolation
        ap: float = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        ap_per_class[class_id] = ap

    # Calculate mAP
    mAP: float = float(np.mean(list(ap_per_class.values())))

    return mAP, ap_per_class


def calculate_map(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    iou_thresholds: List[float] = [
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    num_classes: int = 1,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Calculate mAP at multiple IoU thresholds (COCO-style).

    Args:
        predictions: List of predictions for each image [N, 6] [x1, y1, x2, y2, conf, class]
        targets: List of ground truth boxes for each image [M, 5] [x1, y1, x2, y2, class]
        iou_thresholds: List of IoU thresholds
        num_classes: Number of classes

    Returns:
        Tuple of (mAP_0.5:0.95, mAP_0.5, mAP_per_iou)
    """
    ap_results: Dict[float, float] = {}

    for iou_threshold in iou_thresholds:
        mAP, _ = calculate_ap(predictions, targets, iou_threshold, num_classes)
        ap_results[iou_threshold] = mAP

    # Calculate mAP@0.5:0.95
    map_05_95: float = float(np.mean(list(ap_results.values())))
    map_05: float = ap_results.get(0.5, 0.0)

    return map_05_95, map_05, ap_results


def calculate_precision_recall(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    confidence_threshold: float = 0.5,
    num_classes: int = 1,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate precision and recall for each class.

    Args:
        predictions: List of predictions for each image [N, 6] [x1, y1, x2, y2, conf, class]
        targets: List of ground truth boxes for each image [M, 5] [x1, y1, x2, y2, class]
        confidence_threshold: Minimum confidence for predictions
        num_classes: Number of classes

    Returns:
        Dictionary with precision and recall per class
    """
    results: Dict[int, Dict[str, float]] = {}

    for class_id in range(num_classes):
        tp: int = 0
        fp: int = 0
        fn: int = 0

        for pred, target in zip(predictions, targets):
            # Filter by class and confidence
            class_mask: torch.Tensor = (pred[:, 5] == class_id) & (
                pred[:, 4] >= confidence_threshold
            )
            pred_filtered: torch.Tensor = pred[class_mask]

            # Get targets for this class
            target_mask: torch.Tensor = target[:, 4] == class_id
            target_filtered: torch.Tensor = target[target_mask]

            if len(pred_filtered) > 0 and len(target_filtered) > 0:
                ious: torch.Tensor = calculate_iou(
                    pred_filtered[:, :4], target_filtered[:, :4]
                )
                matches: torch.Tensor = (ious >= 0.5).any(dim=1)

                tp += matches.sum().item()
                fp += (~matches).sum().item()
                fn += max(0, len(target_filtered) - matches.sum().item())
            elif len(pred_filtered) > 0:
                fp += len(pred_filtered)
                fn += len(target_filtered)
            else:
                fn += len(target_filtered)

        precision: float = tp / (tp + fp + 1e-6)
        recall: float = tp / (tp + fn + 1e-6)

        results[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * (precision * recall) / (precision + recall + 1e-6),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return results
