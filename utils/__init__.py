"""Shared utilities package."""

from .model_utils import (
    adapt_input_channels,
    freeze_backbone,
    unfreeze_model,
    count_parameters,
    get_optimizer,
    get_scheduler,
)

from .metrics import (
    calculate_iou,
    calculate_ap,
    calculate_map,
    calculate_precision_recall,
)

from .logger import Logger, setup_logging

__all__ = [
    # Model utilities
    "adapt_input_channels",
    "freeze_backbone",
    "unfreeze_model",
    "count_parameters",
    "get_optimizer",
    "get_scheduler",
    # Metrics
    "calculate_iou",
    "calculate_ap",
    "calculate_map",
    "calculate_precision_recall",
    # Logger
    "Logger",
    "setup_logging",
]
