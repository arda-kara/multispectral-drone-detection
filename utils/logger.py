"""Logging utilities for TensorBoard and WandB."""

from typing import Optional, Dict, Any
import logging
from pathlib import Path
from datetime import datetime

import torch

logger: logging.Logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    logger.warning("TensorBoard not available")

TENSORBOARD_AVAILABLE: bool = SummaryWriter is not None

try:
    import wandb
except ImportError:
    wandb = None
    logger.warning("Weights & Biases not available")

WANDB_AVAILABLE: bool = wandb is not None


class Logger:
    """Unified logging interface for TensorBoard and WandB."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: Path = Path("runs"),
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "multispectral-drone-detection",
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for TensorBoard logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: WandB project name
            wandb_config: Configuration to log to WandB
        """
        self.experiment_name: str = experiment_name
        self.use_tensorboard: bool = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb: bool = use_wandb and WANDB_AVAILABLE

        # Initialize TensorBoard
        if self.use_tensorboard:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tb_log_dir: Path = log_dir / f"{experiment_name}_{timestamp}"
            self.tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer: Optional["SummaryWriter"] = SummaryWriter(
                str(self.tb_log_dir)
            )
            logger.info(f"TensorBoard logging to {self.tb_log_dir}")
        else:
            self.tb_writer = None

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=wandb_project, name=experiment_name, config=wandb_config or {}
            )
            logger.info(f"WandB logging to project {wandb_project}")

    def log_scalar(
        self, tag: str, value: float, step: int, phase: str = "train"
    ) -> None:
        """
        Log a scalar value.

        Args:
            tag: Metric name
            value: Metric value
            step: Training step
            phase: Phase (train/val)
        """
        full_tag: str = f"{phase}/{tag}"

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar(full_tag, value, step)

        if self.use_wandb:
            wandb.log({full_tag: value}, step=step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int,
        phase: str = "train",
    ) -> None:
        """
        Log multiple scalar values.

        Args:
            main_tag: Main metric category
            tag_scalar_dict: Dictionary of metric names and values
            step: Training step
            phase: Phase (train/val)
        """
        full_tag: str = f"{phase}/{main_tag}"

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalars(full_tag, tag_scalar_dict, step)

        if self.use_wandb:
            log_dict: Dict[str, float] = {
                f"{full_tag}/{k}": v for k, v in tag_scalar_dict.items()
            }
            wandb.log(log_dict, step=step)

    def log_image(self, tag: str, image: Any, step: int, phase: str = "train") -> None:
        """
        Log an image.

        Args:
            tag: Image tag
            image: Image tensor or array
            step: Training step
            phase: Phase (train/val)
        """
        full_tag: str = f"{phase}/{tag}"

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_image(full_tag, image, step)

        if self.use_wandb:
            wandb.log({full_tag: wandb.Image(image)}, step=step)

    def log_images(
        self,
        tag: str,
        images: Any,
        step: int,
        phase: str = "train",
        max_images: int = 16,
    ) -> None:
        """
        Log multiple images in a grid.

        Args:
            tag: Image tag
            images: Image tensors or arrays
            step: Training step
            phase: Phase (train/val)
            max_images: Maximum number of images to log
        """
        if len(images) > max_images:
            images = images[:max_images]

        full_tag: str = f"{phase}/{tag}"

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_images(full_tag, images, step)

        if self.use_wandb:
            wandb.log({full_tag: [wandb.Image(img) for img in images]}, step=step)

    def log_histogram(
        self, tag: str, values: Any, step: int, phase: str = "train"
    ) -> None:
        """
        Log a histogram.

        Args:
            tag: Histogram tag
            values: Values to histogram
            step: Training step
            phase: Phase (train/val)
        """
        full_tag: str = f"{phase}/{tag}"

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(full_tag, values, step)

        if self.use_wandb:
            wandb.log({full_tag: wandb.Histogram(values)}, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_hparams(params, {})

        if self.use_wandb:
            wandb.config.update(params)

    def log_text(self, tag: str, text: str, step: int, phase: str = "train") -> None:
        """
        Log text.

        Args:
            tag: Text tag
            text: Text to log
            step: Training step
            phase: Phase (train/val)
        """
        full_tag: str = f"{phase}/{tag}"

        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_text(full_tag, text, step)

        if self.use_wandb:
            wandb.log({full_tag: text}, step=step)

    def save_model(
        self,
        model_state_dict: Any,
        path: Path,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            model_state_dict: Model state dict
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional metrics to log with checkpoint
        """
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "state_dict": model_state_dict,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

        logger.info(f"Saved checkpoint to {path}")

    def close(self) -> None:
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()

        logger.info("Closed all loggers")


def setup_logging(
    log_file: Optional[Path] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Setup basic logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    return logging.getLogger(__name__)
