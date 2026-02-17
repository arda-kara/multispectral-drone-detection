"""Faster R-CNN training script using Detectron2 for Stage 1 and Stage 2."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import json

import torch

try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import (
        DefaultTrainer,
        default_writers,
        default_setup,
        HookBase,
    )
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.utils.logger import setup as detectron2_setup
    from detectron2.utils.visualizer import Visualizer
    from detectron2.modeling import build_model
except ImportError:
    detectron2 = None
    model_zoo = None
    DefaultTrainer = None
    get_cfg = None
    DatasetCatalog = None
    MetadataCatalog = None
    register_coco_instances = None
    detectron2_setup = None
    Visualizer = None
    build_model = None
    print("Detectron2 package not available. Install from source")

import os
import numpy as np
from PIL import Image

DETECTRON2_AVAILABLE: bool = detectron2 is not None

from config import Config
from utils.model_utils import adapt_input_channels, freeze_backbone, unfreeze_model
from utils.logger import Logger, setup_logging

logger: logging.Logger = logging.getLogger(__name__)


class FasterRCNNTrainer:
    """Trainer for Faster R-CNN using Detectron2."""

    def __init__(
        self,
        backbone: str = "resnet50",
        config: Optional[Config] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize Faster R-CNN trainer.

        Args:
            backbone: Backbone architecture
            config: Configuration object
            experiment_name: Name for logging
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 is required. Install from source")

        self.backbone: str = backbone
        self.config: Config = config or Config()
        self.experiment_name: str = experiment_name or f"fasterrcnn_{backbone}_training"

        # Initialize logger
        self.logger: Logger = Logger(
            experiment_name=self.experiment_name,
            log_dir=self.config.paths.runs,
            use_tensorboard=self.config.logging.use_tensorboard,
            use_wandb=self.config.logging.use_wandb,
            wandb_project=self.config.logging.wandb_project,
        )

        # Setup Detectron2 config
        self.cfg: Any = get_cfg()
        self.model: Optional[Any] = None

        # Device
        self.device: str = (
            self.config.training.device if torch.cuda.is_available() else "cpu"
        )

    def setup_config(
        self,
        train_dataset_name: str,
        val_dataset_name: str,
        num_classes: int,
        input_channels: int = 3,
    ) -> None:
        """
        Setup Detectron2 configuration.

        Args:
            train_dataset_name: Name of training dataset
            val_dataset_name: Name of validation dataset
            num_classes: Number of classes
            input_channels: Number of input channels
        """
        # Load default config
        if self.backbone == "resnet50":
            config_file: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        elif self.backbone == "resnet101":
            config_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        # Dataset configuration
        self.cfg.DATASETS.TRAIN = (train_dataset_name,)
        self.cfg.DATASETS.TEST = (val_dataset_name,)

        # DataLoader configuration
        self.cfg.DATALOADER.NUM_WORKERS = self.config.training.workers

        # Model configuration
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # Training configuration
        self.cfg.SOLVER.IMS_PER_BATCH = self.config.training.batch_size_rcnn
        self.cfg.SOLVER.BASE_LR = self.config.training.learning_rate_stage1
        self.cfg.SOLVER.WARMUP_ITERS = self.config.training.warmup_epochs * 10
        self.cfg.SOLVER.MAX_ITER = self.config.training.epochs_stage1 * 10
        self.cfg.SOLVER.STEPS = []
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.MOMENTUM = self.config.training.momentum
        self.cfg.SOLVER.WEIGHT_DECAY = self.config.training.weight_decay

        # Output directory
        self.cfg.OUTPUT_DIR = str(self.config.paths.stage1_checkpoints / "fasterrcnn")

        # Input configuration
        self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)
        self.cfg.INPUT.MAX_SIZE_TRAIN = 640
        self.cfg.INPUT.MIN_SIZE_TEST = 640
        self.cfg.INPUT.MAX_SIZE_TEST = 640

        # Checkpoint period
        self.cfg.SOLVER.CHECKPOINT_PERIOD = self.config.logging.save_interval * 10

        logger.info(f"Configured Faster R-CNN with {self.backbone} backbone")

    def adapt_input_channels(self, input_channels: int = 1) -> None:
        """
        Adapt the model to accept different number of input channels.

        Args:
            input_channels: Number of input channels
        """
        logger.info(f"Adapting input channels to {input_channels}")

        # Build model to access backbone
        self.model = build_model(self.cfg)
        self.model.eval()

        # Get the backbone's first convolutional layer
        backbone: Any = self.model.backbone
        if hasattr(backbone, "stem"):
            # ResNet backbone
            first_conv: Any = backbone.stem[0]
        else:
            raise ValueError("Unable to find backbone's first conv layer")

        # Adapt the layer
        if input_channels != first_conv.in_channels:
            original_weight: torch.Tensor = first_conv.weight.data
            out_channels: int = first_conv.out_channels
            kernel_size: tuple = first_conv.kernel_size
            stride: tuple = first_conv.stride
            padding: tuple = first_conv.padding

            # Create new conv layer
            from torch.nn import Conv2d

            new_conv: Conv2d = Conv2d(
                input_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=(first_conv.bias is not None),
            )

            # Initialize weights (average RGB channels for grayscale)
            with torch.no_grad():
                if input_channels == 1 and first_conv.in_channels == 3:
                    # Average RGB weights
                    new_weight: torch.Tensor = original_weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(new_weight)
                else:
                    # Random initialization for new channels
                    new_conv.weight = torch.nn.Parameter(
                        torch.randn_like(new_conv.weight) * 0.01
                    )

                # Copy bias if exists
                if first_conv.bias is not None:
                    new_conv.bias = torch.nn.Parameter(first_conv.bias.clone())

            # Replace the layer
            if hasattr(backbone, "stem"):
                backbone.stem[0] = new_conv

            # Update config
            self.cfg.MODEL.PIXEL_MEAN = [128.0] * input_channels
            self.cfg.MODEL.PIXEL_STD = [128.0] * input_channels

            logger.info(
                f"Successfully adapted first conv layer: 3 -> {input_channels} channels"
            )

    def train_stage1(
        self, train_dataset_name: str, val_dataset_name: str, num_classes: int
    ) -> Path:
        """
        Train Faster R-CNN on Stage 1 (pre-training) data.

        Args:
            train_dataset_name: Name of training dataset
            val_dataset_name: Name of validation dataset
            num_classes: Number of classes

        Returns:
            Path to saved checkpoint
        """
        logger.info(f"Starting Stage 1 training for Faster R-CNN ({self.backbone})")

        # Setup configuration
        self.setup_config(
            train_dataset_name, val_dataset_name, num_classes, input_channels=3
        )

        # Create trainer
        trainer: Detectron2Trainer = Detectron2Trainer(self.cfg)
        trainer.resume_or_load(resume=False)

        # Train
        trainer.train()

        # Save final checkpoint
        checkpoint_path: Path = Path(self.cfg.OUTPUT_DIR) / "model_final.pth"
        logger.info(f"Stage 1 training complete. Best model saved to {checkpoint_path}")

        self.logger.close()
        return checkpoint_path

    def train_stage2(
        self,
        train_dataset_name: str,
        val_dataset_name: str,
        num_classes: int,
        modality: str = "rgb",
        pretrained_path: Optional[Path] = None,
    ) -> Path:
        """
        Fine-tune Faster R-CNN on Stage 2 (multi-modal) data.

        Args:
            train_dataset_name: Name of training dataset
            val_dataset_name: Name of validation dataset
            num_classes: Number of classes
            modality: Modality ('rgb', 'lwir', 'uv')
            pretrained_path: Path to Stage 1 pretrained weights

        Returns:
            Path to saved checkpoint
        """
        logger.info(
            f"Starting Stage 2 fine-tuning for Faster R-CNN ({modality} modality)"
        )

        # Determine input channels
        if modality == "rgb":
            input_channels: int = self.config.models.input_channels_rgb
        elif modality in ["lwir", "uv"]:
            input_channels = 1
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Setup configuration
        self.setup_config(
            train_dataset_name, val_dataset_name, num_classes, input_channels
        )

        # Update for Stage 2
        self.cfg.SOLVER.BASE_LR = self.config.training.learning_rate_stage2
        self.cfg.SOLVER.MAX_ITER = self.config.training.epochs_stage2 * 10
        self.cfg.OUTPUT_DIR = str(
            self.config.paths.stage2_checkpoints / f"fasterrcnn_{modality}"
        )

        # Load pretrained weights
        if pretrained_path and pretrained_path.exists():
            self.cfg.MODEL.WEIGHTS = str(pretrained_path)
            logger.info(f"Loaded pretrained weights from {pretrained_path}")

        # Adapt input channels if needed
        if input_channels != 3:
            self.adapt_input_channels(input_channels)

        # Create trainer
        trainer: Detectron2Trainer = Detectron2Trainer(self.cfg)
        trainer.resume_or_load(resume=True)

        # Train with custom hooks for transfer learning
        freeze_backbone: bool = True
        freeze_iters: int = min(
            self.config.models.freeze_backbone_epochs * 10,
            self.cfg.SOLVER.MAX_ITER // 2,
        )

        # Register custom hooks
        class BackboneFreezeHook(HookBase):
            """Custom hook to unfreeze backbone after specified iterations."""

            def __init__(self, freeze_iters, unfreeze_callback):
                super().__init__()
                self.freeze_iters = freeze_iters
                self.unfreeze_callback = unfreeze_callback
                self.unfrozen = False

            def after_step(self):
                if self.trainer.iter == self.freeze_iters and not self.unfrozen:
                    logger.info(f"Unfreezing backbone at iteration {self.trainer.iter}")
                    self.unfreeze_callback(self.trainer.model.backbone)
                    self.unfrozen = True

        # Add hook to trainer
        trainer.register_hooks([BackboneFreezeHook(freeze_iters, unfreeze_model)])

        # Train the model
        trainer.train()

        logger.info(f"Stage 2 fine-tuning complete")

        self.logger.close()
        checkpoint_path: Path = Path(self.cfg.OUTPUT_DIR) / "model_final.pth"
        return checkpoint_path


class Detectron2Trainer(DefaultTrainer):
    """Custom Detectron2 trainer with additional logging."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build evaluator."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        return COCOEvaluator(dataset_name, output_dir=output_folder)


def register_coco_dataset(
    name: str,
    images_dir: Path,
    annotations_file: Path,
    thing_classes: Optional[List[str]] = None,
) -> None:
    """
    Register a COCO format dataset with Detectron2.

    Args:
        name: Dataset name
        images_dir: Directory containing images
        annotations_file: Path to COCO annotations JSON
        thing_classes: List of class names
    """
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)

    register_coco_instances(name, {}, str(annotations_file), str(images_dir))

    if thing_classes:
        MetadataCatalog.get(name).thing_classes = thing_classes

    logger.info(f"Registered dataset: {name}")


def train_fasterrcnn_stage1(
    data_dir: Path, backbone: str = "resnet50", config: Optional[Config] = None
) -> Path:
    """
    Train Faster R-CNN in Stage 1.

    Args:
        data_dir: Path to COCO format dataset
        backbone: Backbone architecture
        config: Configuration object

    Returns:
        Path to saved checkpoint
    """
    config = config or Config()
    detectron2_setup(name="multispectral_drone_detection")
    setup_logging()

    # Register datasets
    register_coco_dataset(
        "drone_train",
        data_dir / "images" / "train",
        data_dir / "annotations" / "train.json",
        thing_classes=["bird", "drone"],
    )

    register_coco_dataset(
        "drone_val",
        data_dir / "images" / "valid",
        data_dir / "annotations" / "valid.json",
        thing_classes=["bird", "drone"],
    )

    # Train
    trainer: FasterRCNNTrainer = FasterRCNNTrainer(
        backbone=backbone,
        config=config,
        experiment_name=f"fasterrcnn_{backbone}_stage1",
    )

    checkpoint_path: Path = trainer.train_stage1(
        train_dataset_name="drone_train", val_dataset_name="drone_val", num_classes=2
    )

    return checkpoint_path


def train_fasterrcnn_stage2(
    data_dir: Path,
    modality: str = "rgb",
    backbone: str = "resnet50",
    pretrained_path: Optional[Path] = None,
    config: Optional[Config] = None,
) -> Path:
    """
    Fine-tune Faster R-CNN in Stage 2.

    Args:
        data_dir: Path to custom dataset
        modality: Modality ('rgb', 'lwir', 'uv')
        backbone: Backbone architecture
        pretrained_path: Path to Stage 1 checkpoint
        config: Configuration object

    Returns:
        Path to saved checkpoint
    """
    config = config or Config()
    detectron2_setup(name="multispectral_drone_detection")
    setup_logging()

    # Register datasets
    modality_dir: Path = data_dir / modality

    register_coco_dataset(
        f"drone_{modality}_train",
        modality_dir / "images" / "train",
        modality_dir / "annotations" / "train.json",
        thing_classes=["drone"],
    )

    register_coco_dataset(
        f"drone_{modality}_val",
        modality_dir / "images" / "val",
        modality_dir / "annotations" / "val.json",
        thing_classes=["drone"],
    )

    # Train
    trainer: FasterRCNNTrainer = FasterRCNNTrainer(
        backbone=backbone,
        config=config,
        experiment_name=f"fasterrcnn_{backbone}_stage2_{modality}",
    )

    checkpoint_path: Path = trainer.train_stage2(
        train_dataset_name=f"drone_{modality}_train",
        val_dataset_name=f"drone_{modality}_val",
        num_classes=1,
        modality=modality,
        pretrained_path=pretrained_path,
    )

    return checkpoint_path


def main() -> None:
    """Main function for Faster R-CNN training."""
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train Faster R-CNN models")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], required=True, help="Training stage"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="rgb",
        choices=["rgb", "lwir", "uv"],
        help="Modality for Stage 2",
    )
    parser.add_argument(
        "--pretrained", type=str, help="Path to pretrained weights (Stage 2 only)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        if args.stage == 1:
            # Stage 1: Pre-training
            stage1_data_dir: Path = Path(args.data)
            if not stage1_data_dir.exists():
                logger.error(f"Data directory not found: {stage1_data_dir}")
                sys.exit(1)

            checkpoint_path: Path = train_fasterrcnn_stage1(
                data_dir=stage1_data_dir, backbone=args.backbone
            )

            logger.info(f"\nStage 1 training complete!")
            logger.info(f"Checkpoint: {checkpoint_path}")

        elif args.stage == 2:
            # Stage 2: Fine-tuning
            stage2_data_dir: Path = Path(args.data)
            if not stage2_data_dir.exists():
                logger.error(f"Data directory not found: {stage2_data_dir}")
                sys.exit(1)

            pretrained_path: Optional[Path] = (
                Path(args.pretrained) if args.pretrained else None
            )

            stage2_checkpoint_path: Path = train_fasterrcnn_stage2(
                data_dir=stage2_data_dir,
                modality=args.modality,
                backbone=args.backbone,
                pretrained_path=pretrained_path,
            )

            logger.info(f"\nStage 2 fine-tuning complete!")
            logger.info(f"Checkpoint: {stage2_checkpoint_path}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
