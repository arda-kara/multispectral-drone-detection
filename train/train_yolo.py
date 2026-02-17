"""YOLO training script for Stage 1 and Stage 2."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

import numpy as np
import torch
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Ultralytics package not available. Install with: pip install ultralytics")

ULTRALYTICS_AVAILABLE: bool = YOLO is not None

from config import Config
from utils.model_utils import adapt_input_channels, freeze_backbone, unfreeze_model
from utils.logger import Logger, setup_logging
from utils.metrics import calculate_map

logger: logging.Logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Trainer for YOLO models (v8, v9, v11)."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Config] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize YOLO trainer.

        Args:
            model_name: Model name ('yolov8n', 'yolov9c', 'yolov11n', etc.)
            config: Configuration object
            experiment_name: Name for logging
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics package is required. Install with: pip install ultralytics"
            )

        self.model_name: str = model_name
        self.config: Config = config or Config()
        self.experiment_name: str = experiment_name or f"{model_name}_training"

        # Initialize logger
        self.logger: Logger = Logger(
            experiment_name=self.experiment_name,
            log_dir=self.config.paths.runs,
            use_tensorboard=self.config.logging.use_tensorboard,
            use_wandb=self.config.logging.use_wandb,
            wandb_project=self.config.logging.wandb_project,
        )

        # Model will be loaded later
        self.model: Optional[YOLO] = None
        self.device: str = (
            self.config.training.device if torch.cuda.is_available() else "cpu"
        )

    def train_stage1(
        self,
        data_yaml: Path,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Path:
        """
        Train YOLO model on Stage 1 (pre-training) data.

        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Path to saved checkpoint
        """
        epochs = epochs or self.config.training.epochs_stage1
        batch_size = batch_size or self.config.training.batch_size_yolo

        logger.info(f"Starting Stage 1 training for {self.model_name}")

        # Load pretrained model
        self.model = YOLO(f"{self.model_name}.pt")
        assert self.model is not None

        # Configure augmentations for tiny objects
        augment_config: Dict[str, Any] = {
            "mosaic": self.config.augmentation.use_mosaic,
            "mixup": self.config.augmentation.use_mixup,
            "copy_paste": False,
            "auto_augment": "randaugment",
            "hsv_h": self.config.augmentation.use_hsv_h,
            "hsv_s": self.config.augmentation.use_hsv_s,
            "hsv_v": self.config.augmentation.use_hsv_v,
            "degrees": 0.0,
            "translate": self.config.augmentation.use_translate,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": self.config.augmentation.use_perspective,
            "flipud": self.config.augmentation.use_flipud,
            "fliplr": self.config.augmentation.use_fliplr,
            "mosaic_prob": 1.0,
            "mixup_prob": 0.15,
        }

        # Train model
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=self.config.training.image_size[0],
            device=self.device,
            workers=self.config.training.workers,
            project=str(self.config.paths.checkpoints),
            name=f"stage1/{self.model_name}",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=self.config.training.learning_rate_stage1,
            lrf=0.01,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
            warmup_epochs=self.config.training.warmup_epochs,
            patience=20,
            save_period=self.config.logging.save_interval,
            val=True,
            plots=True,
            **augment_config,
        )

        # Save final checkpoint
        checkpoint_path: Path = Path(results.save_dir) / "weights" / "best.pt"
        logger.info(f"Stage 1 training complete. Best model saved to {checkpoint_path}")

        self.logger.close()
        return checkpoint_path

    def train_stage2(
        self,
        data_dir: Path,
        modality: str = "rgb",
        pretrained_path: Optional[Path] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Path:
        """
        Fine-tune YOLO model on Stage 2 (multi-modal) data.

        Args:
            data_dir: Path to custom multi-modal dataset
            modality: Modality ('rgb', 'lwir', 'uv')
            pretrained_path: Path to Stage 1 pretrained weights
            epochs: Number of fine-tuning epochs
            batch_size: Batch size

        Returns:
            Path to saved checkpoint
        """
        epochs = epochs or self.config.training.epochs_stage2
        batch_size = batch_size or self.config.training.batch_size_yolo

        logger.info(
            f"Starting Stage 2 fine-tuning for {self.model_name} ({modality} modality)"
        )

        # Determine input channels
        if modality == "rgb":
            input_channels = self.config.models.input_channels_rgb
        elif modality in ["lwir", "uv"]:
            input_channels = 1
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Adapt input channels if needed
        if input_channels != 3:
            logger.info(f"Adapting input channels from 3 to {input_channels}")
            assert self.model is not None
            # Get the model's nn.Module
            model_module: torch.nn.Module = self.model.model

            # Adapt first conv layer
            adapt_input_channels(
                model_module,
                original_channels=3,
                target_channels=input_channels,
                init_method="average",
            )

            # Update model
            self.model.model = model_module

        # Create data.yaml for custom dataset
        data_yaml: Path = self._create_custom_data_yaml(data_dir, modality)

        # Configure fine-tuning hyperparameters
        freeze_backbone: bool = True
        freeze_epochs: int = min(self.config.models.freeze_backbone_epochs, epochs // 2)

        # Train model with transfer learning
        for epoch in range(epochs):
            current_epoch: int = epoch + 1

            # Determine if we should unfreeze backbone
            if current_epoch == freeze_epochs + 1 and freeze_backbone:
                logger.info(f"Unfreezing backbone at epoch {current_epoch}")
                assert self.model is not None
                unfreeze_model(self.model.model)
                freeze_backbone = False

            # Adjust learning rate
            if current_epoch <= self.config.training.warmup_epochs:
                current_lr: float = self.config.training.learning_rate_stage2 * (
                    current_epoch / self.config.training.warmup_epochs
                )
            elif freeze_backbone:
                current_lr = self.config.training.learning_rate_stage2 * 0.1
            else:
                # Cosine annealing
                progress: float = (current_epoch - freeze_epochs) / (
                    epochs - freeze_epochs
                )
                current_lr = (
                    self.config.training.learning_rate_stage2
                    * 0.5
                    * (1 + np.cos(np.pi * progress))
                )

            # Train for one epoch
            assert self.model is not None
            results = self.model.train(
                data=str(data_yaml),
                epochs=1,
                batch=batch_size,
                imgsz=self.config.training.image_size[0],
                device=self.device,
                workers=self.config.training.workers,
                project=str(self.config.paths.stage2_checkpoints),
                name=f"{self.model_name}_{modality}",
                exist_ok=False,
                optimizer="AdamW",
                lr0=current_lr,
                lrf=0.01,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
                warmup_epochs=0,
                patience=10,
                save=(current_epoch % self.config.logging.save_interval == 0),
                val=(current_epoch % self.config.logging.eval_interval == 0),
                plots=(current_epoch % 10 == 0),
                mosaic=False,  # Disable mosaic for fine-tuning
                mixup=False,  # Disable mixup for fine-tuning
            )

            # Log metrics
            if hasattr(results, "metrics"):
                self.logger.log_scalar(
                    "loss/train",
                    results.results_dict.get("train/box_loss", 0),
                    current_epoch,
                    "train",
                )
                self.logger.log_scalar(
                    "mAP",
                    results.metrics.get("metrics/mAP50(B)", 0),
                    current_epoch,
                    "val",
                )

            logger.info(
                f"Epoch {current_epoch}/{epochs} complete. mAP: {results.metrics.get('metrics/mAP50(B)', 0):.4f}"
            )

        # Save final checkpoint
        checkpoint_dir: Path = (
            self.config.paths.stage2_checkpoints / f"{self.model_name}_{modality}"
        )
        checkpoint_path: Path = checkpoint_dir / "weights" / "best.pt"

        # Ensure checkpoint directory exists
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "weights").mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Stage 2 fine-tuning complete. Best model saved to {checkpoint_path}"
        )

        self.logger.close()
        return checkpoint_path

    def _create_custom_data_yaml(self, data_dir: Path, modality: str) -> Path:
        """
        Create data.yaml file for custom dataset.

        Args:
            data_dir: Path to custom dataset
            modality: Modality

        Returns:
            Path to data.yaml
        """
        data_yaml: Dict[str, Any] = {
            "path": str(data_dir / modality),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,  # Assuming single class (drone)
            "names": {0: "drone"},
        }

        yaml_path: Path = data_dir / f"data_{modality}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return yaml_path

    def evaluate(
        self,
        checkpoint_path: Path,
        data_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Dict[str, float]:
        """
        Evaluate trained model.

        Args:
            checkpoint_path: Path to model checkpoint
            data_path: Path to evaluation data
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model {checkpoint_path}")

        # Load model
        self.model = YOLO(str(checkpoint_path))
        assert self.model is not None

        # Run validation
        val_results = self.model.val(
            data=str(data_path),
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            plots=True,
        )

        metrics: Dict[str, float] = {
            "mAP50": val_results.metrics.get("metrics/mAP50(B)", 0),
            "mAP50_95": val_results.metrics.get("metrics/mAP50-95(B)", 0),
            "precision": val_results.metrics.get("metrics/precision(B)", 0),
            "recall": val_results.metrics.get("metrics/recall(B)", 0),
        }

        logger.info(f"Evaluation results: {metrics}")
        return metrics


def train_yolo_stage1(
    model_names: List[str], data_yaml: Path, config: Optional[Config] = None
) -> List[Path]:
    """
    Train multiple YOLO models in Stage 1.

    Args:
        model_names: List of model names to train
        data_yaml: Path to data.yaml
        config: Configuration object

    Returns:
        List of checkpoint paths
    """
    config = config or Config()
    setup_logging()

    checkpoint_paths: List[Path] = []

    for model_name in model_names:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'=' * 50}\n")

        trainer: YOLOTrainer = YOLOTrainer(
            model_name=model_name, config=config, experiment_name=f"{model_name}_stage1"
        )

        try:
            checkpoint_path: Path = trainer.train_stage1(data_yaml)
            checkpoint_paths.append(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback

            traceback.print_exc()

    return checkpoint_paths


def train_yolo_stage2(
    model_names: List[str],
    modality: str,
    data_dir: Path,
    stage1_checkpoints: Optional[Dict[str, Optional[Path]]] = None,
    config: Optional[Config] = None,
) -> List[Path]:
    """
    Fine-tune multiple YOLO models in Stage 2.

    Args:
        model_names: List of model names to fine-tune
        modality: Modality ('rgb', 'lwir', 'uv')
        data_dir: Path to custom dataset
        stage1_checkpoints: Dictionary mapping model names to Stage 1 checkpoints
        config: Configuration object

    Returns:
        List of checkpoint paths
    """
    config = config or Config()
    setup_logging()

    checkpoint_paths: List[Path] = []

    for model_name in model_names:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Fine-tuning {model_name} ({modality})")
        logger.info(f"{'=' * 50}\n")

        # Get pretrained checkpoint
        pretrained_path: Optional[Path] = None
        if stage1_checkpoints:
            pretrained_path = stage1_checkpoints.get(model_name)

        trainer: YOLOTrainer = YOLOTrainer(
            model_name=model_name,
            config=config,
            experiment_name=f"{model_name}_stage2_{modality}",
        )

        try:
            checkpoint_path: Path = trainer.train_stage2(
                data_dir=data_dir, modality=modality, pretrained_path=pretrained_path
            )
            checkpoint_paths.append(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to fine-tune {model_name}: {e}")
            import traceback

            traceback.print_exc()

    return checkpoint_paths


def main() -> None:
    """Main function for YOLO training."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO models")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], required=True, help="Training stage"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n", "yolov9c", "yolov11n"],
        help="YOLO models",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data directory or data.yaml"
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
            data_yaml: Path = Path(args.data)
            if not data_yaml.exists():
                logger.error(f"Data YAML not found: {data_yaml}")
                sys.exit(1)

            stage1_checkpoints: List[Path] = train_yolo_stage1(
                model_names=args.models, data_yaml=data_yaml
            )

            logger.info(f"\nStage 1 training complete!")
            logger.info(f"Checkpoints: {stage1_checkpoints}")

        elif args.stage == 2:
            # Stage 2: Fine-tuning
            data_dir: Path = Path(args.data)
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                sys.exit(1)

            pretrained_path: Optional[Path] = (
                Path(args.pretrained) if args.pretrained else None
            )

            stage2_checkpoints: List[Path] = train_yolo_stage2(
                model_names=args.models,
                modality=args.modality,
                data_dir=data_dir,
                stage1_checkpoints={args.models[0]: pretrained_path}
                if pretrained_path
                else None,
            )

            logger.info(f"\nStage 2 fine-tuning complete!")
            logger.info(f"Checkpoints: {stage2_checkpoints}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
