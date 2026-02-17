"""Stage 2 orchestration: Fine-tuning on custom multi-modal data."""

from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging
import sys

from config import Config
from train.train_yolo import train_yolo_stage2
from train.train_rcnn import train_fasterrcnn_stage2
from utils.logger import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


class Stage2Trainer:
    """Orchestrator for Stage 2 fine-tuning."""

    def __init__(
        self,
        custom_data_dir: Path,
        config: Optional[Config] = None,
        yolo_models: Optional[List[str]] = None,
        rcnn_backbone: str = "resnet50",
        stage1_checkpoints: Optional[Dict[str, Optional[Path]]] = None,
    ) -> None:
        """
        Initialize Stage 2 trainer.

        Args:
            custom_data_dir: Path to custom multi-modal dataset
            config: Configuration object
            yolo_models: List of YOLO models to fine-tune
            rcnn_backbone: Backbone for Faster R-CNN
            stage1_checkpoints: Dictionary of Stage 1 checkpoints
        """
        self.custom_data_dir: Path = custom_data_dir
        self.config: Config = config or Config()
        self.yolo_models: List[str] = yolo_models or self.config.models.yolo_models
        self.rcnn_backbone: str = rcnn_backbone
        self.stage1_checkpoints: Dict[str, Optional[Path]] = stage1_checkpoints or {}

        # Results storage: {modality: {model_name: checkpoint_path}}
        self.results: Dict[str, Dict[str, Optional[Path]]] = {}

        # Setup logging
        setup_logging(level=logging.INFO)

    def verify_custom_data(self) -> None:
        """Verify custom dataset structure."""
        logger.info("Verifying custom dataset structure...")

        if not self.custom_data_dir.exists():
            logger.error(f"Custom data directory not found: {self.custom_data_dir}")
            sys.exit(1)

        # Check for RGB, LWIR, UV directories
        modalities: List[str] = ["rgb", "lwir", "uv"]
        for modality in modalities:
            modality_dir: Path = self.custom_data_dir / modality
            if modality_dir.exists():
                logger.info(f"  Found {modality.upper()} data at {modality_dir}")
            else:
                logger.warning(
                    f"  {modality.upper()} data directory not found: {modality_dir}"
                )

    def load_stage1_checkpoints(self) -> None:
        """Load Stage 1 checkpoint paths."""
        logger.info("Loading Stage 1 checkpoints...")

        stage1_dir: Path = self.config.paths.stage1_checkpoints

        if not stage1_dir.exists():
            logger.warning("Stage 1 checkpoints directory not found")
            logger.warning("Models will be initialized from pretrained weights only")
            return

        # Find YOLO checkpoints
        for model_name in self.yolo_models:
            ckpt_dir: Path = stage1_dir / model_name / "weights"
            if ckpt_dir.exists():
                best_ckpt: Path = ckpt_dir / "best.pt"
                if best_ckpt.exists():
                    self.stage1_checkpoints[model_name] = best_ckpt
                    logger.info(f"  Found {model_name}: {best_ckpt}")

        # Find Faster R-CNN checkpoint
        rcnn_ckpt: Path = stage1_dir / "fasterrcnn" / "model_final.pth"
        if rcnn_ckpt.exists():
            self.stage1_checkpoints["fasterrcnn"] = rcnn_ckpt
            logger.info(f"  Found fasterrcnn: {rcnn_ckpt}")

        if not self.stage1_checkpoints:
            logger.warning("No Stage 1 checkpoints found")

    def fine_tune_yolo(self, modality: str) -> Dict[str, Optional[Path]]:
        """
        Fine-tune YOLO models on a specific modality.

        Args:
            modality: Modality ('rgb', 'lwir', 'uv')

        Returns:
            Dictionary mapping model names to checkpoint paths
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"STAGE 2: Fine-tuning YOLO models on {modality.upper()} modality")
        logger.info("=" * 60 + "\n")

        checkpoints: List[Path] = train_yolo_stage2(
            model_names=self.yolo_models,
            modality=modality,
            data_dir=self.custom_data_dir,
            stage1_checkpoints=self.stage1_checkpoints,
            config=self.config,
        )

        checkpoint_dict: Dict[str, Optional[Path]] = {
            model: ckpt for model, ckpt in zip(self.yolo_models, checkpoints)
        }
        return checkpoint_dict

    def fine_tune_fasterrcnn(self, modality: str) -> Path:
        """
        Fine-tune Faster R-CNN on a specific modality.

        Args:
            modality: Modality ('rgb', 'lwir', 'uv')

        Returns:
            Path to checkpoint
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"STAGE 2: Fine-tuning Faster R-CNN on {modality.upper()} modality")
        logger.info("=" * 60 + "\n")

        pretrained_path: Optional[Path] = self.stage1_checkpoints.get("fasterrcnn")

        checkpoint_path: Path = train_fasterrcnn_stage2(
            data_dir=self.custom_data_dir,
            modality=modality,
            backbone=self.rcnn_backbone,
            pretrained_path=pretrained_path,
            config=self.config,
        )

        return checkpoint_path

    def run(
        self, modalities: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Optional[Path]]]:
        """
        Run complete Stage 2 fine-tuning pipeline.

        Args:
            modalities: List of modalities to fine-tune on

        Returns:
            Nested dictionary {modality: {model_name: checkpoint_path}}
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING STAGE 2: DOMAIN ADAPTATION")
        logger.info("=" * 60 + "\n")

        modalities = modalities or ["rgb", "lwir", "uv"]

        try:
            # Step 1: Verify data
            self.verify_custom_data()

            # Step 2: Load Stage 1 checkpoints
            self.load_stage1_checkpoints()

            # Step 3: Fine-tune on each modality
            for modality in modalities:
                modality_dir: Path = self.custom_data_dir / modality

                if not modality_dir.exists():
                    logger.warning(f"Skipping {modality}: data directory not found")
                    continue

                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing {modality.upper()} modality")
                logger.info(f"{'=' * 60}\n")

                # Fine-tune YOLO models
                yolo_checkpoints: Dict[str, Optional[Path]] = self.fine_tune_yolo(
                    modality
                )
                self.results[modality] = yolo_checkpoints

                # Fine-tune Faster R-CNN
                rcnn_checkpoint: Path = self.fine_tune_fasterrcnn(modality)
                self.results[modality]["fasterrcnn"] = rcnn_checkpoint

            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 2 FINE-TUNING COMPLETE")
            logger.info("=" * 60 + "\n")

            logger.info("Fine-tuned models:")
            for modality, model_checkpoints in self.results.items():
                logger.info(f"\n{modality.upper()}:")
                for model_name, ckpt_path in model_checkpoints.items():
                    logger.info(f"  {model_name}: {ckpt_path}")

            return self.results

        except Exception as e:
            logger.error(f"Error during Stage 2 training: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def run_single_model(
        self, model_name: str, modalities: Optional[List[str]] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Fine-tune a single model on multiple modalities.

        Args:
            model_name: Name of model to fine-tune
            modalities: List of modalities to fine-tune on

        Returns:
            Dictionary mapping modality to checkpoint path
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fine-tuning {model_name} on multiple modalities")
        logger.info(f"{'=' * 60}\n")

        modalities = modalities or ["rgb", "lwir", "uv"]
        single_model_results: Dict[str, Optional[Path]] = {}

        for modality in modalities:
            modality_dir: Path = self.custom_data_dir / modality

            if not modality_dir.exists():
                logger.warning(f"Skipping {modality}: data directory not found")
                continue

            # Check if YOLO or Faster R-CNN
            if "yolo" in model_name.lower() or model_name.startswith("yolo"):
                # YOLO model
                model_checkpoints: List[Path] = train_yolo_stage2(
                    model_names=[model_name],
                    modality=modality,
                    data_dir=self.custom_data_dir,
                    stage1_checkpoints={
                        model_name: self.stage1_checkpoints.get(model_name)
                    }
                    if self.stage1_checkpoints
                    else None,
                    config=self.config,
                )
                single_model_results[modality] = (
                    model_checkpoints[0] if model_checkpoints else None
                )
            else:
                # Faster R-CNN
                single_checkpoint_path: Path = train_fasterrcnn_stage2(
                    data_dir=self.custom_data_dir,
                    modality=modality,
                    backbone=model_name.replace("fasterrcnn_", ""),
                    pretrained_path=self.stage1_checkpoints.get("fasterrcnn"),
                    config=self.config,
                )
                single_model_results[modality] = single_checkpoint_path

        return single_model_results


def main() -> None:
    """Main function for Stage 2 orchestration."""
    # Setup logging
    setup_logging()

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Domain Adaptation")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to custom multi-modal dataset"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["rgb", "lwir", "uv"],
        choices=["rgb", "lwir", "uv"],
        help="Modalities to fine-tune on",
    )
    parser.add_argument(
        "--yolo-models", nargs="+", default=None, help="YOLO models to fine-tune"
    )
    parser.add_argument(
        "--rcnn-backbone", type=str, default="resnet50", help="Faster R-CNN backbone"
    )
    parser.add_argument(
        "--single-model", type=str, help="Fine-tune a single model only"
    )

    args = parser.parse_args()

    # Create trainer
    trainer: Stage2Trainer = Stage2Trainer(
        custom_data_dir=Path(args.data),
        yolo_models=args.yolo_models,
        rcnn_backbone=args.rcnn_backbone,
    )

    # Run Stage 2
    if args.single_model:
        single_model_results: Dict[str, Optional[Path]] = trainer.run_single_model(
            args.single_model, args.modalities
        )
        logger.info(f"\nFine-tuned {args.single_model}:")
        for modality, ckpt_path in single_model_results.items():
            logger.info(f"  {modality}: {ckpt_path}")
    else:
        all_models_results: Dict[str, Dict[str, Optional[Path]]] = trainer.run(
            modalities=args.modalities
        )

    logger.info("\nAll Stage 2 models saved!")


if __name__ == "__main__":
    main()
