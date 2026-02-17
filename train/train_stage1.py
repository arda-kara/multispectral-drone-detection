"""Stage 1 orchestration: Pre-training on generic drone-vs-bird data."""

from typing import Optional, List, Dict
from pathlib import Path
import logging
import sys

from config import Config
from train.roboflow_ingestion import RoboflowIngestion
from train.train_yolo import train_yolo_stage1
from train.train_rcnn import train_fasterrcnn_stage1
from utils.logger import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


class Stage1Trainer:
    """Orchestrator for Stage 1 pre-training."""

    def __init__(
        self,
        api_key: str,
        config: Optional[Config] = None,
        yolo_models: Optional[List[str]] = None,
        rcnn_backbone: str = "resnet50",
    ) -> None:
        """
        Initialize Stage 1 trainer.

        Args:
            api_key: Roboflow API key
            config: Configuration object
            yolo_models: List of YOLO models to train
            rcnn_backbone: Backbone for Faster R-CNN
        """
        self.api_key: str = api_key
        self.config: Config = config or Config()
        self.yolo_models: List[str] = yolo_models or self.config.models.yolo_models
        self.rcnn_backbone: str = rcnn_backbone

        # Results storage
        self.results: Dict[str, Path] = {}

        # Setup logging
        setup_logging(level=logging.INFO)

    def download_and_prepare_data(self) -> Dict[str, Path]:
        """
        Download and prepare Stage 1 data from Roboflow.

        Returns:
            Dictionary mapping format to data directory
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Downloading and preparing data from Roboflow")
        logger.info("=" * 60 + "\n")

        ingestion: RoboflowIngestion = RoboflowIngestion(
            api_key=self.api_key, config=self.config
        )

        # Download dataset
        downloaded_paths: List[Path] = ingestion.download_dataset()

        # Organize datasets
        data_dirs: Dict[str, Path] = {}

        for path in downloaded_paths:
            if "yolov8" in path.name.lower():
                yolo_dir: Path = ingestion.organize_yolo_format(
                    source_dir=path, target_dir=self.config.paths.roboflow_yolo
                )
                data_dirs["yolo"] = yolo_dir
            elif "coco" in path.name.lower():
                coco_dir: Path = ingestion.organize_coco_format(
                    source_dir=path, target_dir=self.config.paths.roboflow_coco
                )
                data_dirs["coco"] = coco_dir

        # Get dataset statistics
        for fmt, data_dir in data_dirs.items():
            stats: Dict = ingestion.get_dataset_statistics(data_dir)
            logger.info(f"{fmt.upper()} format: {stats}")

        return data_dirs

    def train_yolo_models(self, data_dir: Path) -> Dict[str, Path]:
        """
        Train YOLO models on Stage 1 data.

        Args:
            data_dir: Path to YOLO format data

        Returns:
            Dictionary mapping model names to checkpoint paths
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Training YOLO models")
        logger.info("=" * 60 + "\n")

        data_yaml: Path = data_dir / "data.yaml"

        if not data_yaml.exists():
            logger.error(f"YOLO data.yaml not found at {data_yaml}")
            raise FileNotFoundError(f"YOLO data.yaml not found at {data_yaml}")

        checkpoints: List[Path] = train_yolo_stage1(
            model_names=self.yolo_models, data_yaml=data_yaml, config=self.config
        )

        checkpoint_dict: Dict[str, Path] = {
            model: ckpt for model, ckpt in zip(self.yolo_models, checkpoints)
        }
        return checkpoint_dict

    def train_fasterrcnn(self, data_dir: Path) -> Path:
        """
        Train Faster R-CNN on Stage 1 data.

        Args:
            data_dir: Path to COCO format data

        Returns:
            Path to checkpoint
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Training Faster R-CNN")
        logger.info("=" * 60 + "\n")

        checkpoint_path: Path = train_fasterrcnn_stage1(
            data_dir=data_dir, backbone=self.rcnn_backbone, config=self.config
        )

        return checkpoint_path

    def run(self, download: bool = True) -> Dict[str, Path]:
        """
        Run complete Stage 1 training pipeline.

        Args:
            download: Whether to download data from Roboflow

        Returns:
            Dictionary mapping model names to checkpoint paths
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING STAGE 1: DOMAIN GENERALIZATION")
        logger.info("=" * 60 + "\n")

        try:
            # Step 1: Download and prepare data
            if download:
                data_dirs: Dict[str, Path] = self.download_and_prepare_data()
            else:
                logger.info("Skipping data download, using existing data")
                data_dirs = {
                    "yolo": self.config.paths.roboflow_yolo,
                    "coco": self.config.paths.roboflow_coco,
                }

                # Verify data exists
                if not data_dirs["yolo"].exists() or not data_dirs["coco"].exists():
                    logger.error(
                        "Data directories not found. Please download data first."
                    )
                    sys.exit(1)

            # Step 2: Train YOLO models
            yolo_checkpoints: Dict[str, Path] = self.train_yolo_models(
                data_dirs["yolo"]
            )
            self.results.update(yolo_checkpoints)

            # Step 3: Train Faster R-CNN
            rcnn_checkpoint: Path = self.train_fasterrcnn(data_dirs["coco"])
            self.results["fasterrcnn"] = rcnn_checkpoint

            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 1 TRAINING COMPLETE")
            logger.info("=" * 60 + "\n")

            logger.info("Trained models:")
            for model_name, ckpt_path in self.results.items():
                logger.info(f"  {model_name}: {ckpt_path}")

            return self.results

        except Exception as e:
            logger.error(f"Error during Stage 1 training: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


def main() -> None:
    """Main function for Stage 1 orchestration."""
    import os

    # Setup logging
    setup_logging()

    # Get API key from environment
    api_key: str = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        logger.error("ROBOFLOW_API_KEY environment variable not set")
        logger.error("Please set it with: export ROBOFLOW_API_KEY=your_api_key")
        sys.exit(1)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Domain Generalization")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip Roboflow download"
    )
    parser.add_argument(
        "--yolo-models", nargs="+", default=None, help="YOLO models to train"
    )
    parser.add_argument(
        "--rcnn-backbone", type=str, default="resnet50", help="Faster R-CNN backbone"
    )

    args = parser.parse_args()

    # Create trainer
    trainer: Stage1Trainer = Stage1Trainer(
        api_key=api_key, yolo_models=args.yolo_models, rcnn_backbone=args.rcnn_backbone
    )

    # Run Stage 1
    results: Dict[str, Path] = trainer.run(download=not args.skip_download)

    logger.info(f"\nAll Stage 1 models saved to: {results}")


if __name__ == "__main__":
    main()
