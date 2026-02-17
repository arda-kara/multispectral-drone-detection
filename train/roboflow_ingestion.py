"""Roboflow data ingestion and format conversion."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import shutil

import yaml

try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None
    print("Roboflow package not available. Install with: pip install roboflow")

ROBOFLOW_AVAILABLE: bool = Roboflow is not None

from config import Config

logger: logging.Logger = logging.getLogger(__name__)


class RoboflowIngestion:
    """Handle Roboflow dataset download and format conversion."""

    def __init__(self, api_key: str, config: Optional[Config] = None) -> None:
        """
        Initialize Roboflow ingestion.

        Args:
            api_key: Roboflow API key
            config: Configuration object
        """
        self.api_key: str = api_key
        self.config: Config = config or Config()
        self.rf: Optional["Roboflow"] = None

        if ROBOFLOW_AVAILABLE:
            self.rf = Roboflow(api_key=self.api_key)
        else:
            raise ImportError(
                "Roboflow package is required. Install with: pip install roboflow"
            )

    def download_dataset(
        self,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[int] = None,
        formats: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Download dataset from Roboflow in specified formats.

        Args:
            workspace: Roboflow workspace name
            project: Roboflow project name
            version: Dataset version
            formats: List of formats to download ('yolov8', 'coco')

        Returns:
            List of downloaded dataset directories
        """
        workspace = workspace or self.config.roboflow.workspace
        project = project or self.config.roboflow.project
        version = version or self.config.roboflow.version
        formats = formats or ["yolov8", "coco"]

        logger.info(
            f"Downloading dataset from Roboflow: {workspace}/{project}/v{version}"
        )

        downloaded_paths: List[Path] = []

        for format_type in formats:
            try:
                if self.rf is None:
                    raise ImportError("Roboflow package not available")
                dataset = self.rf.workspace(workspace).project(project).version(version)
                download_path: Path = self.config.paths.roboflow_data / format_type

                # Download dataset
                dataset.download(
                    model_format=format_type,
                    location=str(download_path),
                    overwrite=True,
                )

                logger.info(f"Downloaded {format_type} format to {download_path}")
                downloaded_paths.append(download_path)

            except Exception as e:
                logger.error(f"Failed to download {format_type} format: {e}")
                raise

        return downloaded_paths

    def organize_yolo_format(self, source_dir: Path, target_dir: Path) -> Path:
        """
        Organize YOLO format data into standard structure.

        Args:
            source_dir: Source directory from Roboflow download
            target_dir: Target directory for organized data

        Returns:
            Organized dataset directory
        """
        logger.info(f"Organizing YOLO format data from {source_dir} to {target_dir}")

        target_dir.mkdir(parents=True, exist_ok=True)

        # Expected YOLO structure: train, valid, test with images and labels
        splits: List[str] = ["train", "valid", "test"]

        for split in splits:
            split_dir: Path = source_dir / split

            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            # Create target directories
            images_dir: Path = target_dir / "images" / split
            labels_dir: Path = target_dir / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for image_path in (split_dir / "images").glob("*"):
                if image_path.is_file():
                    shutil.copy2(image_path, images_dir / image_path.name)

            for label_path in (split_dir / "labels").glob("*"):
                if label_path.is_file():
                    shutil.copy2(label_path, labels_dir / label_path.name)

            logger.info(f"Processed {split}: {len(list(images_dir.glob('*')))} images")

        # Create data.yaml for YOLO
        self._create_yolo_data_yaml(target_dir)

        logger.info(f"YOLO format data organized at {target_dir}")
        return target_dir

    def organize_coco_format(self, source_dir: Path, target_dir: Path) -> Path:
        """
        Organize COCO format data into standard structure.

        Args:
            source_dir: Source directory from Roboflow download
            target_dir: Target directory for organized data

        Returns:
            Organized dataset directory
        """
        logger.info(f"Organizing COCO format data from {source_dir} to {target_dir}")

        target_dir.mkdir(parents=True, exist_ok=True)

        # Expected COCO structure: train, valid, test with annotations.json
        splits: List[str] = ["train", "valid", "test"]

        for split in splits:
            split_dir: Path = source_dir / split

            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            # Create target directories
            images_dir: Path = target_dir / "images" / split
            images_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            for image_path in split_dir.glob("*.jpg"):
                if image_path.is_file():
                    shutil.copy2(image_path, images_dir / image_path.name)

            # Copy annotations
            ann_file: Path = split_dir / "_annotations.coco.json"
            if ann_file.exists():
                target_ann: Path = target_dir / "annotations" / f"{split}.json"
                target_ann.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ann_file, target_ann)

            logger.info(f"Processed {split}: {len(list(images_dir.glob('*')))} images")

        logger.info(f"COCO format data organized at {target_dir}")
        return target_dir

    def _create_yolo_data_yaml(self, dataset_dir: Path) -> None:
        """
        Create data.yaml file for YOLO training.

        Args:
            dataset_dir: Dataset directory
        """
        data_yaml: Dict[str, Any] = {
            "path": str(dataset_dir),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "nc": 2,  # Assuming 2 classes: drone, bird
            "names": {0: "bird", 1: "drone"},
        }

        yaml_path: Path = dataset_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        logger.info(f"Created data.yaml at {yaml_path}")

    def get_dataset_statistics(self, dataset_dir: Path) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Args:
            dataset_dir: Dataset directory

        Returns:
            Dictionary with dataset statistics
        """
        stats: Dict[str, Any] = {"splits": {}, "total_images": 0}

        # Check if YOLO format
        if (dataset_dir / "images").exists():
            for split in ["train", "valid", "test"]:
                images_dir: Path = dataset_dir / "images" / split
                if images_dir.exists():
                    count: int = len(list(images_dir.glob("*")))
                    stats["splits"][split] = count
                    stats["total_images"] += count

        logger.info(f"Dataset statistics: {stats}")
        return stats


def main() -> None:
    """Main function for testing Roboflow ingestion."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get API key from environment or command line
    import os

    api_key: str = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable not set")
        print("Usage: export ROBOFLOW_API_KEY=your_api_key")
        sys.exit(1)

    # Initialize ingestion
    ingestion: RoboflowIngestion = RoboflowIngestion(api_key=api_key)

    try:
        # Download dataset
        downloaded_paths: List[Path] = ingestion.download_dataset()

        # Organize datasets
        for path in downloaded_paths:
            if "yolo" in path.name.lower():
                ingestion.organize_yolo_format(
                    source_dir=path, target_dir=Path("data/roboflow/yolo")
                )
            elif "coco" in path.name.lower():
                ingestion.organize_coco_format(
                    source_dir=path, target_dir=Path("data/roboflow/coco")
                )

        print("\nDataset download and organization complete!")

    except Exception as e:
        logger.error(f"Error during dataset ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
