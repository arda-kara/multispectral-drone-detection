"""Multi-modal data loaders for Stage 2 fine-tuning."""

from typing import Optional, Tuple, List, Dict, Any, Callable
from pathlib import Path
import logging
import json

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import Config

logger: logging.Logger = logging.getLogger(__name__)


class MultiModalTransform:
    """Transformations for multi-modal data."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        modality: str = "rgb",
    ) -> None:
        """
        Initialize transformations.

        Args:
            image_size: Target image size
            augment: Whether to apply augmentations
            modality: Data modality ('rgb', 'lwir', 'uv')
        """
        self.image_size: Tuple[int, int] = image_size
        self.augment: bool = augment
        self.modality: str = modality

    def __call__(
        self, image: np.ndarray, bbox: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply transformations to image and bounding box.

        Args:
            image: Input image (H, W, C) for RGB or (H, W) for grayscale
            bbox: Optional bounding box [x1, y1, x2, y2, class]

        Returns:
            Tuple of (transformed_image, transformed_bbox)
        """
        # Convert to PIL Image
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode="L")
        else:
            # RGB
            pil_image = Image.fromarray(image)

        # Resize
        pil_image = TF.resize(pil_image, self.image_size)

        # Convert to tensor
        if self.modality == "rgb":
            tensor_image = TF.to_tensor(pil_image)  # (C, H, W)
        else:
            # Grayscale for LWIR/UV
            tensor_image = TF.to_tensor(pil_image)  # (1, H, W)
            # Ensure single channel
            tensor_image = tensor_image[:1, :, :]

        # Normalize
        if self.modality == "rgb":
            tensor_image = TF.normalize(
                tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            tensor_image = TF.normalize(tensor_image, mean=[0.5], std=[0.5])

        # Transform bbox if provided
        transformed_bbox: Optional[torch.Tensor] = None
        if bbox is not None:
            # bbox is [x1, y1, x2, y2, class] in absolute coordinates
            # Need to normalize to [0, 1] then resize
            h, w = image.shape[:2]
            bbox_norm: np.ndarray = bbox.copy()
            bbox_norm[0] /= w
            bbox_norm[1] /= h
            bbox_norm[2] /= w
            bbox_norm[3] /= h
            transformed_bbox = torch.tensor(bbox_norm, dtype=torch.float32)

        return tensor_image, transformed_bbox


class MultiModalDataset(Dataset):
    """Multi-modal dataset for drone detection."""

    def __init__(
        self,
        data_dir: Path,
        modality: str = "rgb",
        split: str = "train",
        image_size: Tuple[int, int] = (640, 640),
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize multi-modal dataset.

        Args:
            data_dir: Root data directory
            modality: Modality ('rgb', 'lwir', 'uv')
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size
            transform: Optional custom transform
        """
        self.data_dir: Path = data_dir
        self.modality: str = modality
        self.split: str = split
        self.image_size: Tuple[int, int] = image_size

        # Set default transform
        if transform is None:
            self.transform: Callable = MultiModalTransform(
                image_size=image_size, augment=(split == "train"), modality=modality
            )
        else:
            self.transform = transform

        # Load annotations
        self.annotations: List[Dict[str, Any]] = self._load_annotations()
        logger.info(f"Loaded {len(self.annotations)} samples for {modality} {split}")

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load annotations from JSON file.

        Returns:
            List of annotation dictionaries
        """
        annotation_file: Path = self.data_dir / f"{self.split}.json"

        if not annotation_file.exists():
            # Try alternative locations
            alt_paths: List[Path] = [
                self.data_dir / "annotations" / f"{self.split}.json",
                self.data_dir / self.modality / f"{self.split}.json",
                self.data_dir / "labels" / f"{self.split}.json",
            ]

            for alt_path in alt_paths:
                if alt_path.exists():
                    annotation_file = alt_path
                    break

        if not annotation_file.exists():
            # Create simple directory-based annotations
            return self._create_dir_based_annotations()

        with open(annotation_file, "r") as f:
            data: Dict[str, Any] = json.load(f)

        annotations: List[Dict[str, Any]] = []

        if "annotations" in data:
            # COCO format
            image_id_to_annotations: Dict[int, List[Dict]] = {}
            for ann in data["annotations"]:
                image_id: int = ann["image_id"]
                if image_id not in image_id_to_annotations:
                    image_id_to_annotations[image_id] = []
                image_id_to_annotations[image_id].append(ann)

            image_id_to_info: Dict[int, Dict] = {
                img["id"]: img for img in data["images"]
            }

            for image_id, anns in image_id_to_annotations.items():
                img_info: Dict = image_id_to_info[image_id]
                annotations.append(
                    {
                        "image_path": str(
                            self.data_dir / self.modality / img_info["file_name"]
                        ),
                        "bboxes": self._parse_coco_annotations(anns),
                    }
                )
        else:
            # Simple format
            for item in data.get("data", []):
                annotations.append(
                    {
                        "image_path": str(
                            self.data_dir / self.modality / item["image"]
                        ),
                        "bboxes": item.get("bboxes", []),
                    }
                )

        return annotations

    def _create_dir_based_annotations(self) -> List[Dict[str, Any]]:
        """
        Create annotations based on directory structure.

        Returns:
            List of annotation dictionaries
        """
        annotations: List[Dict[str, Any]] = []

        # Look for images directory
        images_dir: Path = self.data_dir / self.modality / "images" / self.split

        if not images_dir.exists():
            # Try alternative structure
            alt_paths: List[Path] = [
                self.data_dir / "images" / self.split,
                self.data_dir / self.split,
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    images_dir = alt_path
                    break

        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return []

        # Look for corresponding labels
        labels_dir: Path = self.data_dir / self.modality / "labels" / self.split
        if not labels_dir.exists():
            labels_dir = self.data_dir / "labels" / self.split

        # Create annotation entries
        for image_path in images_dir.glob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:
                continue

            # Try to find corresponding label file
            label_path: Path = labels_dir / f"{image_path.stem}.txt"

            bboxes: List[List[float]] = []

            if label_path.exists():
                # Read YOLO format labels
                with open(label_path, "r") as f:
                    for line in f:
                        parts: List[str] = line.strip().split()
                        if len(parts) >= 5:
                            class_id: int = int(parts[0])
                            x_center: float = float(parts[1])
                            y_center: float = float(parts[2])
                            width: float = float(parts[3])
                            height: float = float(parts[4])
                            bboxes.append([x_center, y_center, width, height, class_id])

            annotations.append({"image_path": str(image_path), "bboxes": bboxes})

        return annotations

    def _parse_coco_annotations(self, annotations: List[Dict]) -> List[List[float]]:
        """
        Parse COCO format annotations.

        Args:
            annotations: List of COCO annotations

        Returns:
            List of bounding boxes [x1, y1, x2, y2, class]
        """
        bboxes: List[List[float]] = []

        for ann in annotations:
            bbox: List[float] = ann["bbox"]  # [x, y, width, height]
            class_id: int = ann["category_id"]

            # Convert to [x1, y1, x2, y2] format
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            bboxes.append([x1, y1, x2, y2, class_id])

        return bboxes

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, bboxes, image_path)
        """
        ann: Dict[str, Any] = self.annotations[idx]
        image_path: Path = Path(ann["image_path"])

        # Load image
        try:
            image: np.ndarray = np.array(Image.open(image_path))

            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Create blank image
            if self.modality == "rgb":
                image = np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                )
            else:
                image = np.zeros(
                    (self.image_size[0], self.image_size[1]), dtype=np.uint8
                )

        # Convert to grayscale for LWIR/UV
        if self.modality in ["lwir", "uv"]:
            image = np.mean(image, axis=-1).astype(np.uint8)

        # Get bounding boxes
        bboxes: List[List[float]] = ann.get("bboxes", [])

        # Apply transformations
        tensor_image: torch.Tensor
        tensor_bboxes: Optional[torch.Tensor]

        if bboxes:
            # Convert bboxes to numpy array
            bbox_array: np.ndarray = np.array(bboxes)
            tensor_image, tensor_bboxes = self.transform(image, bbox_array)
        else:
            tensor_image, tensor_bboxes = self.transform(image, None)
            tensor_bboxes = torch.zeros((0, 5), dtype=torch.float32)

        return tensor_image, tensor_bboxes, str(image_path)


def create_multimodal_dataloaders(
    data_dir: Path,
    modality: str = "rgb",
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640),
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Root data directory
        modality: Data modality
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset: MultiModalDataset = MultiModalDataset(
        data_dir=data_dir, modality=modality, split="train", image_size=image_size
    )

    val_dataset: MultiModalDataset = MultiModalDataset(
        data_dir=data_dir, modality=modality, split="val", image_size=image_size
    )

    test_dataset: MultiModalDataset = MultiModalDataset(
        data_dir=data_dir, modality=modality, split="test", image_size=image_size
    )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    """Test multi-modal dataloader."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create test dataloader
    try:
        train_loader, val_loader, test_loader = create_multimodal_dataloaders(
            data_dir=Path("data/custom"), modality="rgb", batch_size=4, num_workers=0
        )

        # Test loading a batch
        for images, bboxes, paths in train_loader:
            print(f"Images shape: {images.shape}")
            print(f"Bboxes shape: {bboxes.shape}")
            print(f"Paths: {paths[:2]}")
            break

        print("\nDataloader test successful!")

    except Exception as e:
        logger.error(f"Error during dataloader test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
