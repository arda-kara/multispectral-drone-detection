"""Configuration management for drone detection pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class PathsConfig:
    """Directory paths configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_root: Path = field(default_factory=lambda: Path("data"))
    roboflow_data: Path = field(default_factory=lambda: Path("data/roboflow"))
    roboflow_yolo: Path = field(default_factory=lambda: Path("data/roboflow/yolo"))
    roboflow_coco: Path = field(default_factory=lambda: Path("data/roboflow/coco"))
    custom_data: Path = field(default_factory=lambda: Path("data/custom"))
    custom_rgb: Path = field(default_factory=lambda: Path("data/custom/rgb"))
    custom_lwir: Path = field(default_factory=lambda: Path("data/custom/lwir"))
    custom_uv: Path = field(default_factory=lambda: Path("data/custom/uv"))
    checkpoints: Path = field(default_factory=lambda: Path("checkpoints"))
    stage1_checkpoints: Path = field(default_factory=lambda: Path("checkpoints/stage1"))
    stage2_checkpoints: Path = field(default_factory=lambda: Path("checkpoints/stage2"))
    runs: Path = field(default_factory=lambda: Path("runs"))


@dataclass
class RoboflowConfig:
    """Roboflow dataset configuration."""

    workspace: str = "drone-wxuiq"
    project: str = "drone-vs-bird-combined"
    version: int = 1
    api_key: Optional[str] = None
    dataset_format_yolo: str = "yolov8"
    dataset_format_coco: str = "coco"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size_yolo: int = 16
    batch_size_rcnn: int = 4
    epochs_stage1: int = 100
    epochs_stage2: int = 50
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 1e-4
    warmup_epochs: int = 5
    momentum: float = 0.937
    weight_decay: float = 5e-4
    image_size: Tuple[int, int] = (640, 640)
    workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""

    use_mosaic: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.5
    use_perspective: bool = True
    perspective_scale: float = 0.5
    use_hsv_h: float = 0.015
    use_hsv_s: float = 0.7
    use_hsv_v: float = 0.4
    use_flipud: float = 0.5
    use_fliplr: float = 0.5
    use_rotate: float = 0.0
    use_translate: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configurations."""

    yolo_models: List[str] = field(
        default_factory=lambda: ["yolov8n", "yolov9c", "yolov11n"]
    )
    faster_rcnn_backbone: str = "resnet50"
    freeze_backbone_epochs: int = 10
    pretrained: bool = True
    input_channels_rgb: int = 3
    input_channels_lwir: int = 1
    input_channels_uv: int = 1


@dataclass
class LoggingConfig:
    """Training logging configuration."""

    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "multispectral-drone-detection"
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 5


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    compute_fps: bool = True
    save_predictions: bool = True


@dataclass
class Config:
    """Main configuration class."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        """Ensure all paths are created."""
        for path_attr in ["paths"]:
            path_obj = getattr(self, path_attr)
            if hasattr(path_obj, "__dict__"):
                for attr_name in dir(path_obj):
                    if not attr_name.startswith("_") and isinstance(
                        getattr(path_obj, attr_name), Path
                    ):
                        path = getattr(path_obj, attr_name)
                        path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config: Config = Config()
