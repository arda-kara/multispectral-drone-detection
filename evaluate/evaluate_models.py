"""Model evaluation and comparison script."""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import logging
import json
import time
from datetime import datetime

import torch
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

ULTRALYTICS_AVAILABLE: bool = YOLO is not None

try:
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog
except ImportError:
    detectron2 = None
    DefaultPredictor = None
    get_cfg = None
    model_zoo = None
    register_coco_instances = None
    MetadataCatalog = None

DETECTRON2_AVAILABLE: bool = detectron2 is not None

from config import Config
from utils.metrics import calculate_map, calculate_precision_recall, calculate_iou
from utils.logger import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for object detection models."""

    def __init__(
        self, checkpoint_dir: Path, data_dir: Path, config: Optional[Config] = None
    ) -> None:
        """
        Initialize model evaluator.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            data_dir: Directory containing evaluation data
            config: Configuration object
        """
        self.checkpoint_dir: Path = checkpoint_dir
        self.data_dir: Path = data_dir
        self.config: Config = config or Config()

        self.results: Dict[str, Dict[str, Any]] = {}
        self.device: str = (
            self.config.training.device if torch.cuda.is_available() else "cpu"
        )

        # Setup logging
        setup_logging()

    def find_checkpoints(self) -> Dict[str, Path]:
        """
        Find all model checkpoints in the directory.

        Returns:
            Dictionary mapping model names to checkpoint paths
        """
        checkpoints: Dict[str, Path] = {}

        # Find YOLO checkpoints
        if ULTRALYTICS_AVAILABLE:
            for ckpt_path in self.checkpoint_dir.glob("*/weights/best.pt"):
                model_name = ckpt_path.parent.parent.name
                checkpoints[model_name] = ckpt_path
                logger.info(f"Found YOLO checkpoint: {model_name}")

        # Find Faster R-CNN checkpoints
        if DETECTRON2_AVAILABLE:
            for ckpt_path in self.checkpoint_dir.glob("*/model_final.pth"):
                model_name = ckpt_path.parent.name
                checkpoints[model_name] = ckpt_path
                logger.info(f"Found Faster R-CNN checkpoint: {model_name}")

        return checkpoints

    def load_yolo_model(self, checkpoint_path: Path) -> Any:
        """
        Load YOLO model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Loaded YOLO model
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available")

        model: YOLO = YOLO(str(checkpoint_path))
        model.to(self.device)
        return model

    def load_fasterrcnn_model(
        self, checkpoint_path: Path, modality: str = "rgb"
    ) -> DefaultPredictor:
        """
        Load Faster R-CNN model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            modality: Modality ('rgb', 'lwir', 'uv')

        Returns:
            Loaded Faster R-CNN predictor
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 not available")

        # Determine config file
        config_file: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

        cfg: Any = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = str(checkpoint_path)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = self.device

        # Adjust for modality
        if modality in ["lwir", "uv"]:
            cfg.MODEL.PIXEL_MEAN = [128.0]
            cfg.MODEL.PIXEL_STD = [128.0]

        predictor: DefaultPredictor = DefaultPredictor(cfg)
        return predictor

    def evaluate_yolo(
        self, model: YOLO, data_dir: Path, conf_threshold: float = 0.25
    ) -> Dict[str, float]:
        """
        Evaluate YOLO model.

        Args:
            model: YOLO model
            data_dir: Path to evaluation data
            conf_threshold: Confidence threshold

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating YOLO model...")

        # Get predictions
        results = model.val(
            data=str(data_dir),
            conf=conf_threshold,
            iou=self.config.evaluation.iou_threshold,
            device=self.device,
        )

        metrics: Dict[str, float] = {
            "mAP50": results.metrics.get("metrics/mAP50(B)", 0),
            "mAP50_95": results.metrics.get("metrics/mAP50-95(B)", 0),
            "precision": results.metrics.get("metrics/precision(B)", 0),
            "recall": results.metrics.get("metrics/recall(B)", 0),
        }

        return metrics

    def evaluate_fasterrcnn(
        self, predictor: DefaultPredictor, data_dir: Path, modality: str = "rgb"
    ) -> Dict[str, float]:
        """
        Evaluate Faster R-CNN model.

        Args:
            predictor: Faster R-CNN predictor
            data_dir: Path to evaluation data
            modality: Modality

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating Faster R-CNN model...")

        # Load evaluation data
        from train.multimodal_dataloaders import MultiModalDataset

        test_dataset: MultiModalDataset = MultiModalDataset(
            data_dir=data_dir,
            modality=modality,
            split="test",
            image_size=self.config.training.image_size,
        )

        # Collect predictions and targets
        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        inference_times: List[float] = []

        for i in range(len(test_dataset)):
            image, bboxes, _ = test_dataset[i]

            # Measure inference time
            start_time: float = time.time()
            outputs: Dict = predictor(image.permute(1, 2, 0).cpu().numpy() * 255)
            inference_time: float = time.time() - start_time
            inference_times.append(inference_time)

            # Extract predictions
            pred_boxes: np.ndarray = (
                outputs["instances"].pred_boxes.tensor.cpu().numpy()
            )
            pred_scores: np.ndarray = outputs["instances"].scores.cpu().numpy()

            if len(pred_boxes) > 0:
                # Filter by confidence
                conf_mask: np.ndarray = (
                    pred_scores >= self.config.evaluation.confidence_threshold
                )
                pred_boxes = pred_boxes[conf_mask]
                pred_scores = pred_scores[conf_mask]

                # Format: [x1, y1, x2, y2, conf, class]
                predictions: np.ndarray = np.zeros((len(pred_boxes), 6))
                predictions[:, :4] = pred_boxes
                predictions[:, 4] = pred_scores
                predictions[:, 5] = 0  # Single class
            else:
                predictions = np.zeros((0, 6))

            all_predictions.append(torch.tensor(predictions))

            # Extract targets
            if bboxes is not None and len(bboxes) > 0:
                targets: np.ndarray = bboxes.numpy()
            else:
                targets = np.zeros((0, 5))

            all_targets.append(torch.tensor(targets))

        # Calculate metrics
        map_05_95: float
        map_05: float
        map_per_iou: Dict[float, float]

        map_05_95, map_05, map_per_iou = calculate_map(
            predictions=all_predictions, targets=all_targets, num_classes=1
        )

        # Calculate precision/recall
        pr_metrics: Dict[int, Dict[str, float]] = calculate_precision_recall(
            predictions=all_predictions,
            targets=all_targets,
            confidence_threshold=self.config.evaluation.confidence_threshold,
            num_classes=1,
        )

        # Calculate FPS
        avg_inference_time: float = float(np.mean(inference_times))
        fps: float = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        metrics: Dict[str, float] = {
            "mAP50": map_05,
            "mAP50_95": map_05_95,
            "precision": pr_metrics.get(0, {}).get("precision", 0),
            "recall": pr_metrics.get(0, {}).get("recall", 0),
            "f1": pr_metrics.get(0, {}).get("f1", 0),
            "fps": fps,
        }

        return metrics

    def evaluate_model(
        self, model_name: str, checkpoint_path: Path, modality: str = "rgb"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.

        Args:
            model_name: Name of the model
            checkpoint_path: Path to model checkpoint
            modality: Modality ('rgb', 'lwir', 'uv')

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name} ({modality})")

        # Determine model type and load
        if "yolo" in model_name.lower():
            model: YOLO = self.load_yolo_model(checkpoint_path)
            yolo_eval_metrics: Dict[str, float] = self.evaluate_yolo(
                model=model, data_dir=self.data_dir / modality
            )
            final_eval_metrics: Dict[str, Any] = yolo_eval_metrics
        elif "fasterrcnn" in model_name.lower() or "rcnn" in model_name.lower():
            predictor: DefaultPredictor = self.load_fasterrcnn_model(
                checkpoint_path, modality
            )
            rcnn_eval_metrics: Dict[str, float] = self.evaluate_fasterrcnn(
                predictor=predictor, data_dir=self.data_dir, modality=modality
            )
            final_eval_metrics: Dict[str, Any] = rcnn_eval_metrics
        else:
            logger.error(f"Unknown model type: {model_name}")
            return {}

        # Add metadata
        final_eval_metrics["model_name"] = model_name
        final_eval_metrics["modality"] = modality
        final_eval_metrics["checkpoint_path"] = str(checkpoint_path)
        final_eval_metrics["evaluated_at"] = datetime.now().isoformat()

        logger.info(f"Results: {final_eval_metrics}")

        return final_eval_metrics

    def evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models in the checkpoint directory.

        Returns:
            Dictionary mapping model names to metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 60 + "\n")

        # Find checkpoints
        checkpoints: Dict[str, Path] = self.find_checkpoints()

        if not checkpoints:
            logger.warning("No checkpoints found in directory")
            return {}

        # Evaluate each model
        for model_name, checkpoint_path in checkpoints.items():
            try:
                # Determine modality from model name or path
                if "lwir" in model_name.lower():
                    modality: str = "lwir"
                elif "uv" in model_name.lower():
                    modality = "uv"
                else:
                    modality = "rgb"

                metrics: Dict[str, float] = self.evaluate_model(
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    modality=modality,
                )

                self.results[model_name] = metrics

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                import traceback

                traceback.print_exc()

        return self.results

    def save_results(
        self, output_path: Optional[Path] = None, format: str = "csv"
    ) -> None:
        """
        Save evaluation results.

        Args:
            output_path: Path to save results
            format: Format ('csv', 'json')
        """
        if not output_path:
            output_path = self.checkpoint_dir / f"evaluation_results.{format}"

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2)
        elif format == "csv":
            import pandas as pd

            df: pd.DataFrame = pd.DataFrame.from_dict(self.results, orient="index")
            df.to_csv(output_path, index=True)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self) -> None:
        """Print evaluation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60 + "\n")

        # Group by modality
        modalities: Dict[str, List[str]] = {}
        for model_name, model_metrics in self.results.items():
            modality: str = model_metrics.get("modality", "unknown")
            if modality not in modalities:
                modalities[modality] = []
            modalities[modality].append(model_name)

        for modality, model_names in modalities.items():
            logger.info(f"\n{modality.upper()} Modality:")
            logger.info("-" * 40)

            # Sort by mAP
            sorted_models: List[str] = sorted(
                model_names, key=lambda x: self.results[x].get("mAP50", 0), reverse=True
            )

            for model_name in sorted_models:
                model_results: Dict[str, Any] = self.results[model_name]
                logger.info(f"  {model_name}:")
                logger.info(f"    mAP@50:     {model_results.get('mAP50', 0):.4f}")
                logger.info(f"    mAP@50:95:  {model_results.get('mAP50_95', 0):.4f}")
                logger.info(f"    Precision:  {model_results.get('precision', 0):.4f}")
                logger.info(f"    Recall:     {model_results.get('recall', 0):.4f}")
                logger.info(f"    FPS:        {model_results.get('fps', 0):.2f}")


def evaluate_all_models(
    checkpoint_dir: Path,
    data_dir: Path,
    output_dir: Optional[Path] = None,
    config: Optional[Config] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        data_dir: Directory containing evaluation data
        output_dir: Directory to save results
        config: Configuration object

    Returns:
        Dictionary with evaluation results
    """
    evaluator: ModelEvaluator = ModelEvaluator(
        checkpoint_dir=checkpoint_dir, data_dir=data_dir, config=config
    )

    # Evaluate all models
    results: Dict[str, Dict[str, float]] = evaluator.evaluate_all()

    # Print summary
    evaluator.print_summary()

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator.save_results(output_dir / "evaluation_results.json", format="json")
        evaluator.save_results(output_dir / "evaluation_results.csv", format="csv")

    return results


def main() -> None:
    """Main function for model evaluation."""
    import sys
    import argparse

    # Setup logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--checkpoints", type=str, required=True, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to evaluation data"
    )
    parser.add_argument("--output", type=str, help="Path to output directory")

    args = parser.parse_args()

    try:
        results: Dict[str, Dict[str, float]] = evaluate_all_models(
            checkpoint_dir=Path(args.checkpoints),
            data_dir=Path(args.data),
            output_dir=Path(args.output) if args.output else None,
        )

        logger.info("\nEvaluation complete!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
