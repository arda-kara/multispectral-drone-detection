# Multispectral Drone Detection Pipeline

A modular, two-stage object detection pipeline for drone detection using multi-modal (RGB, LWIR, UV) imagery. Supports YOLOv8, YOLOv9, YOLOv11 (Ultralytics) and Faster R-CNN (Detectron2).

## Project Structure

```
multispectral-drone-detection/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Centralized configuration
├── data/                          # Datasets
│   ├── roboflow/                  # Stage 1: Generic drone-vs-bird data
│   └── custom/                    # Stage 2: Multi-modal custom dataset
├── train/                         # Training pipeline
│   ├── roboflow_ingestion.py      # Roboflow download & format routing
│   ├── multimodal_dataloaders.py  # Multi-modal Dataset/DataLoader
│   ├── train_yolo.py              # YOLO training
│   ├── train_rcnn.py              # Faster R-CNN training
│   ├── train_stage1.py            # Stage 1 orchestration
│   └── train_stage2.py            # Stage 2 orchestration
├── inference/                     # Future inference pipeline
├── utils/                         # Shared utilities
│   ├── model_utils.py             # Model adaptation utilities
│   ├── metrics.py                 # Evaluation metrics
│   └── logger.py                  # Logging utilities
├── evaluate/                      # Evaluation module
│   └── evaluate_models.py         # Model benchmarking
└── checkpoints/                   # Saved model weights
    ├── stage1/                    # Pre-trained models
    └── stage2/                    # Fine-tuned models
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM)
- Roboflow API key

### Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Detectron2 requires installation from source:

```bash
python -m pip install 'torch==2.0.0+cu117' -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install 'detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html'
```

## Configuration

Set your Roboflow API key:

```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

## Training Pipeline

### Stage 1: Domain Generalization (Pre-training)

Pre-train models on generic drone-vs-bird data from Roboflow to learn small-object features.

```bash
# Train all models (YOLOv8, YOLOv9, YOLOv11, Faster R-CNN)
python train/train_stage1.py

# Skip Roboflow download if data already exists
python train/train_stage1.py --skip-download

# Train specific models only
python train/train_stage1.py --yolo-models yolov8n yolov9c --rcnn-backbone resnet50
```

### Stage 2: Domain Adaptation (Fine-tuning)

Fine-tune pre-trained models on custom multi-modal data.

```bash
# Fine-tune all models on all modalities
python train/train_stage2.py --data data/custom

# Fine-tune specific modalities
python train/train_stage2.py --data data/custom --modalities rgb lwir

# Fine-tune a single model
python train/train_stage2.py --data data/custom --single-model yolov8n

# Use specific YOLO models
python train/train_stage2.py --data data/custom --yolo-models yolov8n yolov9c
```

## Data Organization

### Stage 1 Data (Roboflow)

Automatically downloaded to `data/roboflow/`:
- `data/roboflow/yolo/` - YOLO format for Ultralytics models
- `data/roboflow/coco/` - COCO format for Faster R-CNN

### Stage 2 Data (Custom Multi-modal)

Organize your custom dataset as:

```
data/custom/
├── rgb/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── lwir/
│   ├── images/
│   └── annotations/
└── uv/
    ├── images/
    └── annotations/
```

Or use the simpler YOLO format:

```
data/custom/
├── rgb/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── [lwir, uv similarly]
```

## Evaluation

Evaluate all trained models:

```bash
# Evaluate Stage 1 models
python evaluate/evaluate_models.py --checkpoints checkpoints/stage1 --data data/roboflow/yolo

# Evaluate Stage 2 models
python evaluate/evaluate_models.py --checkpoints checkpoints/stage2 --data data/custom --output results/

# The evaluation script will:
# - Load all checkpoints
# - Run inference on test data
# - Calculate mAP, precision, recall, FPS
# - Generate CSV and JSON reports
```

## Model Details

### YOLO Models (Ultralytics)
- **YOLOv8**: Latest generation YOLO with improved accuracy and speed
- **YOLOv9**: Architectural improvements for better small-object detection
- **YOLOv11**: Experimental version with new features

### Faster R-CNN (Detectron2)
- **Backbone**: ResNet50 or ResNet101 with FPN
- **Region Proposal Network**: Standard RPN
- **ROI Head**: Multi-scale feature extraction

## Channel Adaptation

The pipeline automatically adapts model input layers for multi-modal training:
- **RGB**: 3-channel input (standard)
- **LWIR**: 1-channel input (grayscale thermal)
- **UV**: 1-channel input (grayscale ultraviolet)

Channel adaptation preserves pre-trained weights by averaging RGB channels for grayscale inputs.

## Transfer Learning Strategy

### Stage 1 (Pre-training)
- Full model training
- High learning rate (1e-3)
- Data augmentations: Mosaic, MixUp, RandomPerspective
- 100 epochs (default)

### Stage 2 (Fine-tuning)
- Load Stage 1 weights
- Freeze backbone for initial epochs (10 default)
- Low learning rate (1e-4)
- Cosine annealing schedule
- 50 epochs (default)
- Disable Mosaic/MixUp

## Logging and Monitoring

Training logs are saved to:
- **TensorBoard**: `runs/` directory
- **WandB**: Optional (configure in `config.py`)
- **Checkpoints**: `checkpoints/stage1/` and `checkpoints/stage2/`

View TensorBoard:

```bash
tensorboard --logdir runs
```

## Configuration

Edit `config.py` to customize:
- Training hyperparameters
- Data augmentation settings
- Model architectures
- Logging preferences
- GPU configuration

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.py`
- Use smaller model variants (e.g., `yolov8n` instead of `yolov8x`)
- Enable mixed precision training (set `AMP=True` in config)

### Roboflow Download Fails
- Verify API key is set correctly
- Check internet connection
- Ensure Roboflow project is public or you have access

### Detectron2 Import Errors
- Ensure Detectron2 is installed from source
- Check PyTorch/Detectron2 version compatibility
- Verify CUDA version matches

## Future Development

The `inference/` directory is reserved for future implementation of:
- Single image/video inference
- Batch processing
- Real-time stream processing
- Model export (ONNX, TensorRT)
- Visualization tools

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multispectral_drone_detection,
  title = {Multispectral Drone Detection Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/multispectral-drone-detection}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact: your.email@example.com
