#!/bin/bash
# Setup script for multispectral drone detection pipeline

set -e

echo "================================"
echo "Multispectral Drone Detection Pipeline Setup"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo ""
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
echo ""
echo "Installing Detectron2..."
python -m pip install 'detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html'

# Install other dependencies
echo ""
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p data/roboflow/yolo
mkdir -p data/roboflow/coco
mkdir -p data/custom/rgb
mkdir -p data/custom/lwir
mkdir -p data/custom/uv
mkdir -p checkpoints/stage1
mkdir -p checkpoints/stage2
mkdir -p runs

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Set your Roboflow API key:"
echo "   export ROBOFLOW_API_KEY=your_api_key_here"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run Stage 1 training:"
echo "   python train/train_stage1.py"
echo ""
echo "4. Run Stage 2 training:"
echo "   python train/train_stage2.py --data data/custom"
echo ""
echo "5. Evaluate models:"
echo "   python evaluate/evaluate_models.py --checkpoints checkpoints/stage2 --data data/custom"
echo ""
