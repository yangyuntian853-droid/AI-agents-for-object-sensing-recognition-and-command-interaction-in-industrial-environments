#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
pip uninstall -y torch torchvision 2>/dev/null || true
pip cache purge 2>/dev/null || true
# cu128 wheels bundle CUDA libs; avoid pulling extra nvidia-* from PyPI
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
