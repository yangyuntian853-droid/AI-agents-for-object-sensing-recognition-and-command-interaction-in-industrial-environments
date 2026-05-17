#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
EDGE=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge
REPO="${EDGE}/YOLO-Master"
export YOLO_MASTER_ROOT="${REPO}"
export PYTHONPATH=/mnt/d/dl/keyan/tiaozhanbei:$PYTHONPATH
cd "${REPO}"
pip install -e . -q
python <<'PY'
import ultralytics
import torch
print("ultralytics:", ultralytics.__file__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
import yolo_master_edge
print("yolo_master_edge:", yolo_master_edge.__file__)
PY
ls "${EDGE}/dataset/images/train/999."* 2>/dev/null | head -1
echo "verify_env OK"
