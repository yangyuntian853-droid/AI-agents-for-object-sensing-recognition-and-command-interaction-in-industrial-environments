#!/usr/bin/env bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
EDGE=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge
export YOLO_MASTER_ROOT="${EDGE}/YOLO-Master"
export PYTHONPATH=/mnt/d/dl/keyan/tiaozhanbei:$PYTHONPATH
cd "${EDGE}"
exec python train_custom.py "$@"
