#!/usr/bin/env bash
set -e
LOGDIR=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/artifacts/train_logs
mkdir -p "$LOGDIR"
EDGE=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
export YOLO_MASTER_ROOT="${EDGE}/YOLO-Master"
export PYTHONPATH=/mnt/d/dl/keyan/tiaozhanbei:$PYTHONPATH
cd "${EDGE}"

DEVICE="${TRAIN_DEVICE:-cpu}"
BATCH="${TRAIN_BATCH:-4}"

echo "=== finetune 50 epochs (device=$DEVICE) ===" | tee "$LOGDIR/finetune.log"
python train_custom.py --mode finetune --variant n --epochs 50 --batch "$BATCH" \
  --device "$DEVICE" --name ft_n --workers 2 \
  2>&1 | tee -a "$LOGDIR/finetune.log"

echo "=== scratch 100 epochs (device=$DEVICE) ===" | tee "$LOGDIR/scratch.log"
python train_custom.py --mode scratch --variant n --epochs 100 --batch "$BATCH" \
  --device "$DEVICE" --name scratch_n --workers 2 \
  2>&1 | tee -a "$LOGDIR/scratch.log"

echo "=== val + export onnx (finetune best) ===" | tee "$LOGDIR/val_export.log"
python train_custom.py --mode finetune --variant n --epochs 1 --batch 1 --device "$DEVICE" \
  --name ft_n_dummy --val-after --export-onnx 2>&1 | tee -a "$LOGDIR/val_export.log" || true

python <<'PY' | tee -a "$LOGDIR/val_export.log"
from pathlib import Path
from yolo_master_edge import YoloMasterDetectionPipeline, YoloMasterVariant

data = "/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/dataset/data.yaml"
best = Path("/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/artifacts/yolo_master_runs/ft_n/weights/best.pt")
pipe = YoloMasterDetectionPipeline(variant=YoloMasterVariant.V01_N)
if best.is_file():
    pipe.load(model_spec=best)
    pipe.val(data)
    pipe.export(format="onnx")
    print("val+export OK", best)
else:
    raise SystemExit(f"missing {best}")
PY

echo "train_all DONE" | tee -a "$LOGDIR/finetune.log"
