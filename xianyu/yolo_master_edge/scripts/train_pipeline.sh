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

DEVICE="${TRAIN_DEVICE:-0}"
BATCH="${TRAIN_BATCH:-8}"

run_train() {
  python train_custom.py "$@"
}

echo "[$(date)] finetune 50 epochs device=$DEVICE" | tee "$LOGDIR/pipeline.log"
run_train --mode finetune --variant n --epochs 50 --batch "$BATCH" \
  --device "$DEVICE" --name ft_n --workers 4 \
  2>&1 | tee "$LOGDIR/ft_n.log"

echo "[$(date)] scratch 100 epochs device=$DEVICE" | tee -a "$LOGDIR/pipeline.log"
run_train --mode scratch --variant n --epochs 100 --batch "$BATCH" \
  --device "$DEVICE" --name scratch_n --workers 4 \
  2>&1 | tee "$LOGDIR/scratch_n.log"

echo "[$(date)] val + export onnx (ft_n best)" | tee -a "$LOGDIR/pipeline.log"
python <<'PY' 2>&1 | tee "$LOGDIR/val_export.log"
from pathlib import Path
from yolo_master_edge import YoloMasterDetectionPipeline, YoloMasterVariant
from yolo_master_edge.plot_metrics import plot_run_metrics

data = "/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/dataset/data.yaml"
run_dir = Path("/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/artifacts/yolo_master_runs/ft_n")
best = run_dir / "weights" / "best.pt"
pipe = YoloMasterDetectionPipeline(variant=YoloMasterVariant.V01_N)
pipe.load(model_spec=best)
metrics = pipe.val(data)
print("val metrics:", metrics)
if (run_dir / "results.csv").is_file():
    plot_run_metrics(run_dir)
onnx_path = pipe.export(format="onnx")
print("exported:", onnx_path)
PY

echo "[$(date)] pipeline DONE" | tee -a "$LOGDIR/pipeline.log"
