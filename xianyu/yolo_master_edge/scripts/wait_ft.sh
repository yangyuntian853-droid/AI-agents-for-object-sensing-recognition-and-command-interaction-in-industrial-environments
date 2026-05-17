#!/usr/bin/env bash
set -e
CSV=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/artifacts/yolo_master_runs/ft_n/results.csv
while true; do
  n=$(wc -l < "$CSV" 2>/dev/null || echo 1)
  ep=$((n - 1))
  echo "$(date +%H:%M:%S) ft_n epoch ${ep}/50"
  if [ "$ep" -ge 50 ]; then
    echo DONE_FT
    exit 0
  fi
  if ! pgrep -f 'train_custom.py --mode finetune' >/dev/null 2>&1; then
    echo STOPPED_FT ep=$ep
    exit 1
  fi
  sleep 120
done
