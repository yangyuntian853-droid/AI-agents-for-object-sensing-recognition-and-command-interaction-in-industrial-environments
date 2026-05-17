#!/usr/bin/env bash
set -e
CSV=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge/artifacts/yolo_master_runs/scratch_n/results.csv
while true; do
  if [ -f "$CSV" ]; then
    n=$(wc -l < "$CSV")
    ep=$((n - 1))
  else
    ep=0
  fi
  echo "$(date +%H:%M:%S) scratch_n epoch ${ep}/100"
  if [ "$ep" -ge 100 ]; then
    echo DONE_SCRATCH
    exit 0
  fi
  if ! pgrep -f 'train_custom.py --mode scratch' >/dev/null 2>&1; then
    if [ "$ep" -ge 100 ]; then
      echo DONE_SCRATCH
      exit 0
    fi
    echo STOPPED_SCRATCH ep=$ep
    exit 1
  fi
  sleep 180
done
