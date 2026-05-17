#!/usr/bin/env bash
set -e
EDGE=/mnt/d/dl/keyan/tiaozhanbei/yolo_master_edge
cd "${EDGE}"
URL="https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-N/YOLO-Master-v0.1-N.pt"
OUT="${EDGE}/yolo_master_n.pt"
if [ -f "${OUT}" ]; then
  echo "exists: ${OUT}"
  exit 0
fi
echo "Downloading YOLO-Master-v0.1-N -> yolo_master_n.pt"
wget -q --show-progress -O "${OUT}" "${URL}"
echo "done: ${OUT}"
