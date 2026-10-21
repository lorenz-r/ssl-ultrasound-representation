#!/bin/bash
# The §15 geometry control, re-run so it covers the I-JEPA column (WI-17).
# I-JEPA wins z_centre/x_centre by a wide margin and trained without any
# augmentation, so it never had position deliberately destroyed the way Policy
# A/B did. That makes "how much of this is just where the crop came from"
# the first question its numbers raise, and §15 is the instrument for it.
set -u
REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
LOG=/mnt/home/benbarkow/logs/eval_geometry_control.log
SENT=/mnt/home/benbarkow/logs/eval_geometry_control_sentinel.log
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate
for split in validation train; do
  echo "[geom] $(date '+%F %T') START $split" >> "$LOG"
  python -u scripts/eval/geometry_control.py --k 4 --split "$split" \
    --out "outputs/nvidia/geometry_k4_${split}" >> "$LOG" 2>&1
  echo "[geom] $(date '+%F %T') END $split rc=$?" >> "$LOG"
  [ -f "outputs/nvidia/geometry_k4_${split}/results.json" ] && \
    cp "outputs/nvidia/geometry_k4_${split}/results.json" "results/nvidia/geometry_k4_${split}.json" && \
    echo "[geom] copied $split" >> "$LOG"
done
echo "[geom] === done $(date '+%F %T') ===" >> "$LOG"
echo "=== WI-17 geom chain exited $(date '+%F %T')" >> "$SENT"
