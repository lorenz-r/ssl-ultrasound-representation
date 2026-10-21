#!/bin/bash
# WI-19: per-pixel segmentation probe + mask figure, one column per K.
#
# Runs `scripts/eval/nvidia_segmentation.py` over the validation split for each K
# that has a trained pair on disk. Validation, not train, for the same reason
# the matrix columns default to it — those 93 frames are the ones the SSL
# encoders never saw — and here it is also the only split that fits in memory
# at full width: feature *maps* are ~2.3 MiB per crop across the nine arms, so
# 830 train frames x 8 crops is ~15 GiB against validation's 1.7 GiB.
#
# ⚠ Memory scales with (crops x channels), which `--frame-chunk` does NOT bound
# — chunking caps only the crop materialization, exactly as it does in
# `nvidia_matrix.py`. Use `--crops-per-frame` to control the feature footprint.
#
# See docs/cluster-segmentation-probe.md for the full operating notes.

set -u

REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
NFS_STORE=/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr
LOCAL_STORE=/tmp/wi8_fsa_k16.zarr
LABEL_MAPS=/mnt/home/benbarkow/stores/wi8_label_maps
CKPT_ROOT=/mnt/home/benbarkow/checkpoints
LOG_DIR=/mnt/home/benbarkow/logs

CROPS_PER_FRAME=${CROPS_PER_FRAME:-8}
SPLIT=${SPLIT:-validation}
KS=${KS:-"4 16 1"}

mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

if [ ! -f "$LABEL_MAPS/manifest.json" ]; then
  echo "[seg] no label maps at $LABEL_MAPS — run scripts/data/fetch_nvidia_label_maps.py first"
  exit 1
fi

# The local staged copy is ~9x faster and this is I/O-bound at crop sampling,
# same as the matrix columns.
STORE="$NFS_STORE"
if [ -f "$LOCAL_STORE/frame_metadata.json" ] &&
   [ "$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)" -ge 923 ]; then
  STORE="$LOCAL_STORE"
fi
echo "[seg] store=$STORE split=$SPLIT crops/frame=$CROPS_PER_FRAME"

rc_all=0
for k in $KS; do
  out="outputs/nvidia/k${k}_${SPLIT}_segmentation"
  log="$LOG_DIR/wi19_k${k}_${SPLIT}_segmentation.log"
  echo "[seg] $(date '+%F %T') K=$k -> $out"
  python -u scripts/eval/nvidia_segmentation.py \
    --k "$k" \
    --split "$SPLIT" \
    --crops-per-frame "$CROPS_PER_FRAME" \
    --store "$STORE" \
    --label-maps "$LABEL_MAPS" \
    --checkpoint-root "$CKPT_ROOT" \
    --out "$out" \
    > "$log" 2>&1
  rc=$?
  echo "[seg] $(date '+%F %T') K=$k rc=$rc"
  [ "$rc" -ne 0 ] && rc_all=$rc
done

echo "[seg] $(date '+%F %T') all columns done rc=$rc_all"
exit "$rc_all"
