#!/bin/bash
# WI-13 — re-run every results-matrix column with the crop-level anatomy targets.
#
# The three §4.1 anatomy axes (`contains_cyst`, `cyst_fraction_crop`,
# `sos_mean_crop`) sat in `not_run` until the per-pixel maps were fetched
# (MEMORY.md §9) and wired into `wi13_matrix.py` (§11). This runs the six
# columns — K in {4,16,1} x {validation, train} — that fill them in.
#
# **Crops-per-frame is per split, not a preference**: 32 for validation (93
# frames) and 8 for train (830), which is what every committed column used.
# Changing it would change which crops each encoder sees and break the
# reproduction check below. Everything else is left at its default, including
# `--seed 0` and the K-derived crop geometry (`LoaderConfig.for_arm`).
#
# Output goes to `figures/wi13_k{K}_{split}_anatomy`, *beside* the existing
# columns rather than over them: the seven old targets must come back bitwise
# identical, and `scripts/wi13_anatomy_verify.py` checks that against the
# committed `tests/data/wi13_*.json` before anything is replaced.
#
# Reads the local staged store when it is complete, NFS otherwise (the local
# copy is ~9x faster and this is I/O-bound at the crop-sampling step).

set -u

REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
NFS_STORE=/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr
LOCAL_STORE=/tmp/wi8_fsa_k16.zarr
LABEL_MAPS=/mnt/home/benbarkow/stores/wi8_label_maps
LOG_DIR=/mnt/home/benbarkow/logs

mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

STORE="$NFS_STORE"
if [ -f "$LOCAL_STORE/frame_metadata.json" ] &&
   [ "$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)" -ge 923 ]; then
  STORE="$LOCAL_STORE"
fi
echo "[anat] store=$STORE"

rc_all=0
for k in 4 16 1; do
  for split in validation train; do
    case "$split" in
      validation) cpf=32 ;;
      train)      cpf=8  ;;
    esac
    out="figures/wi13_k${k}_${split}_anatomy"
    log="$LOG_DIR/wi13_k${k}_${split}_anatomy.log"
    echo "[anat] $(date '+%F %T') K=$k split=$split crops/frame=$cpf -> $out"
    python -u scripts/wi13_matrix.py \
      --k "$k" \
      --split "$split" \
      --crops-per-frame "$cpf" \
      --store "$STORE" \
      --label-maps "$LABEL_MAPS" \
      --out "$out" \
      > "$log" 2>&1
    rc=$?
    echo "[anat] $(date '+%F %T') K=$k split=$split rc=$rc"
    [ "$rc" -ne 0 ] && rc_all=$rc
  done
done

echo "[anat] $(date '+%F %T') all columns done rc=$rc_all"
exit "$rc_all"
