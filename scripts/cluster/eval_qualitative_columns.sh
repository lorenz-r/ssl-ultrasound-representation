#!/bin/bash
# Qualitative columns (WI-13): encoder grids (PCA/t-SNE/UMAP) + per-encoder clustering.
#
#   bash scripts/cluster/eval_qualitative_columns.sh                # K=4, both splits
#   bash scripts/cluster/eval_qualitative_columns.sh 4:validation 16:validation
#
# Each argument is `K:split`. Default is the K=4 pair, which is the column the
# probe tables in docs/MEMORY.md quote and therefore the one the figures have to
# match. Crops-per-frame is left to the script, which derives it from the split
# (32 / 8) so the crops are the ones the probe column used.
#
# **Run this after `eval_anatomy_columns.sh`, not beside it.** t-SNE is CPU-bound
# and this box has 4 cores; sharing them with the probe chain's loaders starves
# both, and the symptom is a GPU that reads 0% while nothing is actually wrong
# (docs/MEMORY.md §6). It also wants the probe `results.json` to already exist — that
# is where the per-panel R² annotations come from.

set -u

REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
NFS_STORE=/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr
LOCAL_STORE=/tmp/wi8_fsa_k16.zarr
LOG_DIR=/mnt/home/benbarkow/logs
CACHE_DIR=/tmp/eval_features

mkdir -p "$LOG_DIR" "$CACHE_DIR"
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

STORE="$NFS_STORE"
if [ -f "$LOCAL_STORE/frame_metadata.json" ] &&
   [ "$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)" -ge 923 ]; then
  STORE="$LOCAL_STORE"
fi
echo "[qual] store=$STORE"

COLUMNS=("$@")
[ "${#COLUMNS[@]}" -eq 0 ] && COLUMNS=(4:validation 4:train)

rc_all=0
for col in "${COLUMNS[@]}"; do
  k="${col%%:*}"
  split="${col##*:}"
  out="outputs/nvidia/k${k}_${split}_qualitative"
  log="$LOG_DIR/wi13_k${k}_${split}_qual.log"
  # Cached on /tmp, not in the repo: it is ~50 MB of float32 per column and is
  # regenerable in a couple of minutes from the checkpoints.
  cache="$CACHE_DIR/k${k}_${split}.npz"
  echo "[qual] $(date '+%F %T') K=$k split=$split -> $out"
  python -u scripts/eval/nvidia_qualitative.py \
    --k "$k" \
    --split "$split" \
    --store "$STORE" \
    --feature-cache "$cache" \
    --out "$out" \
    > "$log" 2>&1
  rc=$?
  echo "[qual] $(date '+%F %T') K=$k split=$split rc=$rc"
  [ "$rc" -ne 0 ] && rc_all=$rc
done

echo "[qual] $(date '+%F %T') all qualitative columns done rc=$rc_all"
exit "$rc_all"
