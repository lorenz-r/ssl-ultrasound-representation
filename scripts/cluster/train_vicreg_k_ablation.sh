#!/bin/bash
# VICReg K-ablation — the runs the other K columns need (WI-13).
#
# A K cell is a different *model*, not a different evaluation: K sets the stem's
# input channel count (`Conv2d(2K, 64, …)`), so K=16 and K=1 each need their own
# training run per policy. WI-12 trained K=4 only.
#
# PLAN.md §5.5's cut-list item 4 sanctions the reduced ladder — "drop K ∈ {8,4},
# keep K=16 and K=1" — so this is 4 runs, not 6. K=4 is already done and is not
# repeated here.
#
# Geometry per arm is *not* passed on the command line: `LoaderConfig.for_arm`
# derives it from K (§3.2c/d), so K=16 gets 64 px crops below z=8 mm and K=1
# gets 128 px at full depth. Passing --crop-size here would let K and crop size
# drift apart, which is the exact failure for_arm exists to prevent.
#
# Runs sequentially — one GPU, and two concurrent jobs would each want ~9 GB of
# host anon memory against a 24 GiB cgroup cap. Retries resume from last.pt, and
# rc=3 (deliberate collapse halt) is not retried; see train_vicreg_policies.sh.

set -u

REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
NFS_STORE=/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr
LOCAL_STORE=/tmp/wi8_fsa_k16.zarr
CKPT_ROOT=/mnt/home/benbarkow/checkpoints
LOG_DIR=/mnt/home/benbarkow/logs
MAX_ATTEMPTS=4

mkdir -p "$CKPT_ROOT" "$LOG_DIR"
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

stage_store() {
  local n
  n=$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)
  if [ ! -f "$LOCAL_STORE/frame_metadata.json" ] || [ "$n" -lt 923 ]; then
    echo "[kabl] staging store to $LOCAL_STORE ($n/923 chunks present)"
    rm -rf "$LOCAL_STORE"
    cp -r "$NFS_STORE" "$LOCAL_STORE" || return 1
  fi
  return 0
}

run_cell() {
  local k=$1 policy=$2
  local out="$CKPT_ROOT/wi13_vicreg_k${k}_policy${policy}"
  local log="$LOG_DIR/wi13_k${k}_policy${policy}.log"
  local attempt=1

  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "[kabl] $(date '+%F %T') K=$k policy=$policy attempt=$attempt/$MAX_ATTEMPTS"
    local store="$LOCAL_STORE"
    stage_store || { echo "[kabl] staging failed; falling back to NFS"; store="$NFS_STORE"; }

    python -u scripts/train/vicreg.py \
      --policy "$policy" \
      --k "$k" \
      --store "$store" \
      --out "$out" \
      --pair-budget 11.8e6 \
      --cache-frames 24 \
      --prefetch-factor 8 \
      --num-workers 4 \
      --checkpoint-every 25 \
      --collapse-patience 10 \
      --log-every-steps 50 \
      >> "$log" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ]; then
      echo "[kabl] $(date '+%F %T') K=$k policy=$policy COMPLETED"
      return 0
    fi
    if [ "$rc" -eq 3 ]; then
      echo "[kabl] $(date '+%F %T') K=$k policy=$policy HALTED (collapse) — not retrying"
      return 1
    fi
    echo "[kabl] $(date '+%F %T') K=$k policy=$policy exited rc=$rc — retrying (resumes from last.pt)"
    attempt=$((attempt + 1))
    sleep 30
  done
  echo "[kabl] $(date '+%F %T') K=$k policy=$policy GAVE UP after $MAX_ATTEMPTS attempts"
  return 1
}

echo "[kabl] === WI-13 K-ablation starting $(date '+%F %T') ==="
for k in 16 1; do
  for policy in A B; do
    run_cell "$k" "$policy"
    echo "[kabl] --- K=$k policy=$policy finished with status $? ---"
  done
done
echo "[kabl] === WI-13 K-ablation done $(date '+%F %T') ==="
