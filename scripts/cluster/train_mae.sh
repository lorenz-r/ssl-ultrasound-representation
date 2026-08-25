#!/bin/bash
# Train the MAE baseline unattended (WI-14), with the same retry/resume
# discipline as the VICReg chains. One run per K: MAE masks rather than making
# view pairs, so there is no A/B policy split (see scripts/train/mae.py).
set -u
REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
NFS_STORE=/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr
LOCAL_STORE=/tmp/wi8_fsa_k16.zarr
CKPT_ROOT=/mnt/home/benbarkow/checkpoints
LOG_DIR=/mnt/home/benbarkow/logs
K=${1:-4}
MAX_ATTEMPTS=4

mkdir -p "$CKPT_ROOT" "$LOG_DIR"
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

n=$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)
if [ ! -f "$LOCAL_STORE/frame_metadata.json" ] || [ "$n" -lt 923 ]; then
  echo "[wi14] staging store ($n/923 chunks present)"
  rm -rf "$LOCAL_STORE"; cp -r "$NFS_STORE" "$LOCAL_STORE" || LOCAL_STORE="$NFS_STORE"
fi

echo "[wi14] === MAE K=$K starting $(date '+%F %T') ==="
attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "[wi14] $(date '+%F %T') attempt=$attempt/$MAX_ATTEMPTS"
  python -u scripts/train/mae.py --k "$K" --store "$LOCAL_STORE" \
    --out "$CKPT_ROOT/wi14_mae_k${K}" --checkpoint-every 25 --log-every-steps 50 \
    >> "$LOG_DIR/wi14_mae_k${K}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "[wi14] $(date '+%F %T') COMPLETED"; break; fi
  echo "[wi14] $(date '+%F %T') exited rc=$rc — retrying (resumes from last.pt)"
  attempt=$((attempt + 1)); sleep 30
done
echo "[wi14] === MAE K=$K done $(date '+%F %T') ==="
