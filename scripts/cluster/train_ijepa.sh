#!/bin/bash
# Train the I-JEPA arm unattended (WI-17), with the same retry/resume
# discipline as the MAE/VICReg chains (scripts/cluster/train_mae.sh). One run per K.
set -u
REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
NFS_STORE=/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr
LOCAL_STORE=/tmp/wi8_fsa_k16.zarr
CKPT_ROOT=/mnt/home/benbarkow/checkpoints
LOG_DIR=/mnt/home/benbarkow/logs
K=${1:-4}
MAX_ATTEMPTS=6

mkdir -p "$CKPT_ROOT" "$LOG_DIR"
cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

n=$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)
if [ ! -f "$LOCAL_STORE/frame_metadata.json" ] || [ "$n" -lt 923 ]; then
  echo "[wi17] staging store ($n/923 chunks present)"
  rm -rf "$LOCAL_STORE"; cp -r "$NFS_STORE" "$LOCAL_STORE" || LOCAL_STORE="$NFS_STORE"
fi

echo "[wi17] === I-JEPA K=$K starting $(date '+%F %T') ==="
attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "[wi17] $(date '+%F %T') attempt=$attempt/$MAX_ATTEMPTS"
  python -u scripts/train/ijepa.py --k "$K" --store "$LOCAL_STORE" \
    --out "$CKPT_ROOT/wi17_ijepa_k${K}" --checkpoint-every 25 --log-every-steps 50 \
    >> "$LOG_DIR/wi17_ijepa_k${K}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "[wi17] $(date '+%F %T') COMPLETED"; break; fi
  if [ "$rc" -eq 2 ]; then echo "[wi17] $(date '+%F %T') HALTED (collapse) — not retrying"; break; fi
  echo "[wi17] $(date '+%F %T') exited rc=$rc — retrying (resumes from last.pt)"
  attempt=$((attempt + 1)); sleep 30
done
echo "[wi17] === I-JEPA K=$K done $(date '+%F %T') ==="
