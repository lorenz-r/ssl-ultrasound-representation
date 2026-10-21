#!/bin/bash
# VICReg on a ViT-S backbone: the missing (objective x backbone) cell (WI-18).
#
# docs/MEMORY.md §16.7: I-JEPA beat every baseline on the WI-13 matrix, but it is a
# different backbone *and* a different objective from the VICReg arms, so §16
# cannot say which produced the result. This run holds the objective fixed at
# VICReg Policy A and swaps only the backbone, giving two clean readings:
#   vicreg_A     -> vicreg_vit_A   isolates the backbone
#   vicreg_vit_A -> ijepa          isolates the objective
#
# Three hyperparameters deviate from the ResNet arms, all measured or argued:
#
#   --batch-size 192  Batch 256 (the ResNet arms' value) OOMs in the real loop:
#                     the synthetic forward/backward peaks at 21.8 GiB and the
#                     resident GpuFrameCache pushes it past 24. 192 measured at
#                     18.2 GiB. VICReg's variance/covariance terms are batch
#                     statistics over a 2048-d projector, so the batch cannot be
#                     dropped to I-JEPA's 64 without crippling the objective —
#                     192 keeps the same estimator regime as 256.
#   --weight-decay 0.05 / --lr 1e-3
#                     The 1e-6 / 3e-4 defaults are the ResNet recipe. These are
#                     I-JEPA's values, so the two *ViT* arms share an optimizer
#                     and the objective comparison is not confounded by it.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is what keeps 192 clear of
# the fragmentation cliff; without it the same config sits much closer to OOM.
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

n=$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)
if [ ! -f "$LOCAL_STORE/frame_metadata.json" ] || [ "$n" -lt 923 ]; then
  echo "[wi18] staging store ($n/923 chunks present)"
  rm -rf "$LOCAL_STORE"; cp -r "$NFS_STORE" "$LOCAL_STORE" || LOCAL_STORE="$NFS_STORE"
fi

echo "[wi18] === VICReg ViT-S K=$K policy A starting $(date '+%F %T') ==="
attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "[wi18] $(date '+%F %T') attempt=$attempt/$MAX_ATTEMPTS"
  python -u scripts/train/vicreg.py --policy A --k "$K" --backbone vit_s \
    --store "$LOCAL_STORE" --out "$CKPT_ROOT/wi18_vicreg_vit_k${K}_policyA" \
    --batch-size 192 --lr 1e-3 --weight-decay 0.05 --cache-frames 24 \
    --checkpoint-every 25 --log-every-steps 50 \
    >> "$LOG_DIR/wi18_vicreg_vit_k${K}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "[wi18] $(date '+%F %T') COMPLETED"; break; fi
  echo "[wi18] $(date '+%F %T') exited rc=$rc — retrying (resumes from last.pt)"
  attempt=$((attempt + 1)); sleep 30
done
echo "[wi18] === VICReg ViT-S K=$K done $(date '+%F %T') ==="
echo "=== WI-18 chain exited $(date '+%F %T')" >> "$LOG_DIR/train_vicreg_vit_sentinel.log"
