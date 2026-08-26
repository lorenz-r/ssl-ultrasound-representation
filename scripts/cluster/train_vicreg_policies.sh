#!/bin/bash
# Run both VICReg arms (Policy A, then Policy B) unattended (WI-12).
#
# PLAN.md WI-12 specifies exactly two runs, differing only in augmentation
# policy. They run *sequentially*: one GPU, and two concurrent jobs would each
# want ~9 GB of host anon memory against a 24 GiB cgroup cap (plus page cache
# from the store), which OOM-kills both.
#
# Each arm is retried on failure. Retries are the point of the checkpointing in
# train/vicreg.py: a retry re-invokes the same command, which resumes from
# <out>/last.pt rather than restarting, so a crash at hour 5 costs at most the
# epochs since the last save, not the run.
#
# The store is read from local disk. The zarr lives on NFS (241 MB/s measured,
# and one epoch reads 830 x 26 MB = 21.6 GB), which starved the GPU at ~319
# pairs/s; the same store on the container's local scratch sustains ~488. /tmp
# is ephemeral per container, so this re-stages it if it went away.
#
# Throughput settings are measured, not guessed (see data/dataset.py for why
# the loader ships frames rather than crops). On this `24g` instance, sweeping
# workers/pinning/prefetch/cache at batch 256:
#
#   crop-shipping loader, the original         550 pairs/s   GPU 44%
#   + occupancy gathered device-side           724 pairs/s   GPU 50%
#   + frames shipped, GPU-resident cache      1570 pairs/s   GPU 92%
#
# cache-frames 24 measured faster than 16, 32 and 48 — below ~20 the cache
# spends too much of the batch admitting, above ~24 the gather loses locality
# for no benefit. That puts one arm at ~2.1 h rather than ~6, and keeps the
# GPU busy enough that an idle reaper has nothing to act on.

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
  # Re-stage if missing or obviously incomplete (923 frame chunks + metadata).
  local n
  n=$(ls "$LOCAL_STORE/fsa" 2>/dev/null | wc -l)
  if [ ! -f "$LOCAL_STORE/frame_metadata.json" ] || [ "$n" -lt 923 ]; then
    echo "[night] staging store to $LOCAL_STORE ($n/923 chunks present)"
    rm -rf "$LOCAL_STORE"
    cp -r "$NFS_STORE" "$LOCAL_STORE" || return 1
  fi
  return 0
}

run_arm() {
  local policy=$1
  local out="$CKPT_ROOT/wi12_vicreg_policy${policy}"
  local log="$LOG_DIR/wi12_policy${policy}.log"
  local attempt=1

  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "[night] $(date '+%F %T') policy=$policy attempt=$attempt/$MAX_ATTEMPTS"
    local store="$LOCAL_STORE"
    stage_store || { echo "[night] staging failed; falling back to NFS"; store="$NFS_STORE"; }

    python -u scripts/train/vicreg.py \
      --policy "$policy" \
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
      echo "[night] $(date '+%F %T') policy=$policy COMPLETED"
      return 0
    fi
    # rc=3 is the deliberate dimensional-collapse halt: re-running would resume
    # the collapsed weights and halt again, so stop and leave it for a human.
    # It is deliberately *not* rc=1 — python exits 1 on any uncaught exception,
    # so the two were indistinguishable and a real crash (the `strict=True`
    # ValueError in _is_collapsing) got logged as "HALTED (collapse)" and
    # skipped rather than retried.
    if [ "$rc" -eq 3 ]; then
      echo "[night] $(date '+%F %T') policy=$policy HALTED (collapse) — not retrying"
      return 1
    fi
    echo "[night] $(date '+%F %T') policy=$policy exited rc=$rc — retrying (resumes from last.pt)"
    attempt=$((attempt + 1))
    sleep 30
  done
  echo "[night] $(date '+%F %T') policy=$policy GAVE UP after $MAX_ATTEMPTS attempts"
  return 1
}

echo "[night] === WI-12 both arms starting $(date '+%F %T') ==="
run_arm A
echo "[night] --- policy A finished with status $? ---"
run_arm B
echo "[night] --- policy B finished with status $? ---"
echo "[night] === WI-12 done $(date '+%F %T') ==="
